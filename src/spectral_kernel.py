from abc import ABC, abstractmethod

import torch
# import gpytorch

from src.spectral_measure import AbstractSpectralMeasure
from src.space import AbstractManifold
from src.utils import cartesian_prod
from gpytorch.lazy import NonLazyTensor, MatmulLazyTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.complex64


class AbstractSpectralKernel(torch.nn.Module, ABC):
    def __init__(self, measure: AbstractSpectralMeasure, manifold: AbstractManifold):
        super().__init__()
        self.measure = measure
        self.manifold = manifold

    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError


class EigenbasisSumKernel(AbstractSpectralKernel):
    def __init__(self, measure, manifold):
        super().__init__(measure, manifold)

        point = manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y=None, normalize=True):
        if y is None:
            y = x
        x_yinv = self.manifold.pairwise_diff(x, y)
        cov = torch.zeros(len(x), len(y), dtype=dtype, device=device)
        for eigenspace in self.manifold.lb_eigenspaces:
            lmd = eigenspace.lb_eigenvalue
            f = eigenspace.basis_sum
            cov += self.measure(lmd) * f(x_yinv).view(x.size()[0], y.size()[0])
        if normalize:
            return cov.real/self.normalizer
        else:
            return cov.real


class EigenbasisKernel(AbstractSpectralKernel):
    def __init__(self, measure, manifold):
        super().__init__(measure, manifold)
        point = self.manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y=None, normalize=True):
        if y is None:
            y = x
        cov = torch.zeros(len(x), len(y))
        for eigenspace in self.manifold.lb_eigenspaces:
            lmd = eigenspace.lb_eigenvalue
            f = eigenspace.basis
            f_x, f_y = f(x).T, f(y).T
            cov += self.measure(lmd) * (f_x @ f_y.T)
        if normalize:
            return cov/self.normalizer
        else:
            return cov


class RandomSpectralKernel(AbstractSpectralKernel):
    """Generalization of Random fourier features method"""
    def __init__(self, measure, manifold):
        super().__init__(measure, manifold)
        manifold.generate_lb_eigenspaces(measure)  # Generate lb_eigenvalues with respect to spectral measure
        point = self.manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def compute_normalizer(self):
        point = self.manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y=None, normalize=True):
        """We don't need summation because manifold.lb_eigenspaces is already vectorized"""
        if self.training:
            self.manifold.generate_lb_eigenspaces(self.measure)
            if normalize:
                self.compute_normalizer()

        if y is None:
            y = x
        x_, y_ = self.manifold.to_group(x), self.manifold.to_group(y)
        x_yinv = self.manifold.pairwise_diff(x_, y_)
        f = self.manifold.lb_eigenspaces
        x_yinv_embed = f(x_yinv)  # (n*m,order)
        eye_embed = f(self.manifold.id)  # (1, order)
        cov_flatten = x_yinv_embed @ (torch.conj(eye_embed).T)
        cov = cov_flatten.view(x.size()[0], y.size()[0])
        if normalize:
            return self.measure.variance[0] * cov.real / self.normalizer
        else:
            return cov.real

class RandomFourierFeatureKernel(torch.nn.Module):
    def __init__(self, kernel: RandomSpectralKernel):
        super().__init__()

        self.kernel = kernel

    def cov(self, x, y=None):
        if self.training:
            self.kernel.manifold.generate_lb_eigenspaces(self.kernel.measure)
            self.kernel.compute_normalizer()
        if y is None:
            y = x
        x_, y_ = self.kernel.manifold.to_group(x), self.kernel.manifold.to_group(y)
        x_embed, y_embed = self.kernel.manifold.lb_eigenspaces(x_), self.kernel.manifold.lb_eigenspaces(y_)
        return self.kernel.measure.variance[0] *\
                (x_embed @ (torch.conj(y_embed).T)).real/self.kernel.normalizer

    def forward(self, x):
        if self.training:
            self.kernel.manifold.generate_lb_eigenspaces(self.kernel.measure)
            self.kernel.compute_normalizer()
        x_ = self.kernel.manifold.to_group(x)
        x_embed = torch.sqrt(torch.abs(self.kernel.measure.variance[0]))*self.kernel.manifold.lb_eigenspaces(x_)
        x_embed = x_embed/torch.sqrt(self.kernel.normalizer)
        x_embed_real = x_embed.real.clone()
        x_embed_imag = x_embed.imag.clone()
        x_embed_lazy = NonLazyTensor(torch.cat((x_embed_real, x_embed_imag), dim=-1))
        return MatmulLazyTensor(x_embed_lazy, x_embed_lazy.t().clone())
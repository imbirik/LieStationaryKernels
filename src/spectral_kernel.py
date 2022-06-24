from abc import ABC, abstractmethod

import torch
# import gpytorch

from src.spectral_measure import AbstractSpectralMeasure
from src.space import AbstractManifold
from src.utils import cartesian_prod

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.complex128


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

    def forward(self, x, y, normalize=True):
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

    def forward(self, x, y, normalize=True):
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


class RandomFourierFeaturesKernel(AbstractSpectralKernel):
    """Generalization of Random fourier features method"""
    def __init__(self, measure, manifold):
        super().__init__(measure, manifold)
        manifold.generate_lb_eigenspaces(measure)  # Generate lb_eigenvalues with respect to spectral measure
        point = manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y, normalize=True):
        """We don't need summation because manifold.lb_eigenspaces is already vectorized"""
        x_, y_ = self.manifold.to_group(x), self.manifold.to_group(y)
        x_yinv = self.manifold.pairwise_diff(x_, y_)
        f = self.manifold.lb_eigenspaces
        x_yinv_embed = f(x_yinv)  # (n*m,order)
        eye_embed = f(self.manifold.id)  # (1, order)
        cov_flatten = x_yinv_embed @ (torch.conj(eye_embed).T)
        cov = cov_flatten.view(x.size()[0], y.size()[0])
        if normalize:
            return cov.real / self.normalizer
        else:
            return cov.real

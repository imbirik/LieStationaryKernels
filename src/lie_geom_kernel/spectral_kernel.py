from abc import ABC, abstractmethod

import torch
import math
from itertools import islice

from lie_geom_kernel.spectral_measure import AbstractSpectralMeasure
from lie_geom_kernel.space import AbstractManifold
from gpytorch.lazy import NonLazyTensor, MatmulLazyTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64


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

    def compute_normalizer(self):
        point = self.manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y=None, normalize=True):
        if y is None:
            y = x

        if self.training:
            if normalize:
                self.compute_normalizer()

        x_y_embed = self.manifold.pairwise_embed(x, y)
        cov = torch.zeros(len(x), len(y), dtype=dtype, device=device)
        for eigenspace in self.manifold.lb_eigenspaces:
            lmd = eigenspace.lb_eigenvalue
            f = eigenspace.phase_function
            cov += self.measure(lmd) * f(x_y_embed).view(x.size()[0], y.size()[0]).real
        if normalize:
            return torch.abs(self.measure.variance[0]) * cov/self.normalizer
        else:
            return cov.real


class EigenbasisKernel(AbstractSpectralKernel):
    def __init__(self, measure, manifold):
        super().__init__(measure, manifold)
        point = self.manifold.id
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


class RandomPhaseKernel(AbstractSpectralKernel):
    def __init__(self, measure, manifold, phase_order=100):
        super().__init__(measure, manifold)

        self.approx_order = manifold.order
        self.phase_order = phase_order
        self.phases = self.sample_phases()

        point = self.manifold.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def compute_normalizer(self):
        point = self.manifold.id
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def sample_phases(self):
        return self.manifold.rand(self.phase_order)

    def make_embedding(self, x):
        embeddings = []
        phases = self.phases  # [num_phase, ...]
        # left multiplication
        phase_x_inv = self.manifold.pairwise_embed(phases, x)  # [len(x), num_phase, ...]

        for i, eigenspace in enumerate(islice(self.manifold.lb_eigenspaces, self.approx_order)):
            lmd = eigenspace.lb_eigenvalue
            f = eigenspace.phase_function
            eigen_embedding = f(phase_x_inv).real.view(self.phase_order, x.size()[0]).T
            eigen_embedding = torch.sqrt(self.measure(lmd)) * eigen_embedding
            eigen_embedding = eigen_embedding / math.sqrt(self.phase_order)
            embeddings.append(eigen_embedding)
        return torch.cat(embeddings, dim=1)

    def forward(self, x, y=None, normalize=True):
        if y is None:
            y = x

        if self.training:
            if normalize:
                self.compute_normalizer()

        x_embed, y_embed = self.make_embedding(x), self.make_embedding(y)

        if normalize:
            x_embed = torch.sqrt(torch.abs(self.measure.variance[0])) * x_embed / torch.sqrt(self.normalizer)
            y_embed = torch.sqrt(torch.abs(self.measure.variance[0])) * y_embed / torch.sqrt(self.normalizer)

        x_embed_lazy = NonLazyTensor(x_embed)
        y_embed_lazy = NonLazyTensor(y_embed)
        return MatmulLazyTensor(x_embed_lazy, y_embed_lazy.t().clone())


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


class RandomFourierFeatureKernel(AbstractSpectralKernel):
    def __init__(self, measure, manifold):
        super().__init__(measure, manifold)
        manifold.generate_lb_eigenspaces(measure)  # Generate lb_eigenvalues with respect to spectral measure
        point = self.manifold.id
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def compute_normalizer(self):
        point = self.manifold.id
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y=None, normalize=True):
        if self.training:
            self.manifold.generate_lb_eigenspaces(self.measure)
            if normalize:
                self.compute_normalizer()

        if y is None:
            y = x

        x_, y_ = self.manifold.to_group(x), self.manifold.to_group(y)
        x_embed = self.manifold.lb_eigenspaces(x_)
        y_embed = self.manifold.lb_eigenspaces(y_)

        if normalize:
            x_embed = torch.sqrt(torch.abs(self.measure.variance[0]))*x_embed/torch.sqrt(self.normalizer)
            y_embed = torch.sqrt(torch.abs(self.measure.variance[0])) * y_embed / torch.sqrt(self.normalizer)

        x_embed_real, x_embed_imag = x_embed.real.clone(), x_embed.imag.clone()
        y_embed_real, y_embed_imag = y_embed.real.clone(), y_embed.imag.clone()

        x_embed_lazy = NonLazyTensor(torch.cat((x_embed_real, x_embed_imag), dim=-1))
        y_embed_lazy = NonLazyTensor(torch.cat((y_embed_real, y_embed_imag), dim=-1))
        return MatmulLazyTensor(x_embed_lazy, y_embed_lazy.t().clone())
import torch
import warnings
from itertools import islice
# from functorch import vmap
from src.utils import cartesian_prod
from math import sqrt
from src.spectral_kernel import EigenSpaceKernel, EigenFunctionKernel


class KarhunenLoeveExpansion(torch.nn.Module):
    def __init__(self, kernel: EigenSpaceKernel, approx_order=None):
        super().__init__()

        if approx_order is None:
            approx_order = sum(kernel.space.lb_eigenspaces_dims)

        if sum(kernel.space.lb_eigenspaces_dims) < approx_order:
            raise ValueError("approx_order must be lower or equal then number of prepared eigenfunctions")

        self.num_eigenspaces = 0
        new_order = 0

        while new_order < approx_order:
            new_order += kernel.space.lb_eigenspaces_dims[self.num_eigenspaces]
            self.num_eigenspaces += 1

        if self.num_eigenspaces != approx_order:
            warnings.warn('Warning approximation order was increased. '
                          '\n New order is {}'.format(new_order))

        self.kernel = kernel
        self.approx_order = new_order
        self.weights = self.sample_weights()

    def sample_weights(self):
        return torch.randn(self.approx_order)

    def forward(self, x):
        res = []
        for lmd, f in islice(zip(self.space.lb_eigenvalues, self.space.lb_eigenbases), self.num_eigenspaces):
            res.append(self.measure(lmd) * f(x).T)
        res = torch.cat(res, 1)  # [len(x), approx_order]
        res = torch.einsum('ij,j->i', res, self.weights) # [len(x)]
        return res


class RandomPhaseApproximation(torch.nn.Module):
    def __init__(self, kernel: EigenFunctionKernel, approx_order=None, phase_order=1000):
        super().__init__()

        self.kernel = kernel

        if approx_order is None:
            approx_order = len(self.kernel.space.lb_eigenvalues)

        if approx_order < len(self.kernel.space.lb_eigenvalues):
            raise ValueError("number of computed eigenfunctions of space must be greater than approx_order")
        self.approx_order = approx_order
        self.phase_order = phase_order
        self.weights = self.sample_weights()
        self.phases = self.sample_phases()

    def sample_weights(self):
        return torch.randn(self.phase_order*self.approx_order)

    def sample_phases(self):
        return [self.kernel.space.rand(self.phase_order) for _ in range(self.approx_order)]

    def resample(self):
        self.weights = self.sample_weights()
        self.phases = self.sample_phases()

    def make_embedding(self, x):
        embeddings = []
        for i in range(self.approx_order):
            lmd, f = self.kernel.space.lb_eigenvalues[i], self.kernel.space.lb_eigenbases_sums[i]
            phase, weight = self.phases[i], self.weights[i]  # [num_phase, ...], [num_phase]

            x_, phase_ = cartesian_prod(x, phase)  # [len(x), num_phase, ...]

            eigen_embedding = torch.sqrt(self.kernel.measure(lmd)) * f(x_, phase_)
            eigen_embedding = eigen_embedding / torch.sqrt(self.kernel.normalizer) / sqrt(self.phase_order)
            embeddings.append(eigen_embedding)
        return torch.cat(embeddings, dim=1)

    def forward(self, x):  # [N, ...]
        embedding = self.make_embedding(x)
        random_embedding = torch.einsum('nm,m->n', embedding, self.weights)  # [len(x), phase_order* approx_order, ...]
        return random_embedding.real

    def _cov(self, x, y):
        x_embed, y_embed = self.make_embedding(x), self.make_embedding(y)
        return (x_embed @ torch.conj(y_embed.T)).real



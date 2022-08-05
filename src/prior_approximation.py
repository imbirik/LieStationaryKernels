import torch
import warnings
from itertools import islice
from math import sqrt
from src.spectral_kernel import EigenbasisKernel, EigenbasisSumKernel, RandomSpectralKernel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64


class KarhunenLoeveExpansion(torch.nn.Module):
    def __init__(self, kernel: EigenbasisKernel, approx_order=None):
        super().__init__()

        if approx_order is None:
            approx_order = sum(kernel.manifold.lb_eigenspaces)

        if sum(kernel.manifold.lb_eigenspaces) < approx_order:
            raise ValueError("approx_order must be lower or equal then the number of prepared eigenfunctions")

        self.num_eigenspaces = 0
        new_order = 0

        while new_order < approx_order:
            new_order += kernel.manifold.lb_eigenspaces[self.num_eigenspaces].lb_eigenvalue
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
        for eigenspace in islice(self.space.lb_eigenspaces, self.num_eigenspaces):
            lmd = eigenspace.lb_eigenvalue
            f = eigenspace.basis
            res.append(self.measure(lmd) * f(x).T)
        res = torch.cat(res, 1)  # [len(x), approx_order]
        res = torch.einsum('ij,j->i', res, self.weights) # [len(x)]
        return res


class RandomPhaseApproximation(torch.nn.Module):
    def __init__(self, kernel: EigenbasisSumKernel, phase_order=1000):
        super().__init__()

        self.kernel = kernel
        self.approx_order = len(self.kernel.manifold.lb_eigenspaces)
        self.phase_order = phase_order
        self.weights = self.sample_weights()
        self.phases = self.sample_phases()

    def sample_weights(self):
        return torch.randn(self.phase_order*self.approx_order, device=device, dtype=dtype)

    def sample_phases(self):
        return self.kernel.manifold.rand(self.phase_order)

    def resample(self):
        self.weights = self.sample_weights()
        self.phases = self.sample_phases()

    def make_embedding(self, x):
        embeddings = []
        phases = self.phases  # [num_phase, ...]
        # left multiplication
        phase_x_inv = self.kernel.manifold.pairwise_embed(phases, x)  # [len(x), num_phase, ...]

        for i, eigenspace in enumerate(islice(self.kernel.manifold.lb_eigenspaces, self.approx_order)):
            lmd = eigenspace.lb_eigenvalue
            f = eigenspace.phase_function
            eigen_embedding = f(phase_x_inv).real.view(self.phase_order, x.size()[0]).T
            eigen_embedding = torch.sqrt(self.kernel.measure.variance[0] * self.kernel.measure(lmd)) * eigen_embedding
            eigen_embedding = eigen_embedding / torch.sqrt(self.kernel.normalizer) / sqrt(self.phase_order)
            embeddings.append(eigen_embedding)
        return torch.cat(embeddings, dim=1)

    def forward(self, x):  # [N, ...]
        embedding = self.make_embedding(x)
        random_embedding = torch.einsum('nm,m->n', embedding, self.weights)  # [len(x), phase_order* approx_order, ...]
        return random_embedding

    def _cov(self, x, y):
        x_embed, y_embed = self.make_embedding(x), self.make_embedding(y)
        return (x_embed @ torch.conj(y_embed.T))


class RandomFourierApproximation(torch.nn.Module):
    def __init__(self, kernel: RandomSpectralKernel):
        super().__init__()

        self.kernel = kernel
        self.weights_real = self.sample_weights()
        self.weights_imag = self.sample_weights()

    def sample_weights(self):
        return torch.randn(self.kernel.manifold.order, dtype=dtype, device=device)

    def resample(self):
        self.weights_real = self.sample_weights()
        self.weights_imag = self.sample_weights()
        self.kernel.manifold.generate_lb_eigenspaces(self.kernel.measure)

    def forward(self, x):  # [N, ...]
        x_ = self.kernel.manifold.to_group(x)
        embedding = self.kernel.manifold.lb_eigenspaces(x_) * torch.sqrt(self.kernel.measure.variance[0])
        sample_real = torch.einsum('nm,m->n', embedding.real, self.weights_real)
        sample_imag = torch.einsum('nm,m->n', embedding.imag, self.weights_imag)
        sample = (sample_real-sample_imag)/sqrt(self.kernel.normalizer)
        return sample

    def _cov(self, x, y):
        x_, y_ = self.kernel.manifold.to_group(x), self.kernel.manifold.to_group(y)
        x_embed, y_embed = self.kernel.manifold.lb_eigenspaces(x_), self.kernel.manifold.lb_eigenspaces(y_)
        # print("max of x_embed:", torch.max(torch.abs(x_embed)))
        return self.kernel.measure.variance[0] * (x_embed @ (torch.conj(y_embed).T)).real/self.kernel.normalizer

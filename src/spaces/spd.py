import torch
from torch.nn import Parameter
from pyro.distributions.rejector import Rejector

from src.space import NonCompactSymmetricSpace, NonCompactSymmetricSpaceExp
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
from src.utils import GOE, StudentGOE, triu_ind



dtype = torch.double
j = torch.tensor([1j]).item()  # imaginary unit
pi = 2*torch.acos(torch.zeros(1)).item()


class SymmetricPositiveDefiniteMatrices(NonCompactSymmetricSpace):
    """Class of Positive definite matrices represented as symmetric space GL(n,R)/O(n,R)"""

    def __init__(self, dim: int, order: int):
        super(SymmetricPositiveDefiniteMatrices, self).__init__()
        self.dim = dim
        self.order = order
        self.id = torch.eye(self.dim, dtype=dtype).view(-1, self.dim, self.dim)

        #self.lb_eigenspaces = None

    def generate_lb_eigenspaces(self, measure):
        shift = self.rand_phase(self.order)
        if isinstance(measure, MaternSpectralMeasure):
            nu, lengthscale = measure.nu, measure.lengthscale
            lmd = SPDMaternSpectralMeasureSampler(self.dim, lengthscale, nu)((self.order,))
        elif isinstance(measure, SqExpSpectralMeasure):
            lengthscale = measure.lengthscale
            lmd = SPDSqExpSpectralMeasureSampler(self.dim, lengthscale)((self.order,))
        else:
            return NotImplementedError
        self.lb_eigenspaces = SPDShiftedExp(2*lmd, shift, self)

    def to_group(self, x):
        return torch.linalg.cholesky(x, upper=True)

    def rand_phase(self, n=1):
        qr = torch.randn((n, self.dim, self.dim), dtype=dtype)
        q, r = torch.linalg.qr(qr)
        r_diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        q *= r_diag_sign[:, None]
        q_det_sign = torch.sign(torch.det(q))
        q[:, :, 0] *= q_det_sign[:, None]
        return q

    def rand(self, n=1):
        """Note, there is no standard method to sample from SPD"""
        rand = torch.randn(n, self.dim, self.dim, dtype=dtype)
        rand_pos = torch.bmm(rand, torch.transpose(rand, -2, -1))
        return rand_pos

    def inv(self, x):
        return torch.linalg.inv(x)


class SPDShiftedExp(NonCompactSymmetricSpaceExp):
        def __init__(self, lmd, shift, manifold):
            super().__init__(lmd=lmd, shift=shift, manifold=manifold)

        def compute_rho(self):
            rho = torch.tensor([(i + 1) - (self.manifold.dim + 1) / 2 for i in range(self.manifold.dim)], dtype=dtype)
            return rho

        def iwasawa_decomposition(self, x):
            h, an = torch.linalg.qr(x, mode='complete')

            diag_sign = torch.diag_embed(torch.diagonal(torch.sign(an), dim1=-2, dim2=-1))
            h = torch.bmm(h, diag_sign)
            an = torch.bmm(diag_sign, an)

            a = torch.diagonal(an, dim1=-2, dim2=-1)

            a_inv = torch.div(torch.ones_like(a), a)
            a_inv = torch.diag_embed(a_inv)

            n = torch.bmm(a_inv, an)

            return h, a, n


class SPDAbstractSpectralMeasureSampler(torch.nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(SPDAbstractSpectralMeasureSampler, self).__init__()

    def log_accept(self, lmd):
        lmd_diff = (lmd[:, None, :] - lmd[:, :, None])[triu_ind(lmd.size()[0], self.dim, 1)].\
            reshape(-1, self.dim * (self.dim - 1) // 2)
        log_accept = pi * torch.abs(lmd_diff)
        log_accept = torch.tanh(log_accept)
        log_accept = torch.sum(torch.log(log_accept), dim=1)
        return log_accept

    def forward(self, shape):
        return Rejector(self.raw_sampler, self.log_accept, 0).rsample(shape)


class SPDMaternSpectralMeasureSampler(SPDAbstractSpectralMeasureSampler):
    def __init__(self, dim, lengthscale, nu):
        super(SPDMaternSpectralMeasureSampler, self).__init__(dim=dim)
        self.lengthscale = Parameter(torch.tensor([lengthscale], dtype=dtype))
        self.nu = Parameter(torch.tensor([nu], dtype=dtype))
        c = (self.dim ** 3 - self.dim) / 48
        self.raw_sampler = StudentGOE(self.dim, self.lengthscale[0], self.nu[0], c)


class SPDSqExpSpectralMeasureSampler(SPDAbstractSpectralMeasureSampler):
    def __init__(self, dim, lengthscale):
        super(SPDSqExpSpectralMeasureSampler, self).__init__(dim=dim)
        self.lengthscale = Parameter(torch.tensor([lengthscale], dtype=dtype))
        self.raw_sampler = GOE(self.dim, self.lengthscale[0])


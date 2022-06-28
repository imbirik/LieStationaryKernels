import torch
import numpy as np
from src.space import NonCompactSymmetricSpace, NonCompactSymmetricSpaceExp
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
from src.utils import GOE_sampler, triu_ind
from math import factorial

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

dtype = torch.float64
j = torch.tensor([1j], device=device, dtype=torch.complex128).item()  # imaginary unit
pi = 2*torch.acos(torch.zeros(1)).item()


class SymmetricPositiveDefiniteMatrices(NonCompactSymmetricSpace):
    """Class of Positive definite matrices represented as symmetric space GL(n,R)/O(n,R)"""

    def __init__(self, n: int, order: int):
        super(SymmetricPositiveDefiniteMatrices, self).__init__()
        self.n = n
        self.dim = n * (n+1)//2
        self.order = order
        self.id = torch.eye(self.n, device=device, dtype=dtype).view(1, self.n, self.n)

        #self.lb_eigenspaces = None

    def generate_lb_eigenspaces(self, measure):
        shift = self.rand_phase(self.order)
        if isinstance(measure, MaternSpectralMeasure):
            nu, lengthscale = measure.nu[0], measure.lengthscale[0]
            scale = 1/torch.sqrt((self.n ** 3 - self.n)/48 + 2 * nu / lengthscale)
            goe_samples = GOE_sampler(self.order, self.n)  # (order, dim)
            chi2_samples = torch.distributions.chi2.Chi2(2 * nu).rsample((self.order,))
            chi_samples = torch.sqrt(chi2_samples)  # (order,)
            lmd = goe_samples/chi_samples[:, None] / scale
        elif isinstance(measure, SqExpSpectralMeasure):
            scale = measure.lengthscale
            goe_samples = GOE_sampler(self.order, self.n)
            lmd = goe_samples/scale
        else:
            return NotImplementedError
        self.lb_eigenspaces = SPDShiftedNormailizedExp(2*lmd, shift, self)

    def to_group(self, x):
        return torch.linalg.cholesky(x, upper=True)

    def rand_phase(self, n=1):
        qr = torch.randn((n, self.n, self.n),device=device, dtype=dtype)
        q, r = torch.linalg.qr(qr)
        r_diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        q *= r_diag_sign[:, None]
        q_det_sign = torch.sign(torch.det(q))
        q[:, :, 0] *= q_det_sign[:, None]
        return q

    def rand(self, n=1):
        """Note, there is no standard method to sample from SPD
            We sample random matrix M=XX^T X_{ij}\sim N(0,1)
            and normalize in such a way that \E(det M) = 1 (see https://mathoverflow.net/questions/13008/)"""
        rand = torch.randn(n, self.n, self.n, device=device, dtype=dtype)
        rand_pos = torch.bmm(rand, torch.transpose(rand, -2, -1)) * \
                   4 / (factorial(self.n+1) ** (1/self.n))
        return rand_pos

    def inv(self, x):
        return torch.linalg.inv(x)

    def _dist_to_id(self, x):
        eig = torch.linalg.eigvalsh(x)
        sq_log_eig = torch.square(torch.log(eig))
        dist = torch.sum(sq_log_eig, dim=1)
        return dist


class SPDShiftExp(NonCompactSymmetricSpaceExp):
        def __init__(self, lmd, shift, manifold):
            super().__init__(lmd=lmd, shift=shift, manifold=manifold)

        def compute_rho(self):
            rho = torch.tensor([(i + 1) - (self.manifold.n + 1) / 2 for i in range(self.manifold.n)],
                               device=device, dtype=dtype)
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


class SPDShiftedNormailizedExp(torch.nn.Module):
    def __init__(self, lmd, shift, manifold):
        super().__init__()
        self.n = manifold.n
        self.exp = SPDShiftExp(lmd, shift, manifold)
        self.coeff = self.c_function_tanh(lmd)  # (m,)

    def c_function_tanh(self, lmd):
        lmd_ = (lmd[:, None, :] - lmd[:, :, None])[triu_ind(lmd.size()[0], self.n, 1)].reshape(-1,
                                                                                       self.n * (self.n - 1) // 2)
        lmd_ = pi * torch.abs(lmd_)
        lmd_ = torch.tanh(lmd_)
        c_function_tanh = torch.sum(torch.log(lmd_), dim=1)

        return torch.exp(c_function_tanh/2)

    def forward(self, x):
        # x has shape (n, dim, dim)

        exp = self.exp(x)  # (n, m)
        return torch.einsum('nm,m->nm', exp, self.coeff)  # (n, m)


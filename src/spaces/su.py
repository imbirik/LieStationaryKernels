import torch
import numpy as np
from src.space import LieGroup, LBEigenspaceWithSum, LieGroupCharacter
from functools import reduce
import operator
import math
import itertools

dtype = torch.cdouble


class SU(LieGroup):
    """SU(dim), special unitary group of degree dim."""

    def __init__(self, dim: int, order: int):
        """
        :param dim: dimension of the space
        :param order: the order of approximation, the number of representations calculated
        """
        self.dim = dim
        self.rank = dim-1
        self.order = order
        self.Eigenspace = SULBEigenspace

        self.rho = np.arange(self.dim - 1, -self.dim, -2) * 0.5

        super().__init__(order=order)

    # def dist(self, x, y):
    #     return torch.arccos(torch.dot(x, y))
    #
    # def difference(self, x, y):
    #     return x @ y.mH

    def rand(self, n=1):
        h = torch.randn((n, self.dim, self.dim), dtype=dtype)
        q, r = torch.linalg.qr(h)
        r_diag = torch.diagonal(r, dim1=-2, dim2=-1)
        r_diag_inv_phase = torch.conj(r_diag / torch.abs(r_diag))
        q *= r_diag_inv_phase[:, None]
        q_det = torch.det(q)
        q_det_inv_phase = torch.conj(q_det / torch.abs((q_det)))
        q[:, :, 0] *= q_det_inv_phase[:, None]
        return q

    def generate_signatures(self, order):
        """Generate the signatures of irreducible representations

        Representations of SO(dim) can be enumerated by partitions of size dim, called signatures.
        :param int order: number of eigenfunctions that will be returned
        :return signatures: signatures of representations likely having the smallest LB eigenvalues
        """
        sign_vals_lim = order if self.dim == 1 else 10 if self.dim == 2 else 5
        signatures = list(itertools.combinations_with_replacement(range(sign_vals_lim, -1, -1), r=self.rank))
        signatures = [sgn + (0,) for sgn in signatures]
        signatures.sort()
        return signatures

    @staticmethod
    def inv(x: torch.Tensor):
        return x.mH


class SULBEigenspace(LBEigenspaceWithSum):
    """The Laplace-Beltrami eigenspace for the special unitary group."""

    def __init__(self, signature, *, manifold: SU):
        """
        :param signature: the signature of a representation
        :param manifold: the "parent" manifold, an instance of SO
        """
        super().__init__(signature, manifold=manifold)

    def compute_dimension(self):
        signature = self.index
        su = self.manifold
        rep_dim = reduce(operator.mul, (reduce(operator.mul, (signature[i - 1] - signature[j - 1] + j - i for j in
                                                              range(i + 1, su.dim + 1))) / math.factorial(su.dim - i)
                                        for i in range(1, su.dim)))
        return int(round(rep_dim))

    def compute_lb_eigenvalue(self):
        np_sgn = np.array(self.index)
        rho = self.manifold.rho
        return np.linalg.norm(rho + np_sgn) ** 2 - np.linalg.norm(rho) ** 2

    def compute_basis_sum(self):
        return SUCharacter(representation=self)


class SUCharacter(LieGroupCharacter):
    """Representation character for special unitary group"""

    @staticmethod
    def torus_embed(x):
        return torch.linalg.eigvals(x)

    def chi(self, x):
        dim = self.representation.manifold.dim
        signature = self.representation.index
        # eps = 0#1e-3*torch.tensor([1+1j]).cuda().item()
        gammas = self.torus_embed(x)
        qs = [pk + dim - k - 1 for k, pk in enumerate(signature)]
        numer_mat = torch.stack([torch.pow(gammas, q) for q in qs], dim=-1)
        vander = torch.tensor([reduce(operator.mul, (gi - gj for gi, gj in itertools.combinations(gamma, r=2)), 1) for gamma in gammas])
        return torch.det(numer_mat) / vander

    @staticmethod
    def close_to_eye(x):
        d = x.shape[1]  # x = [n,d,d]
        x_ = x.reshape((x.shape[0], -1))  # [n, d * d]

        eye = torch.reshape(torch.torch.eye(d, dtype=dtype).reshape((-1, d * d)), (1, d * d))  # [1, d * d]
        eyes = eye.repeat(x.shape[0], 1)  # [n, d * d]

        return torch.all(torch.isclose(x_, eyes), dim=1)

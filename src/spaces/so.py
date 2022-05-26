import torch
import numpy as np
from src.utils import fixed_length_partitions
from src.space import LieGroup, LBEigenspaceWithSum, LieGroupCharacter
import sympy as sp
from functools import reduce
import operator
import math
import itertools as it

dtype = torch.double


class SO(LieGroup):
    """SO(dim), special orthogonal group of degree dim."""

    def __init__(self, dim: int, order: int):
        """
        :param dim: dimension of the space
        :param order: the order of approximation, the number of representations calculated
        """
        if dim <= 2 or dim == 4:
            raise ValueError("Dimensions 1, 2, 4 are not supported")
        self.dim = dim
        self.rank = dim // 2
        self.order = order
        self.Eigenspace = SOLBEigenspace
        if self.dim % 2 == 0:
            self.rho = np.arange(self.rank-1, -1, -1)
        else:
            self.rho = np.arange(self.rank-1, -1, -1) + 0.5
        super().__init__(order=order)

    def dist(self, x, y):
        return torch.arccos(torch.dot(x, y))

    def difference(self, x, y):
        return x @ y.T

    def rand(self, n=1):
        h = torch.randn((n, self.dim, self.dim), dtype=dtype)
        q, r = torch.linalg.qr(h)
        r_diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        q *= r_diag_sign[:, None]
        q_det_sign = torch.sign(torch.det(q))
        q[:, :, 0] *= q_det_sign[:, None]
        return q

    def generate_signatures(self, order):
        """Generate the signatures of irreducible representations

        Representations of SO(dim) can be enumerated by partitions of size dim, called signatures.
        :param int order: number of eigenfunctions that will be returned
        :return signatures: signatures of representations likely having the smallest LB eigenvalues
        """
        signatures = []
        if self.dim == 3:
            signature_sum = order
        else:
            signature_sum = 20
        for signature_sum in range(0, signature_sum):
            for i in range(0, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank-i))
                    signatures.append(tuple(signature))
                    if self.dim % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))
        return signatures

    @staticmethod
    def inv(x: torch.Tensor):
        return x.mT


class SOLBEigenspace(LBEigenspaceWithSum):
    """The Laplace-Beltrami eigenspace for the special orthogonal group."""
    def __init__(self, signature, *, manifold: SO):
        """
        :param signature: the signature of a representation
        :param manifold: the "parent" manifold, an instance of SO
        """
        super().__init__(signature, manifold=manifold)

    def compute_dimension(self):
        signature = self.index
        so = self.manifold
        if so.dim % 2 == 1:
            qs = [pk + so.rank - k - 1 / 2 for k, pk in enumerate(signature)]
            rep_dim = reduce(operator.mul, (2 * qs[k] / math.factorial(2 * k + 1) for k in range(0, so.rank))) \
                      * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                              for i, j in it.combinations(range(so.rank), 2)), 1)
            return int(round(rep_dim))
        else:
            qs = [pk + so.rank - k - 1 if k != so.rank - 1 else abs(pk) for k, pk in enumerate(signature)]
            rep_dim = int(reduce(operator.mul, (2 / math.factorial(2 * k) for k in range(1, so.rank)))
                          * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                                  for i, j in it.combinations(range(so.rank), 2)), 1))
            return int(round(rep_dim))

    def compute_lb_eigenvalue(self):
        np_sgn = np.array(self.index)
        rho = self.manifold.rho
        return np.linalg.norm(rho + np_sgn) ** 2 - np.linalg.norm(rho) ** 2

    def compute_basis_sum(self):
        return SOCharacter(representation=self)


class SOCharacter(LieGroupCharacter):
    """Representation character for special orthogonal group"""
    @staticmethod
    def torus_embed(x):
        #TODO :check
        eigv = torch.linalg.eigvals(x)
        sorted_ind = torch.sort(torch.view_as_real(eigv), dim=1).indices[:, :, 0]
        eigv = torch.gather(eigv, dim=1, index=sorted_ind)
        gamma = eigv[:, 0:-1:2]
        return gamma

    @staticmethod
    def xi0(qs, gamma):
        a = torch.stack([torch.pow(gamma, q) + torch.pow(gamma, -q) for q in qs], dim=-1)
        return torch.det(a)

    @staticmethod
    def xi1(qs, gamma):
        a = torch.stack([torch.pow(gamma, q) - torch.pow(gamma, -q) for q in qs], dim=-1)
        return torch.det(a)

    def chi(self, x):
        rank = self.representation.manifold.rank
        signature = self.representation.index
        # eps = 0#1e-3*torch.tensor([1+1j]).cuda().item()
        gamma = self.torus_embed(x)
        if self.representation.dimension % 2:
            qs = [pk + rank - k - 1 / 2 for k, pk in enumerate(signature)]
            return self.xi1(qs, gamma) / \
                   self.xi1([k - 1 / 2 for k in range(rank, 0, -1)], gamma)
        else:
            qs = [pk + rank - k - 1 if k != rank - 1 else abs(pk)
                  for k, pk in enumerate(signature)]
            if signature[-1] == 0:
                return self.xi0(qs, gamma) / \
                       self.xi0(list(reversed(range(rank))), gamma)
            else:
                sign = math.copysign(1, signature[-1])
                return (self.xi0(qs, gamma) + self.xi1(qs, gamma) * sign) / \
                       self.xi0(list(reversed(range(rank))), gamma)

    @staticmethod
    def close_to_eye(x):
        d = x.shape[1]  # x = [n,d,d]
        x_ = x.reshape((x.shape[0], -1))  # [n, d * d]

        eye = torch.reshape(torch.torch.eye(d, dtype=dtype).reshape((-1, d * d)), (1, d * d))  # [1, d * d]
        eyes = eye.repeat(x.shape[0], 1)  # [n, d * d]

        return torch.all(torch.isclose(x_, eyes), dim=1)

import torch
import numpy as np
from src.utils import fixed_length_partitions
from src.space import LieGroup
import sympy as sp
from functools import reduce
import operator
import math
import itertools as it

dtype = torch.double


class SO(LieGroup):
    '''
    S^{dim} sphere is contained in R^{dim+1}
    '''

    def __init__(self, dim: int, order: int):
        '''
        :param dim: sphere dimension
        :param order: order of approximation. Number of eigenspaces under consideration.
        '''
        super(SO, self).__init__()
        self.dim = dim
        self.rank = dim // 2
        self.order = order
        if self.dim % 2 == 0:
            self.rho = np.arange(self.rank)[::-1]
        else:
            self.rho = np.arange(self.rank)[::-1] + 0.5

        if dim <= 2 or dim == 4:
            raise ValueError("dimensions 1,2,4 are not supported")

        self.signatures, self.eigenvalues, self.eigenspaces_dims = self._generate_signatures(self.order)
        self.eigenfunctions = [SOCharacter(self.dim, signature) for signature in self.signatures]

    def dist(self, x, y):
        return torch.arccos(torch.dot(x, y))

    def difference(self, x, y):
        return x @ y.T

    def rand(self, n=1):
        h = torch.randn((n, self.dim, self.dim), dtype=torch.double)
        q, r = torch.linalg.qr(h)
        diag_sign = torch.diag_embed(torch.diagonal(torch.sign(r), dim1=-2, dim2=-1))
        q = torch.bmm(q, diag_sign)
        det_sign = torch.sign(torch.det(q))
        sign_matirx = torch.eye(self.dim, dtype=torch.double).reshape((-1, self.dim, self.dim)).repeat((n, 1, 1))
        sign_matirx[:, 0, 0] = det_sign
        q = q @ sign_matirx
        return q

    def _generate_signatures(self, order):
        '''
        Representations of SO can be enumerated by partitions of size dim that we will call signatures.
        :param int order: number of eigenfunctions that will be returned
        :return signatures, eigenspaces_dims, eigenvalues: top order representations sorted by eigenvalues.
        '''
        signatures = []
        if self.dim == 3:
            signature_sum = order
        else:
            signature_sum = 20
        for signature_sum in range(0, signature_sum):
            for i in range(1, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank-i))
                    signatures.append(tuple(signature))
                    if self.dim % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))

        def _compute_dim(signature):
            if self.dim % 2 == 1:
                qs = [pk + self.rank - k - 1 / 2 for k, pk in enumerate(signature)]
                rep_dim = reduce(operator.mul, (2 * qs[k] / math.factorial(2 * k + 1) for k in range(0, self.rank))) \
                             * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                                  for i, j in it.combinations(range(self.rank), 2)), 1)
                return int(round(rep_dim))
            else:
                qs = [pk + self.rank - k - 1 if k != self.rank - 1 else abs(pk) for k, pk in enumerate(signature)]
                rep_dim = int(reduce(operator.mul, (2 / math.factorial(2 * k) for k in range(1, self.rank)))
                              * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                             for i, j in it.combinations(range(self.rank), 2)), 1))
                return int(round(rep_dim))

        def _compute_eigenvalue(sgn):
            np_sgn = np.array(sgn)
            return np.linalg.norm(self.rho + np_sgn) ** 2 - np.linalg.norm(self.rho) ** 2

        signatures_vals = []
        for sgn in signatures:
            dim = _compute_dim(sgn)
            eigenvalue = _compute_eigenvalue(sgn)
            signatures_vals.append([sgn, dim, eigenvalue])

        signatures_vals.sort(key=lambda x: x[2])
        signatures_vals = signatures_vals[:order]

        signatures = np.array([x[0] for x in signatures_vals])
        dims = torch.tensor([x[1] for x in signatures_vals], dtype=dtype)
        eigenvalues = torch.tensor([x[2] for x in signatures_vals])

        return signatures, dims, eigenvalues


class SOCharacter(torch.nn.Module):
    def __init__(self, dim, signature):
        super(SOCharacter, self).__init__()
        self.dim = dim
        self.signature = signature
        self.rank = dim // 2

    def torus_embed(self, x):
        eigv = torch.linalg.eigvals(x)
        sorted_ind = torch.sort(torch.view_as_real(eigv), dim=1).indices[:, :, 0]
        eigv = torch.gather(eigv, dim=1, index=sorted_ind)
        gamma = eigv[:, 0:-1:2]
        return gamma

    def xi0(self, qs, gamma):
        a = torch.stack([torch.pow(gamma, q) + torch.pow(gamma, -q) for q in qs], dim=-1)
        return torch.det(a)

    def xi1(self, qs, gamma):
        a = torch.stack([torch.pow(gamma, q) - torch.pow(gamma, -q) for q in qs], dim=-1)
        return torch.det(a)

    def forward(self, x):
        eps = 0#1e-3*torch.tensor([1+1j]).cuda().item()
        gamma = self.torus_embed(x)+eps

        if self.dim % 2:
            qs = [pk + self.rank - k - 1 / 2 for k, pk in enumerate(self.signature)]
            return self.xi1(qs, gamma) / \
                   self.xi1([k - 1 / 2 for k in range(self.rank, 0, -1)], gamma)
        else:
            qs = [pk + self.rank - k - 1 if k != self.rank - 1 else abs(pk)
                  for k, pk in enumerate(self.signature)]
            if self.signature[-1] == 0:
                return self.xi0(qs, gamma) / \
                       self.xi0(list(reversed(range(self.rank))), gamma)
            else:
                sign = math.copysign(1, self.signature[-1])
                return (self.xi0(qs, gamma) + self.xi1(qs, gamma) * sign) / \
                       self.xi0(list(reversed(range(self.rank))), gamma)

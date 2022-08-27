import torch
import numpy as np
from lie_geom_kernel.space import CompactLieGroup, LBEigenspaceWithBasis, LieGroupCharacter, TranslatedCharactersBasis
from functools import reduce
import operator
import math
import itertools
import more_itertools
from lie_geom_kernel.utils import partition_dominance_cone
import sympy
from sympy.matrices.determinant import _det as sp_det
import json
from pathlib import Path

dtype = torch.cdouble
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pi = 2*torch.acos(torch.zeros(1)).item()


class SU(CompactLieGroup):
    """
    SU(n), special unitary group of degree `n`.
    """
    def __init__(self, n: int, order=10):
        """
        :param n: dimension of the space
        :param order: the order of approximation, the number of representations calculated
        """
        self.n = n
        self.dim = n*n-1
        self.rank = n-1
        self.order = order
        self.Eigenspace = SULBEigenspace

        self.rho = np.arange(self.n - 1, -self.n, -2) * 0.5
        self.id = torch.eye(self.n, device=device, dtype=dtype).view(1, self.n, self.n)
        super().__init__(order=order)

    def dist(self, x, y):
        raise NotImplementedError

    def difference(self, x, y):
        return x @ y.mH

    def rand(self, num=1):
        if self.n == 2:
            sphere_point = torch.randn((num, 4), dtype=torch.double, device=device)
            sphere_point /= torch.linalg.vector_norm(sphere_point, dim=-1, keepdim=True)
            a = torch.view_as_complex(sphere_point[..., :2]).unsqueeze(-1)
            b = torch.view_as_complex(sphere_point[..., 2:]).unsqueeze(-1)
            r1 = torch.hstack((a, -b.conj())).unsqueeze(-1)
            r2 = torch.hstack((b, a.conj())).unsqueeze(-1)
            q = torch.cat((r1, r2), -1)
            return q
        else:
            h = torch.randn((num, self.n, self.n), dtype=dtype, device=device)
            q, r = torch.linalg.qr(h)
            r_diag = torch.diagonal(r, dim1=-2, dim2=-1)
            r_diag_inv_phase = torch.conj(r_diag / torch.abs(r_diag))
            q *= r_diag_inv_phase[:, None]
            q_det = torch.det(q)
            q_det_inv_phase = torch.conj(q_det / torch.abs(q_det))
            q[:, :, 0] *= q_det_inv_phase[:, None]
            return q

    def generate_signatures(self, order):
        """Generate the signatures of irreducible representations

        Representations of SU(dim) can be enumerated by partitions of size dim, called signatures.
        :param int order: number of eigenfunctions that will be returned
        :return signatures: signatures of representations likely having the smallest LB eigenvalues
        """
        sign_vals_lim = 100 if self.n in (1, 2) else 30 if self.n == 3 else 10
        signatures = list(itertools.combinations_with_replacement(range(sign_vals_lim, -1, -1), r=self.rank))
        signatures = [sgn + (0,) for sgn in signatures]
        signatures.sort()
        return signatures

    @staticmethod
    def inv(x: torch.Tensor):
        # (n, dim, dim)
        return torch.conj(torch.transpose(x, -2, -1))

    @staticmethod
    def close_to_id(x):
        d = x.shape[-1]  # x = [...,d,d]
        x_ = x.reshape(x.shape[:-2] + (-1,))  # [..., d * d]
        eyes = torch.broadcast_to(torch.flatten(torch.eye(d, dtype=dtype, device=device)), x_.shape)  # [..., d * d]
        return torch.all(torch.isclose(x_, eyes, atol=1e-5), dim=-1)

    def torus_representative(self, x):
        return torch.linalg.eigvals(x)

    def pairwise_dist(self, x, y):
        """For n points x_i and m points y_j computed dist(x_i,y_j)
            TODO: CHECK
        """
        x_y_ = self.pairwise_embed(x, y)
        log_x_y_ = torch.arccos(x_y_.real)
        dist = math.sqrt(2)*torch.minimum(log_x_y_, 2*pi-log_x_y_)
        dist = torch.norm(dist, dim=1).reshape(x.shape[0], y.shape[0])
        return dist


class SULBEigenspace(LBEigenspaceWithBasis):
    """
    The Laplace-Beltrami eigenspace for the special unitary group.
    """
    def __init__(self, signature, *, manifold: SU):
        """
        :param signature: the signature of a representation
        :param manifold: the "parent" manifold, an instance of SU
        """
        super().__init__(signature, manifold=manifold)

    def compute_dimension(self):
        signature = self.index
        su = self.manifold
        rep_dim = reduce(operator.mul, (reduce(operator.mul, (signature[i - 1] - signature[j - 1] + j - i for j in
                                                              range(i + 1, su.n + 1))) / math.factorial(su.n - i)
                                        for i in range(1, su.n)))
        return int(round(rep_dim))

    def compute_lb_eigenvalue(self):
        sgn = np.array(self.index, dtype=float)
        # transform the signature into the same basis as rho
        sgn -= np.mean(sgn)
        rho = self.manifold.rho
        lb_eigenvalue = (np.linalg.norm(rho + sgn) ** 2 - np.linalg.norm(rho) ** 2)  # / (2 * self.manifold.n)
        return lb_eigenvalue.item()

    def compute_phase_function(self):
        return SUCharacter(representation=self)

    def compute_basis(self):
        return TranslatedCharactersBasis(representation=self)


class SUCharacter(LieGroupCharacter):
    def __init__(self, *, representation: SULBEigenspace, precomputed=True):
        super().__init__(representation=representation)
        if precomputed:
            group_name = '{}({})'.format(self.representation.manifold.__class__.__name__,
                                         self.representation.manifold.n)
            file_path = Path(__file__).with_name('precomputed_characters.json')
            with file_path.open('r') as file:
                character_formulas = json.load(file)
                try:
                    cs, ms = character_formulas[group_name][str(self.representation.index)]
                    self.coeffs, self.monoms = (torch.tensor(data, dtype=torch.int, device=device) for data in (cs, ms))
                except KeyError as e:
                    raise KeyError('Unable to retrieve character parameters for signature {} of {}, '
                                   'perhaps it is not precomputed.'.format(e.args[0], group_name)) from None

    def _compute_character_formula(self):
        n = self.representation.manifold.n
        gammas = sympy.symbols(' '.join('g{}'.format(i) for i in range(1, n + 1)))
        qs = [pk + n - k - 1 for k, pk in enumerate(self.representation.index)]
        numer_mat = sympy.Matrix(n, n, lambda i, j: gammas[i]**qs[j])
        numer = sympy.Poly(sp_det(numer_mat, method='berkowitz'))
        denom = sympy.Poly(sympy.prod(gammas[i] - gammas[j] for i, j in itertools.combinations(range(n), r=2)))
        monomials_tuples = list(itertools.chain.from_iterable(
            more_itertools.distinct_permutations(p) for p in partition_dominance_cone(self.representation.index)
        ))
        monomials = [sympy.polys.monomials.Monomial(m, gammas).as_expr() for m in monomials_tuples]
        chi_coeffs = list(more_itertools.always_iterable(sympy.symbols(' '.join('c{}'.format(i) for i in range(1, len(monomials) + 1)))))
        chi_poly = sympy.Poly(sum(c * m for c, m in zip(chi_coeffs, monomials)), gammas)
        pr = chi_poly * denom - numer
        sol = list(sympy.linsolve(pr.coeffs(), chi_coeffs)).pop()
        p = sympy.Poly(sum(c * m for c, m in zip(sol, monomials)), gammas)
        coeffs = list(map(int, p.coeffs()))
        monoms = [list(map(int, monom)) for monom in p.monoms()]
        return coeffs, monoms

    def chi(self, gammas):
        char_val = torch.zeros(gammas.shape[:-1], dtype=dtype, device=device)
        for coeff, monom in zip(self.coeffs, self.monoms):
            char_val += coeff * torch.prod(gammas ** monom, dim=-1)
        return char_val

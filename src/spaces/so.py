import torch
import numpy as np
from src.utils import fixed_length_partitions, partition_dominance_or_subpartition_cone
from src.space import CompactLieGroup, LBEigenspaceWithSum, LieGroupCharacter
from functools import reduce
import operator
import math
import itertools
import more_itertools
import sympy
from sympy.matrices.determinant import _det as sp_det
import json
from pathlib import Path

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pi = 2*torch.acos(torch.zeros(1)).item()


class SO(CompactLieGroup):
    """SO(dim), special orthogonal group of degree dim."""

    def __init__(self, n: int, order=20):
        """
        :param dim: dimension of the space
        :param order: the order of approximation, the number of representations calculated
        """
        if n <= 2 and order:
            raise ValueError("Dimensions 1, 2 are not supported")
        self.n = n
        self.dim = n * (n-1)//2
        self.rank = n // 2
        self.order = order
        self.Eigenspace = SOLBEigenspace
        if self.n % 2 == 0:
            self.rho = np.arange(self.rank-1, -1, -1)
        else:
            self.rho = np.arange(self.rank-1, -1, -1) + 0.5
        self.id = torch.eye(self.n, device=device, dtype=dtype)
        CompactLieGroup.__init__(self, order=order)


    def difference(self, x, y):
        return x @ y.T

    def dist(self, x, y):
        """Batched geodesic distance"""
        diff = torch.bmm(x, self.inv(y))
        torus_diff = self.torus_representative(diff)
        log_torus_diff = torch.arccos(torus_diff.real)
        dist = math.sqrt(2) * torch.minimum(log_torus_diff, 2 * pi - log_torus_diff)
        dist = torch.norm(dist, dim=1)
        return dist

    def pairwise_dist(self, x, y):
        """For n points x_i and m points y_j computed dist(x_i,y_j)"""
        x_y_ = self.pairwise_embed(x, y)
        log_x_y_ = torch.arccos(x_y_.real)
        dist = math.sqrt(2)*torch.minimum(log_x_y_, 2*pi-log_x_y_)
        dist = torch.norm(dist, dim=1).reshape(x.shape[0], y.shape[0])
        return dist

    def rand(self, n=1):
        h = torch.randn((n, self.n, self.n), device=device, dtype=dtype)
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
        if self.n == 3:
            signature_sum = 200
        else:
            signature_sum = 30
        for signature_sum in range(0, signature_sum):
            for i in range(0, self.rank + 1):
                for signature in fixed_length_partitions(signature_sum, i):
                    signature.extend([0] * (self.rank-i))
                    signatures.append(tuple(signature))
                    if self.n % 2 == 0 and signature[-1] != 0:
                        signature[-1] = -signature[-1]
                        signatures.append(tuple(signature))
        return signatures

    @staticmethod
    def inv(x: torch.Tensor):
        # (n, dim, dim)
        return torch.transpose(x, -2, -1)

    @staticmethod
    def close_to_id(x):
        d = x.shape[-1]  # x = [...,d,d]
        x_ = x.reshape(x.shape[:-2] + (-1,))  # [..., d * d]
        eyes = torch.broadcast_to(torch.flatten(torch.eye(d, dtype=dtype, device=device)), x_.shape)  # [..., d * d]
        return torch.all(torch.isclose(x_, eyes, atol=1e-5), dim=-1)

    def torus_representative(self, x):
        if self.n % 2 == 1:
            eigvals = torch.linalg.eigvals(x)
            sorted_ind = torch.sort(torch.view_as_real(eigvals), dim=-2).indices[..., 0]
            eigvals = torch.gather(eigvals, dim=-1, index=sorted_ind)
            gamma = eigvals[..., 0:-1:2]
            return gamma
        else:
            eigvals, eigvecs = torch.linalg.eig(x)
            # c is a matrix transforming x into its canonical form (with 2x2 blocks)
            c = torch.zeros_like(eigvecs)
            c[..., ::2] = eigvecs[..., ::2].real
            c[..., 1::2] = eigvecs[..., ::2].imag
            c *= math.sqrt(2)
            eigvals[..., 0] = torch.pow(eigvals[..., 0], torch.det(c))
            gamma = eigvals[..., ::2]
            return gamma


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
        if so.n % 2 == 1:
            qs = [pk + so.rank - k - 1 / 2 for k, pk in enumerate(signature)]
            rep_dim = reduce(operator.mul, (2 * qs[k] / math.factorial(2 * k + 1) for k in range(0, so.rank))) \
                      * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                              for i, j in itertools.combinations(range(so.rank), 2)), 1)
            return int(round(rep_dim))
        else:
            qs = [pk + so.rank - k - 1 if k != so.rank - 1 else abs(pk) for k, pk in enumerate(signature)]
            rep_dim = int(reduce(operator.mul, (2 / math.factorial(2 * k) for k in range(1, so.rank)))
                          * reduce(operator.mul, ((qs[i] - qs[j]) * (qs[i] + qs[j])
                                                  for i, j in itertools.combinations(range(so.rank), 2)), 1))
            return int(round(rep_dim))

    def compute_lb_eigenvalue(self):
        np_sgn = np.array(self.index)
        rho = self.manifold.rho
        # killing_form_coeff = 4 * self.manifold.rank - (4 if self.manifold.n % 2 == 0 else 2)
        lb_eigenvalue = (np.linalg.norm(rho + np_sgn) ** 2 - np.linalg.norm(rho) ** 2)  # / killing_form_coeff
        return lb_eigenvalue.item()

    def compute_basis_sum(self):
        return SOCharacterDenominatorFree(representation=self)
        # if self.manifold.n == 3:
        #     return SO3Character(representation=self)
        # else:
        #     return SOCharacter(representation=self)


class SOCharacter(LieGroupCharacter):
    """Representation character for special orthogonal group"""
    # @staticmethod

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
        gamma = self.representation.manifold.torus_embed(x)
        if self.representation.manifold.n % 2:
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
                       (1 * self.xi0(list(reversed(range(rank))), gamma))


class SOCharacterDenominatorFree(LieGroupCharacter):
    def __init__(self, *, representation: SOLBEigenspace, precomputed=True):
        super().__init__(representation=representation)
        if precomputed:
            group_name = '{}({})'.format(self.representation.manifold.__class__.__name__, self.representation.manifold.n)
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
        rank = self.representation.manifold.rank
        signature = self.representation.index
        gammas = sympy.symbols(' '.join('g{}'.format(i + 1) for i in range(rank)))
        gammas = list(more_itertools.always_iterable(gammas))
        gammas_conj = sympy.symbols(' '.join('gc{}'.format(i + 1) for i in range(rank)))
        gammas_conj = list(more_itertools.always_iterable(gammas_conj))
        chi_variables = gammas + gammas_conj
        if n % 2:
            gammas_sqrt = sympy.symbols(' '.join('gr{}'.format(i + 1) for i in range(rank)))
            gammas_sqrt = list(more_itertools.always_iterable(gammas_sqrt))
            gammas_conj_sqrt = sympy.symbols(' '.join('gcr{}'.format(i + 1) for i in range(rank)))
            gammas_conj_sqrt = list(more_itertools.always_iterable(gammas_conj_sqrt))
            chi_variables = gammas_sqrt + gammas_conj_sqrt
            def xi1(qs):
                mat = sympy.Matrix(rank, rank, lambda i, j: gammas_sqrt[i]**qs[j]-gammas_conj_sqrt[i]**qs[j])
                return sympy.Poly(sp_det(mat, method='berkowitz'), chi_variables)
            # qs = [sympy.Integer(2*pk + 2*rank - 2*k - 1) / 2 for k, pk in enumerate(signature)]
            qs = [2 * pk + 2 * rank - 2 * k - 1 for k, pk in enumerate(signature)]
            # denom_pows = [sympy.Integer(2*k - 1) / 2 for k in range(rank, 0, -1)]
            denom_pows = [2 * k - 1 for k in range(rank, 0, -1)]
            numer = xi1(qs)
            denom = xi1(denom_pows)
        else:
            def xi0(qs):
                mat = sympy.Matrix(rank, rank, lambda i, j: gammas[i] ** qs[j] + gammas_conj[i] ** qs[j])
                return sympy.Poly(sp_det(mat, method='berkowitz'), chi_variables)
            def xi1(qs):
                mat = sympy.Matrix(rank, rank, lambda i, j: gammas[i] ** qs[j] - gammas_conj[i] ** qs[j])
                return sympy.Poly(sp_det(mat, method='berkowitz'), chi_variables)
            qs = [pk + rank - k - 1 if k != rank - 1 else abs(pk) for k, pk in enumerate(signature)]
            pm = signature[-1]
            numer = xi0(qs)
            if pm:
                numer += (1 if pm > 0 else -1) * xi1(qs)
            denom = xi0(list(reversed(range(rank))))
        partition = tuple(map(abs, self.representation.index)) + tuple([0] * self.representation.manifold.rank)
        monomials_tuples = itertools.chain.from_iterable(
            more_itertools.distinct_permutations(p) for p in partition_dominance_or_subpartition_cone(partition)
        )
        monomials_tuples = filter(lambda p: all(p[i] == 0 or p[i + rank] == 0 for i in range(rank)), monomials_tuples)
        monomials_tuples = list(monomials_tuples)
        monomials = [sympy.polys.monomials.Monomial(m, chi_variables).as_expr()
                     for m in monomials_tuples]
        chi_coeffs = list(more_itertools.always_iterable(
            sympy.symbols(' '.join('c{}'.format(i) for i in range(1, len(monomials) + 1)))))
        exponents = [n % 2 + 1] * len(monomials)  # the correction s.t. chi is the same polynomial for both oddities of n
        chi_poly = sympy.Poly(sum(c * m**d for c, m, d in zip(chi_coeffs, monomials, exponents)), chi_variables)
        pr = chi_poly * denom - numer
        if n % 2:
            pr = sympy.Poly(pr.subs((g*gc, 1) for g, gc in zip(gammas_sqrt, gammas_conj_sqrt)), chi_variables)
        else:
            pr = sympy.Poly(pr.subs((g*gc, 1) for g, gc in zip(gammas, gammas_conj)), chi_variables)
        sol = list(sympy.linsolve(pr.coeffs(), chi_coeffs)).pop()
        if n % 2:
            chi_variables = gammas + gammas_conj
            chi_poly = sympy.Poly(chi_poly.subs([gr ** 2, g] for gr, g in zip(gammas_sqrt + gammas_conj_sqrt, chi_variables)), chi_variables)
        p = sympy.Poly(chi_poly.subs((c, c_val) for c, c_val in zip(chi_coeffs, sol)), chi_variables)
        coeffs = list(map(int, p.coeffs()))
        monoms = [list(map(int, monom)) for monom in p.monoms()]
        return coeffs, monoms

    def chi(self, gammas):
        gammas = torch.cat((gammas, gammas.conj()), dim=-1)
        char_val = torch.zeros(gammas.shape[:-1], dtype=torch.cdouble, device=device)
        for coeff, monom in zip(self.coeffs, self.monoms):
            char_val += coeff * torch.prod(gammas ** monom, dim=-1)
        return char_val


class SO3Character(LieGroupCharacter):
    @staticmethod

    def chi(self, x):
        l = self.representation.index[0]
        gamma = self.representation.manifold.torus_representative(x)
        numer = torch.pow(gamma, l+0.5) - torch.pow(torch.conj(gamma), l+0.5)
        denom = torch.sqrt(gamma) - torch.sqrt(torch.conj(gamma))
        return numer / denom

import torch
import numpy as np
from lie_stationary_kernels.utils import fixed_length_partitions, partition_dominance_or_subpartition_cone
from lie_stationary_kernels.space import CompactLieGroup, LBEigenspaceWithBasis, LieGroupCharacter, TranslatedCharactersBasis
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


class SO(CompactLieGroup):
    """
    SO(n), special orthogonal group of degree `n`.
    """
    def __init__(self, n: int, order=20):
        """
        :param n: dimension of the space
        :param order: the order of approximation, the number of representations calculated
        """
        if n <= 2 and order:
            raise ValueError("Dimensions 1, 2 are not supported")
        self.n = n
        self.dim = n * (n-1) // 2
        self.rank = n // 2
        self.order = order
        self.Eigenspace = SOLBEigenspace
        if self.n % 2 == 0:
            self.rho = np.arange(self.rank-1, -1, -1)
        else:
            self.rho = np.arange(self.rank-1, -1, -1) + 0.5
        self.id = torch.eye(self.n, device=device, dtype=dtype).view(1, self.n, self.n)
        CompactLieGroup.__init__(self, order=order)

    def difference(self, x, y):
        return x @ y.T

    def dist(self, x, y):
        """Batched geodesic distance"""
        diff = torch.bmm(x, self.inv(y))
        torus_diff = self.torus_representative(diff)
        log_torus_diff = torch.arccos(torus_diff.real)
        dist = math.sqrt(2) * torch.minimum(log_torus_diff, 2 * math.pi - log_torus_diff)
        dist = torch.norm(dist, dim=1)
        return dist

    def pairwise_dist(self, x, y):
        """For n points x_i and m points y_j computed dist(x_i,y_j)"""
        x_y_ = self.pairwise_embed(x, y)
        log_x_y_ = torch.arccos(x_y_.real)
        dist = math.sqrt(2) * torch.minimum(log_x_y_, 2 * math.pi - log_x_y_)
        dist = torch.norm(dist, dim=1).reshape(x.shape[0], y.shape[0])
        return dist

    def rand(self, num=1):
        if self.n == 2:
            # SO(2) = S^1
            thetas = 2 * math.pi * torch.rand((num, 1), dtype=dtype, device=device)
            c = torch.cos(thetas)
            s = torch.sin(thetas)
            r1 = torch.hstack((c, s)).unsqueeze(-2)
            r2 = torch.hstack((-s, c)).unsqueeze(-2)
            q = torch.cat((r1, r2), dim=-2)
            return q
        elif self.n == 3:
            # explicit parametrization via the double cover SU(2) = S^3
            sphere_point = torch.randn((num, 4), dtype=torch.double, device=device)
            sphere_point /= torch.linalg.vector_norm(sphere_point, dim=-1, keepdim=True)
            x, y, z, w = (sphere_point[..., i].unsqueeze(-1) for i in range(4))
            xx = x ** 2
            yy = y ** 2
            zz = z ** 2
            xy = x * y
            xz = x * z
            xw = x * w
            yz = y * z
            yw = y * w
            zw = z * w
            del sphere_point, x, y, z, w
            r1 = torch.hstack((1-2*(yy+zz), 2*(xy-zw), 2*(xz+yw))).unsqueeze(-1)
            r2 = torch.hstack((2*(xy+zw), 1-2*(xx+zz), 2*(yz-xw))).unsqueeze(-1)
            r3 = torch.hstack((2*(xz-yw), 2*(yz+xw), 1-2*(xx+yy))).unsqueeze(-1)
            del xx, yy, zz, xy, xz, xw, yz, yw, zw
            q = torch.cat((r1, r2, r3), -1)
            return q
        else:
            h = torch.randn((num, self.n, self.n), device=device, dtype=dtype)
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
        if self.n == 3:
            # In SO(3) the torus representative is determined by the non-trivial pair of eigenvalues,
            # which can be calculated from the trace
            trace = torch.einsum('...ii->...', x)
            real = (trace - 1) / 2
            imag = torch.sqrt(torch.max(1-torch.square(real), torch.zeros_like(real)))
            return torch.view_as_complex(torch.cat((real.unsqueeze(-1), imag.unsqueeze(-1)), -1)).unsqueeze(-1)
        elif self.n % 2 == 1:
            # In SO(2n+1) the torus representative is determined by the (unordered) non-trivial eigenvalues
            eigvals = torch.linalg.eigvals(x)
            sorted_ind = torch.sort(torch.view_as_real(eigvals), dim=-2).indices[..., 0]
            eigvals = torch.gather(eigvals, dim=-1, index=sorted_ind)
            gamma = eigvals[..., 0:-1:2]
            return gamma
        else:
            # In SO(2n) each unordered set of eigenvalues determines two conjugacy classes
            eigvals, eigvecs = torch.linalg.eig(x)
            sorted_ind = torch.sort(torch.view_as_real(eigvals), dim=-2).indices[..., 0]
            eigvals = torch.gather(eigvals, dim=-1, index=sorted_ind)
            eigvecs = torch.gather(eigvecs, dim=-1, index=sorted_ind.unsqueeze(-2).broadcast_to(eigvecs.shape))
            # c is a matrix transforming x into its canonical form (with 2x2 blocks)
            c = torch.zeros_like(eigvecs)
            c[..., ::2] = eigvecs[..., ::2] + eigvecs[..., 1::2]
            c[..., 1::2] = (eigvecs[..., ::2] - eigvecs[..., 1::2])
            # eigenvectors calculated by LAPACK are either real or purely imaginary, make everything real
            # WARNING: might depend on the implementation of the eigendecomposition!
            c = c.real + c.imag
            # normalize s.t. det(c)≈±1, probably unnecessary
            c /= math.sqrt(2)
            torch.pow(eigvals[..., 0], torch.det(c).sgn(), out=eigvals[..., 0])
            gamma = eigvals[..., ::2]
            return gamma


class SOLBEigenspace(LBEigenspaceWithBasis):
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

    def compute_phase_function(self):
        return SOCharacter(representation=self)

    def compute_basis(self):
        return TranslatedCharactersBasis(representation=self)


class SOCharacter(LieGroupCharacter):
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

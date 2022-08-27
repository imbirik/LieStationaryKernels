import sys
import itertools
import unittest
from parameterized import parameterized_class
import torch
from torch.autograd.functional import _vmap as vmap
import numpy as np

from lie_stationary_kernels.spectral_kernel import EigenbasisSumKernel
from lie_stationary_kernels.spectral_measure import SqExpSpectralMeasure
from lie_stationary_kernels.prior_approximation import RandomPhaseApproximation

from lie_stationary_kernels.space import TranslatedCharactersBasis

from lie_stationary_kernels.spaces import SO, SU

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(precision=2, sci_mode=False, linewidth=160, edgeitems=15)


# Parametrized test, produces test classes called Test_Group.dim.order, for example, Test_SO.3.10 or Test_SU.2.5
@parameterized_class([
    {'group': SO, 'dim': 3, 'order': 20, 'dtype': torch.double},
    {'group': SO, 'dim': 4, 'order': 20, 'dtype': torch.double},
    {'group': SO, 'dim': 5, 'order': 20, 'dtype': torch.double},
    {'group': SO, 'dim': 6, 'order': 20, 'dtype': torch.double},
    {'group': SO, 'dim': 7, 'order': 20, 'dtype': torch.double},
    {'group': SO, 'dim': 8, 'order': 20, 'dtype': torch.double},
    {'group': SU, 'dim': 2, 'order': 20, 'dtype': torch.cdouble},
    {'group': SU, 'dim': 3, 'order': 20, 'dtype': torch.cdouble},
    {'group': SU, 'dim': 4, 'order': 20, 'dtype': torch.cdouble},
    {'group': SU, 'dim': 5, 'order': 20, 'dtype': torch.cdouble},
    {'group': SU, 'dim': 6, 'order': 20, 'dtype': torch.cdouble},
], class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["group"].__name__}.'
                                                 f'{params_dict["dim"]}.{params_dict["order"]}')
class TestCompactLieGroups(unittest.TestCase):

    def setUp(self) -> None:
        self.group = self.group(n=self.dim, order=self.order)

        self.lengthscale, self.nu = 2.0, 5.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.group)
        self.space_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.group)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**4)

        self.n, self.m = 20, 20
        self.x, self.y = self.group.rand(self.n), self.group.rand(self.m)

    def test_character_conjugation_invariance(self):
        num_samples_x = 20
        num_samples_g = 20
        xs = self.group.rand(num_samples_x).unsqueeze(0)
        gs = self.group.rand(num_samples_g).unsqueeze(1)
        conjugates = torch.matmul(torch.matmul(gs,xs), self.group.inv(gs))
        for irrep in self.group.lb_eigenspaces:
            chi = irrep.phase_function
            chi_vals_xs = chi.evaluate(xs)
            chi_vals_conj = chi.evaluate(conjugates)
            self.assertTrue(torch.allclose(chi_vals_xs, chi_vals_conj))

    def test_character_at_identity_equals_dimension(self):
        identity = torch.eye(self.dim, dtype=self.dtype, device=device).unsqueeze(0)
        for irrep in self.group.lb_eigenspaces:
            chi_val = irrep.phase_function.evaluate(identity).item()
            self.assertEqual(round(chi_val.real), irrep.dimension)
            self.assertEqual(round(chi_val.imag), 0)

    def test_characters_orthogonality(self):
        num_samples_x = 10**5
        xs = self.group.rand(num_samples_x)
        gammas = self.group.torus_representative(xs)
        num_irreps = len(self.group.lb_eigenspaces)
        scalar_products = torch.zeros((num_irreps, num_irreps), dtype=torch.cdouble)
        for a, b in itertools.product(enumerate(self.group.lb_eigenspaces), repeat=2):
            i, irrep1 = a
            j, irrep2 = b
            chi1, chi2 = irrep1.phase_function, irrep2.phase_function
            scalar_products[i, j] = torch.mean(torch.conj(chi1.chi(gammas)) * chi2.chi(gammas))
        print(torch.max(torch.abs(scalar_products - torch.eye(num_irreps, dtype=torch.cdouble))).item())
        self.assertTrue(torch.allclose(scalar_products, torch.eye(num_irreps, dtype=torch.cdouble), atol=5e-2))

    @unittest.skip('Very long, not part of the standard testing routine.')
    def test_translated_characters_basis_orthogonality(self):
        sys.stdout.write('\n')
        for irrep in self.group.lb_eigenspaces:
            torch.cuda.empty_cache()
            dim = irrep.dimension
            if self.dtype in (torch.double, torch.float64) and dim <= 20 or self.dtype == torch.cdouble and dim <= 10:
                basis = TranslatedCharactersBasis(representation=irrep)
                num_samples = 10 ** 4
                num_batches = 10 ** 2
                sc_prod = torch.zeros((dim ** 2, dim ** 2), dtype=torch.cdouble, device=device)
                for batch in range(num_batches):
                    xs = self.group.rand(num_samples)
                    xsb = basis.forward(xs)
                    sc_prod += torch.einsum('...i,...j->ij', xsb, xsb.conj()) / num_samples
                    sys.stdout.write('{} {} {}        \r'.format(irrep.index, irrep.dimension, batch))
                    sys.stdout.flush()
                sc_prod /= num_batches
                eyes = torch.eye(dim ** 2, dtype=torch.cdouble, device=device)
                print(irrep.index, dim, round(irrep.lb_eigenvalue, 2), torch.max(torch.abs(sc_prod-eyes)).item())
                self.assertTrue(torch.allclose(sc_prod, eyes, atol=5e-2))

    def test_sampler(self):
        true_ans = torch.eye(self.dim, dtype=self.dtype, device=device).reshape((1, self.dim, self.dim)).repeat(self.n, 1, 1)
        self.assertTrue(torch.allclose(vmap(self.group.difference)(self.x, self.x), true_ans))

    def _test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        # print(torch.std(cov_func-cov_prior)/torch.std(cov_func))
        # print(torch.max(torch.abs(cov_prior-cov_func)))
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=1e-2))

    def embed(self, f, x):
        phase, weight = self.sampler.phases[0], self.sampler.weights[0]  # [num_phase, ...], [num_phase]
        x_phase_inv = self.group.pairwise_diff(x, phase)
        eigen_embedding = f(x_phase_inv).view(x.size()[0], phase.size()[0])
        eigen_embedding = eigen_embedding / np.sqrt(self.sampler.phase_order)
        return eigen_embedding

    def _test_eigenfunction(self) -> None:
        x, y = self.group.rand(2), self.group.rand(2)
        y = x
        x_yinv = self.group.pairwise_diff(x, y)
        for eigenspace in self.group.lb_eigenspaces:
            f = eigenspace.phase_function
            dim_sq_f = eigenspace.dimension ** 2
            cov1 = f(x_yinv).view(2, 2)/dim_sq_f
            embed_x, embed_y = self.embed(f, x), self.embed(f, y)
            cov2 = (embed_x @ torch.conj(embed_y.T))/dim_sq_f
            print(cov1 - cov2)
            self.assertTrue(torch.allclose(cov1, cov2, atol=2e-1, rtol=2e-1))
            # print('passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)

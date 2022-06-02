import unittest
from parameterized import parameterized_class
import torch
from functorch import vmap
import numpy as np

from src.spectral_kernel import EigenbasisSumKernel, EigenbasisKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation
from src.utils import cartesian_prod

from src.spaces.so import SO
from src.spaces.su import SU



# Parametrized test, produces test classes called Test_Group.dim.order, for example, Test_SO.3.10 or Test_SU.2.5
@parameterized_class([
    {'group': SO, 'dim': 3, 'order': 10, 'dtype': torch.double},
    {'group': SO, 'dim': 5, 'order': 10, 'dtype': torch.double},
    {'group': SO, 'dim': 6, 'order': 10, 'dtype': torch.double},
    {'group': SU, 'dim': 2, 'order': 10, 'dtype': torch.cdouble},
    {'group': SU, 'dim': 3, 'order': 10, 'dtype': torch.cdouble},
    {'group': SU, 'dim': 4, 'order': 10, 'dtype': torch.cdouble},
], class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["group"].__name__}.'
                                                 f'{params_dict["dim"]}.{params_dict["order"]}')
class TestCompactLieGroups(unittest.TestCase):

    def setUp(self) -> None:
        # self.dim, self.order = 3, 10
        self.space = self.group(dim=self.dim, order=self.order)

        self.lengthscale, self.nu = 2.0, 5.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.space_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**4)

        self.n, self.m = 20, 20
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_sampler(self):
        true_ans = torch.eye(self.dim, dtype=self.dtype).reshape((1, self.dim, self.dim)).repeat(self.n, 1, 1)
        self.assertTrue(torch.allclose(vmap(self.space.difference)(self.x, self.x), true_ans))

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        # print(torch.std(cov_func-cov_prior)/torch.std(cov_func))
        # print(torch.max(torch.abs(cov_prior-cov_func)))
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=1e-2))

    def embed(self, f, x):
        phase, weight = self.sampler.phases[0], self.sampler.weights[0]  # [num_phase, ...], [num_phase]
        x_phase_inv = self.space.pairwise_diff(x, phase)
        eigen_embedding = f(x_phase_inv).view(x.size()[0], phase.size()[0])
        eigen_embedding = eigen_embedding / np.sqrt(
            self.sampler.phase_order)
        return eigen_embedding

    def test_eigenfunction(self) -> None:
        x, y = self.space.rand(2), self.space.rand(2)
        y = x
        x_yinv = self.space.pairwise_diff(x, y)
        for eigenspace in self.space.lb_eigenspaces:
            f = eigenspace.basis_sum
            cov1 = f(x_yinv).view(2, 2)
            embed_x, embed_y = self.embed(f, x), self.embed(f, y)
            cov2 = (embed_x @ torch.conj(embed_y.T))
            self.assertTrue(torch.allclose(cov1, cov2, atol=2e-1, rtol=2e-1))
            print('passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)

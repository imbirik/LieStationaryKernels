import unittest
from parameterized import parameterized_class
import torch
import numpy as np
from src.spaces.stiefel import Stiefel
from src.spectral_kernel import EigenbasisSumKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Parametrized test, produces test classes called Test_Group.dim.order, for example, Test_SO.3.10 or Test_SU.2.5
@parameterized_class([
    {'space': Stiefel, 'n': 3, 'm': 2, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 4, 'm': 2, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 4, 'm': 3, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 5, 'm': 2, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 5, 'm': 3, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 5, 'm': 4, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 6, 'm': 2, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 6, 'm': 3, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 6, 'm': 4, 'order': 10, 'dtype': torch.double},
    {'space': Stiefel, 'n': 6, 'm': 5, 'order': 10, 'dtype': torch.double},
], class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["space"].__name__}.'
                                                 f'{params_dict["n"]}.{params_dict["m"]}.{params_dict["order"]}')
class TestStiefel(unittest.TestCase):

    def setUp(self) -> None:
        # self.n, self.m = 5, 2
        self.average_order = 10**3
        self.space = self.space(n=self.n, m=self.m, order=self.order, average_order=self.average_order)

        self.lengthscale, self.nu = 1.5, 5.0
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**3)

        self.x_size, self.y_size = 5, 5
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_sampler(self):
        x_norm = torch.norm(self.x, dim=1)
        self.assertTrue(torch.allclose(x_norm, torch.ones_like(x_norm)))

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.x)
        cov_prior = self.sampler._cov(self.x, self.x)
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=5e-2))

    def embed(self, f, x):
        phase, weight = self.sampler.phases, self.sampler.weights[0]  # [num_phase, ...], [num_phase]
        x_phase_inv = self.space.pairwise_embed(x, phase)
        eigen_embedding = f(x_phase_inv).view(x.size()[0], phase.size()[0])
        eigen_embedding = eigen_embedding / np.sqrt(
            self.sampler.phase_order)
        return eigen_embedding

    def test_eigenfunction(self) -> None:
        x, y = self.space.rand(2), self.space.rand(2)
        x_yinv = self.space.pairwise_embed(x, y)
        for eigenspace in self.space.lb_eigenspaces:
            f = eigenspace.basis_sum
            dim_sq_f = f.representation.dimension ** 2
            cov1 = f(x_yinv).view(2, 2)/dim_sq_f
            embed_x, embed_y = self.embed(f, x), self.embed(f, y)
            cov2 = (embed_x @ torch.conj(embed_y.T))/dim_sq_f
            self.assertTrue(torch.allclose(cov1, cov2, atol=5e-2, rtol=5e-2))
            # print(torch.max(torch.abs(cov1 - cov2)).item())
            # print('passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)

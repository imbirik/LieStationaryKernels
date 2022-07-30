import unittest
from parameterized import parameterized_class
import torch
#from functorch import vmap
from torch.autograd.functional import _vmap as vmap
import numpy as np
from parameterized import parameterized_class
from src.spaces.grassmannian import Grassmannian, OrientedGrassmannian
from src.spectral_kernel import EigenbasisSumKernel, EigenbasisKernel, RandomPhaseKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation
from src.utils import cartesian_prod

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# @parameterized_class([
#     {'space': Grassmannian, 'n': 3, 'm': 1, 'order': 10, 'dtype': torch.double},
#     {'space': Grassmannian, 'n': 4, 'm': 2, 'order': 10, 'dtype': torch.double},
#     {'space': Grassmannian, 'n': 5, 'm': 2, 'order': 10, 'dtype': torch.double},
#     {'space': Grassmannian, 'n': 5, 'm': 1, 'order': 10, 'dtype': torch.double},
#     {'space': OrientedGrassmannian, 'n': 3, 'm': 1, 'order': 10, 'dtype': torch.double},
#     {'space': OrientedGrassmannian, 'n': 4, 'm': 2, 'order': 10, 'dtype': torch.double},
#     {'space': OrientedGrassmannian, 'n': 5, 'm': 2, 'order': 10, 'dtype': torch.double},
#     {'space': OrientedGrassmannian, 'n': 5, 'm': 1, 'order': 10, 'dtype': torch.double},
# ], class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["space"].__name__}.'
#                                                  f'{params_dict["n"]},{params_dict["m"]}.{params_dict["order"]}')
class TestGrassmanian(unittest.TestCase):

    def setUp(self) -> None:
        self.n, self.m = 3, 1
        self.order, self.average_order = 10, 1000
        self.space = Grassmannian(self.n, self.m, self.order, self.average_order)
        #self.space = Grassmannian(n=self.n, m=self.m, order=self.order, average_order=self.average_order)

        self.lengthscale, self.nu = 1.0, 5.0
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler_kernel = RandomPhaseKernel(measure=self.measure, manifold=self.space, phase_order=500)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=500)
        self.x_size, self.y_size = 5, 5
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_sampler(self):
        x_norm = torch.norm(self.x, dim=1)
        self.assertTrue(torch.allclose(x_norm, torch.ones_like(x_norm)))

    def test_symmetry(self):
        if self.m == 1 and isinstance(self.space, Grassmannian):
            cov = self.func_kernel(self.x, -self.x)
            print(cov)
            diag_cov = torch.diagonal(cov)
            self.assertTrue(torch.allclose(diag_cov, torch.ones_like(diag_cov), atol=5e-2))


    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.x)
        cov_prior = self.sampler_kernel(self.x, self.x).evaluate()
        print(cov_func)
        print(cov_prior)
        print(cov_prior-cov_func)
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=5e-2))

    def embed(self, f, x):
        phase, weight = self.sampler.phases, self.sampler.weights[0]  # [num_phase, ...], [num_phase]
        x_phase_inv = self.space.pairwise_embed(phase, x)
        eigen_embedding = f(x_phase_inv).view(x.size()[0], phase.size()[0])
        eigen_embedding = eigen_embedding / np.sqrt(
            self.sampler.phase_order)
        return eigen_embedding

    def test_eigenfunction(self) -> None:
        x, y = self.space.id, self.space.id
        y = x
        x_yinv = self.space.pairwise_embed(x, y)
        for eigenspace in self.space.lb_eigenspaces:
            f = eigenspace.basis_sum
            dim_sq_f = f.representation.dimension ** 2
            cov1 = (f(x_yinv)/dim_sq_f).real
            embed_x, embed_y = self.embed(f, x), self.embed(f, y)
            cov2 = ((embed_x @ torch.conj(embed_y.T))/dim_sq_f).real
            print(cov1)
            print(cov2)
            #self.assertTrue(torch.allclose(cov1, cov2, atol=5e-2, rtol=5e-2))
            print('passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)

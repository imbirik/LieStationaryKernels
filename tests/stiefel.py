import unittest
import torch
#from functorch import vmap
from torch.autograd.functional import _vmap as vmap
import numpy as np
from src.spaces.stiefel import Stiefel
from src.spectral_kernel import EigenbasisSumKernel, EigenbasisKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation
from src.utils import cartesian_prod

dtype = torch.double
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TestStiefel(unittest.TestCase):

    def setUp(self) -> None:
        self.n, self.m= 7, 4
        self.order, self.average_order = 10, 100
        self.space = Stiefel(n=self.n, m=self.m, order=self.order, average_order=self.average_order)

        self.lengthscale, self.nu = 1.1, 5.0
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.space_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**4)

        self.x_size, self.y_size = 10, 10
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_sampler(self):
        true_ans = torch.eye(self.n, dtype=dtype, device=device).reshape((1, self.n, self.n)).repeat(self.x_size, 1, 1)
        self.assertTrue(torch.allclose(vmap(self.space.difference)(self.x, self.x), true_ans))

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.x)
        cov_prior = self.sampler._cov(self.x, self.x)
        print(cov_prior)
        print(cov_func)
        print(cov_prior-cov_func)
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=5e-2))

    def embed(self, f, x):
        phase, weight = self.sampler.phases[0], self.sampler.weights[0]  # [num_phase, ...], [num_phase]
        x_phase_inv = self.space.pairwise_diff(x, phase)
        eigen_embedding = f(x_phase_inv).view(x.size()[0], phase.size()[0])
        eigen_embedding = eigen_embedding / np.sqrt(
            self.sampler.phase_order)
        return eigen_embedding

    def test_eigenfunction(self) -> None:
        x, y = self.space.rand(2), self.space.rand(2)
        x_yinv = self.space.pairwise_diff(x, y)
        for eigenspace in self.space.lb_eigenspaces:
            f = eigenspace.basis_sum
            dim_sq_f = f.representation.dimension ** 2
            cov1 = f(x_yinv).view(2, 2)/dim_sq_f
            embed_x, embed_y = self.embed(f, x), self.embed(f, y)
            cov2 = (embed_x @ torch.conj(embed_y.T))/dim_sq_f
            print(cov1)
            print(cov2)
            self.assertTrue(torch.allclose(cov1, cov2, atol=5e-2, rtol=5e-2))
            print('passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)

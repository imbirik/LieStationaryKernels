import unittest
import torch
from torch import vmap
import numpy as np
from src.spaces.so import SO
from src.spectral_kernel import EigenbasisSumKernel, EigenbasisKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation
from src.utils import cartesian_prod

dtype = torch.double


class TestSO(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 3, 10
        self.space = SO(dim=self.dim, order=self.order)

        self.lengthscale, self.nu = 2.0, 5.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.space_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**5)

        self.n, self.m = 20, 20
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_sampler(self):
        true_ans = torch.eye(self.dim, dtype=dtype).reshape((1, self.dim, self.dim)).repeat(self.n, 1, 1)
        self.assertTrue(torch.allclose(vmap(self.space.difference)(self.x, self.x), true_ans))

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        # print(torch.std(cov_func-cov_prior)/torch.std(cov_func))
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=1e-2))

    def embed(self, f, x):
        phase, weight = self.sampler.phases[0], self.sampler.weights[0]  # [num_phase, ...], [num_phase]
        x_, phase_ = cartesian_prod(x, phase)  # [len(x), num_phase, ...]
        eigen_embedding = f(x_, phase_)
        eigen_embedding = eigen_embedding / np.sqrt(
            self.sampler.phase_order)
        return eigen_embedding

    def test_eigenfunction(self) -> None:
        x, y = self.space.rand(2), self.space.rand(2)
        y = x
        x_, y_ = cartesian_prod(x, y)
        for eigenspace in self.space.lb_eigenspaces:
            f = eigenspace.basis_sum
        # for f in self.space.lb_eigenbases_sums:
            cov1 = f(x_, y_)
            embed_x, embed_y = self.embed(f, x), self.embed(f, y)
            cov2 = (embed_x @ torch.conj(embed_y.T))
            # print(cov2)
            # print(cov1)
        # self.assertTrue(torch.allclose(cov1, cov2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
import unittest
import torch
# import functorch
import numpy as np
from src.spaces.sphere import Sphere
from src.spectral_kernel import EigenbasisSumKernel, EigenbasisKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation

dtype = torch.double


class TestSphere(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 3, 8
        self.space = Sphere(self.dim, order=self.order)

        self.lengthscale, self.nu = 3.0, 2.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.space_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**5)

        self.n, self.m = 10, 20
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_kernel(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_space = self.space_kernel(self.x, self.y)
        self.assertTrue(torch.allclose(cov_space, cov_func))

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=1e-2))

    def test_sampler(self):
        true_ans = torch.ones(self.n, dtype=dtype)
        self.assertTrue(torch.allclose(torch.vmap(torch.dot)(self.x, self.x), true_ans))

    def test_harmonics(self):
        n = 100000
        x = torch.randn(n, self.dim + 1, dtype=dtype)
        x = x / torch.norm(x, dim=1, keepdim=True)
        for i in range(self.order):
            num_harmonics = self.space.lb_eigenspaces[i].dimension
            embed = self.space.lb_eigenspaces[i].basis(x)/np.sqrt(n)
            cov = torch.einsum('ij,kj->ik', embed, embed)
            eye = torch.eye(num_harmonics, dtype=dtype)
            self.assertTrue(torch.allclose(cov, eye, atol=1e-1))


if __name__ == '__main__':
    unittest.main(verbosity=2)


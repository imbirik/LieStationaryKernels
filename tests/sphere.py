import unittest
import torch
from torch.autograd.functional import _vmap as vmap

import numpy as np
from lie_stationary_kernels.spaces.sphere import Sphere
from lie_stationary_kernels.spectral_kernel import EigenbasisSumKernel
from lie_stationary_kernels.spectral_measure import SqExpSpectralMeasure
from lie_stationary_kernels.prior_approximation import RandomPhaseApproximation
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TestSphere(unittest.TestCase):

    def setUp(self) -> None:
        self.n, self.order = 2, 15
        self.space = Sphere(self.n, order=self.order)

        self.lengthscale, self.nu, self.variance = 1.0, 2.0, 5
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale, self.variance)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.space_kernel = EigenbasisSumKernel(measure=self.measure, manifold=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10**5)

        self.x_size, self.y_size = 10, 20
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_kernel(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_space = self.space_kernel(self.x, self.y)
        self.assertTrue(torch.allclose(cov_space, cov_func))

    def test_prior(self) -> None:
        self.y = self.x
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        print(cov_prior)
        print(cov_func)
        # print(torch.max(torch.abs(cov_prior-cov_func)).item())
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=5e-2))

    def test_sampler(self):
        true_ans = torch.ones(self.x_size, dtype=dtype, device=device)
        self.assertTrue(torch.allclose(vmap(torch.dot)(self.x, self.x), true_ans))

    def test_harmonics(self):
        n = 100000
        x = torch.randn(n, self.n + 1, dtype=dtype, device=device)
        x = x / torch.norm(x, dim=1, keepdim=True)
        for i in range(self.order):
            num_harmonics = self.space.lb_eigenspaces[i].dimension
            embed = self.space.lb_eigenspaces[i].basis(x)
            embed /= np.sqrt(n)
            cov = torch.einsum('ij,kj->ik', embed, embed)
            eye = torch.eye(num_harmonics, dtype=dtype, device=device)
            self.assertTrue(torch.allclose(cov, eye, atol=1e-1))


if __name__ == '__main__':
    unittest.main(verbosity=2)


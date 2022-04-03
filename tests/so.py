import unittest
import torch
from functorch import vmap
import numpy as np
from src.spaces.so import SO
from src.spectral_kernel import EigenFunctionKernel, EigenSpaceKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation
dtype = torch.double


class TestSO(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 3, 5
        self.space = SO(self.dim, order=self.order)

        self.lengthscale, self.nu = 1.0, 2.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenFunctionKernel(measure=self.measure, space=self.space)
        self.space_kernel = EigenFunctionKernel(measure=self.measure, space=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=10000)

        self.n, self.m = 10, 20
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_sampler(self):
        true_ans = torch.eye(self.dim, dtype=dtype).reshape((1, self.dim, self.dim)).repeat(self.n, 1, 1)
        self.assertTrue(torch.allclose(vmap(self.space.difference)(self.x, self.x), true_ans))

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        print(cov_func)
        print(cov_prior)
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=1e-2))

    def test_eigenfunction(self) -> None:
        x, y = self.space.rand(1)[0], self.space.rand(1)[0]
        self.assertTrue(torch.allclose(x @ x.T, torch.eye(self.dim, dtype=dtype)))
        for f in self.space.eigenfunctions:
            print(f(x, y))

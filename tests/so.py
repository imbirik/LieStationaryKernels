import unittest
import torch
import functorch
import numpy as np
from src.spaces.so import SO
from src.spectral_kernel import EigenFunctionKernel, EigenSpaceKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation
dtype = torch.double


class TestSphere(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 3, 8
        self.space = SO(self.dim, order=self.order)

        self.lengthscale, self.nu = 3.0, 2.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.func_kernel = EigenFunctionKernel(measure=self.measure, space=self.space)
        self.space_kernel = EigenFunctionKernel(measure=self.measure, space=self.space)
        self.sampler = RandomPhaseApproximation(kernel=self.func_kernel, phase_order=100000)

        n, m = 10, 20
        x, y = torch.randn(n, self.dim + 1, dtype=dtype), torch.randn(m, self.dim + 1, dtype=dtype)
        self.x, self.y = x / torch.norm(x, dim=1, keepdim=True), y / torch.norm(y, dim=1, keepdim=True)

    def test_prior(self) -> None:
        cov_func = self.func_kernel(self.x, self.y)
        cov_prior = self.sampler._cov(self.x, self.y)
        self.assertTrue(torch.allclose(cov_prior, cov_func, atol=1e-2))
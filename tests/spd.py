import unittest
import torch
from src.spaces.spd import SymmetricPositiveDefiniteMatrices
from src.spectral_kernel import RandomFourierFeaturesKernel
from src.prior_approximation import RandomFourierApproximation
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure

dtype = torch.double

class TestSPD(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 5, 100000
        self.space = SymmetricPositiveDefiniteMatrices(dim=self.dim, order=self.order)

        self.lengthscale, self.nu = 4.0, 5.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.kernel = RandomFourierFeaturesKernel(self.measure, self.space)
        self.sampler = RandomFourierApproximation(self.kernel)
        self.n, self.m = 5, 5
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_kernel(self):
        cov_kernel = self.kernel(self.x, self.x)
        cov_sampler = self.sampler._cov(self.x, self.x)
        print(cov_sampler)
        print(cov_kernel)
        self.assertTrue(torch.allclose(cov_sampler, cov_kernel, atol=5e-2))

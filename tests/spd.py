import unittest
import torch
import sys
from src.spaces.spd import SymmetricPositiveDefiniteMatrices
from src.spectral_kernel import RandomFourierFeaturesKernel
from src.prior_approximation import RandomFourierApproximation
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure

sys.setrecursionlimit(1000000)

dtype = torch.double
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TestSPD(unittest.TestCase):

    def setUp(self) -> None:
        print("device: ", device)
        self.dim, self.order = 5, 100000
        self.space = SymmetricPositiveDefiniteMatrices(dim=self.dim, order=self.order)

        self.lengthscale, self.nu = 10.0, 10+self.dim*(self.dim+1)/4
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu).to(device)

        self.kernel = RandomFourierFeaturesKernel(self.measure, self.space)
        print("kernel initialized")
        self.sampler = RandomFourierApproximation(self.kernel)
        print("sampler initialized")
        self.n, self.m = 5, 5
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_kernel(self):
        print("Comparing of pointwise evaluated and finite-dimensionally approximated kernels")
        cov_kernel = self.kernel(self.x, self.x).cpu()
        print(cov_kernel)
        cov_sampler = self.sampler._cov(self.x, self.x).cpu()
        print(cov_sampler)
        self.assertTrue(torch.allclose(cov_sampler, cov_kernel, atol=2e-2))

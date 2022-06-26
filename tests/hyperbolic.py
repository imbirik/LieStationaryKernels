import unittest
import torch
from src.spaces.hyperbolic import HyperbolicSpace, HypShiftExp
from src.spectral_kernel import RandomSpectralKernel
from src.prior_approximation import RandomFourierApproximation
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure

dtype = torch.float32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class TestHyperbolic(unittest.TestCase):

    def setUp(self) -> None:
        self.n, self.order = 5, 10**6
        self.space = HyperbolicSpace(n=self.n, order=self.order)

        self.lengthscale, self.nu = 5.0, 5.0 + self.space.dim
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.kernel = RandomSpectralKernel(self.measure, self.space)
        self.sampler = RandomFourierApproximation(self.kernel)
        self.x_size, self.y_size = 3, 3
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_kernel(self):
        print('dist', self.space._dist_to_id(self.x))
        cov_kernel = self.kernel(self.x, self.y)
        cov_sampler = self.sampler._cov(self.x, self.y)
        # print(cov_sampler)
        # print(cov_kernel)
        # print(torch.max(torch.abs(cov_sampler-cov_kernel)).item())
        self.assertTrue(torch.allclose(cov_sampler, cov_kernel, atol=5e-2))

    def test_spherical_function(self):
        self.y = self.x
        # print('norm', self.kernel.normalizer)
        shift = self.space.rand_phase(self.order)
        lmd = torch.randn(1, device=device, dtype=dtype).repeat(self.order)
        exp = HypShiftExp(lmd, shift, self.space)
        x_, y_ = self.space.to_group(self.x), self.space.to_group(self.y)
        x_embed, y_embed = exp(x_), exp(y_)
        cov1 = (x_embed @ (torch.conj(y_embed).T)).real/self.order

        x_yinv = self.space.pairwise_diff(x_, y_)
        x_yinv_embed = exp(x_yinv)  # (n*m,order)
        eye_embed = exp(self.space.id)  # (1, order)
        cov_flatten = x_yinv_embed @ (torch.conj(eye_embed).T)
        cov2 = cov_flatten.view(self.x.size()[0], self.y.size()[0]).real/self.order
        print(cov1)
        print(cov2)
        # print(torch.max(torch.abs(cov1 - cov2)).item())
        self.assertTrue(torch.allclose(cov1, cov2, atol=5e-2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
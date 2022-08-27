import unittest
import torch
from lie_stationary_kernels.spaces.hyperbolic import HyperbolicSpace, HypShiftExp
from lie_stationary_kernels.spectral_kernel import RandomSpectralKernel, RandomFourierFeatureKernel
from lie_stationary_kernels.prior_approximation import RandomFourierApproximation
from lie_stationary_kernels.spectral_measure import SqExpSpectralMeasure

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TestHyperbolic(unittest.TestCase):

    def setUp(self) -> None:
        self.n, self.order = 5, 10**6
        self.space = HyperbolicSpace(n=self.n, order=self.order)

        self.lengthscale, self.nu = 3.0, 5.0 + self.space.dim
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.kernel = RandomSpectralKernel(self.measure, self.space)
        self.rff_kernel = RandomFourierFeatureKernel(self.measure, self.space)
        self.sampler = RandomFourierApproximation(self.kernel)
        self.x_size, self.y_size = 5, 5
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_kernel(self):
        self.y = self.x
        print('dist', self.space._dist_to_id(self.x))
        cov_kernel = self.kernel(self.x, self.y)
        cov_rff = self.rff_kernel(self.x, self.y).evaluate()
        cov_sampler = self.sampler._cov(self.x, self.y)
        print(cov_sampler)
        print(cov_rff)
        print(cov_kernel)
        # print(torch.max(torch.abs(cov_sampler-cov_kernel)).item())
        self.assertTrue(torch.allclose(cov_sampler, cov_kernel, atol=5e-2))
        self.assertTrue(torch.allclose(cov_sampler, cov_rff, atol=5e-2))

    def test_spherical_function(self):
        self.y = self.x
        # print('norm', self.kernel.normalizer)
        shift = self.space.rand_phase(self.order)
        lmd = 10*torch.randn(1, device=device, dtype=dtype).repeat(self.order)
        exp = HypShiftExp(lmd, shift, self.space)
        x_, y_ = self.space.to_group(self.x), self.space.to_group(self.y)
        x_embed, y_embed = exp(x_), exp(y_)
        cov1 = (x_embed @ (torch.conj(y_embed).T)).real/self.order

        x_yinv = self.space.pairwise_diff(x_, y_)
        x_yinv_embed = exp(x_yinv)  # (n*m,order)
        eye_embed = exp(self.space.id)  # (1, order)
        cov_flatten = x_yinv_embed @ (torch.conj(eye_embed).T)
        cov2 = cov_flatten.view(self.x.size()[0], self.y.size()[0]).real/self.order
        # print(torch.max(torch.abs(cov1 - cov2)).item())
        print(cov1)
        print(cov2)
        self.assertTrue(torch.allclose(cov1, cov2, atol=5e-2))

    def test_dist(self):
        dist1 = self.space._dist_to_id(self.x)
        dist2 = self.space.pairwise_dist(self.x, self.space.id.view(-1, self.n)).squeeze()
        self.assertTrue(torch.allclose(dist1, dist2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
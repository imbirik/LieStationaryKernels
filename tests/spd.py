import unittest
import torch
import numpy as np
import scipy
from scipy import integrate
from src.spaces.spd import SymmetricPositiveDefiniteMatrices, SPDShiftExp
from src.spectral_kernel import RandomFourierFeaturesKernel
from src.prior_approximation import RandomFourierApproximation
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = torch.double
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.set_printoptions(precision=3)
#device = 'cpu'


def heat_kernel(x1, x2, t=1.0):
    def heat_kernel_unnorm(x1, x2, t=1.0):
        _, singular_values, _ = np.linalg.svd(x1.dot(np.linalg.inv(x2)))
        # Note: singular values that np.linalg.svd outputs are sorted, the following
        # code relies on this fact.
        H1, H2 = np.log(singular_values[0]), np.log(singular_values[1])
        assert (H1 >= H2)

        r_H_sq = H1 ** 2 + H2 ** 2
        alpha = H1 - H2

        # Non-integral part
        result = 1.0
        result *= np.exp(-r_H_sq / (4 * t))

        # Integrand
        def link_function(x):
            res = 1.0
            res *= (2 * x + alpha)
            res *= np.exp(-x * (x + alpha) / (2 * t))
            res *= pow(np.sinh(x) * np.sinh(x + alpha), -1 / 2)
            return res

        # Evaluating the integral

        # scipy.integrate.quad is much more accurate than np.trapz with
        # b_vals = np.logspace(-3., 1, 1000), at least if we believe
        # that Mathematica's NIntegrate is accurate.
        integral, error = scipy.integrate.quad(link_function, 0, np.inf)

        # print('H1 =', H1, 'H2 =', H2, 'Integral =', integral, 'Error =', error)

        result *= integral

        return result

    x0 = np.eye(2)
    return (heat_kernel_unnorm(x1, x2, t) / heat_kernel_unnorm(x0, x0, t))


class TestSPD(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 2, 100000
        self.space = SymmetricPositiveDefiniteMatrices(dim=self.dim, order=self.order)

        self.lengthscale, self.nu = 4.0, 5.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.kernel = RandomFourierFeaturesKernel(self.measure, self.space)
        self.sampler = RandomFourierApproximation(self.kernel)
        self.n, self.m = 5, 5
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

    def test_kernel(self):
        print(self.space._dist_to_id(self.x))
        cov_kernel = self.kernel(self.x, self.x)
        cov_sampler = self.sampler._cov(self.x, self.x)
        print(cov_sampler)
        print(cov_kernel)
        self.assertTrue(torch.allclose(cov_sampler, cov_kernel, atol=5e-2))

    def test_spherical_function(self):
        self.y = self.x
        print(self.kernel.normalizer)
        shift = self.space.rand_phase(self.order)
        lmd = torch.randn(1, self.dim, device=device, dtype=dtype).repeat(self.order, 1)
        exp = SPDShiftExp(lmd, shift, self.space)

        x_, y_ = self.space.to_group(self.x), self.space.to_group(self.y)
        x_embed, y_embed = exp(x_), exp(y_)
        cov1 = (x_embed @ (torch.conj(y_embed).T)).real/self.kernel.normalizer

        x_yinv = self.space.pairwise_diff(x_, y_)
        x_yinv_embed = exp(x_yinv)  # (n*m,order)
        eye_embed = exp(self.space.id)  # (1, order)
        cov_flatten = x_yinv_embed @ (torch.conj(eye_embed).T)
        cov2 = cov_flatten.view(self.x.size()[0], self.y.size()[0]).real/self.kernel.normalizer

        print(cov1)
        print(cov2)
        self.assertTrue(torch.allclose(cov1, cov2, atol=1e-2))

    def test_compare_with_sawyer(self):
        assert self.dim == 2, "dim should be equal to 2"
        cov_kernel = self.kernel(self.x, self.x).cpu().detach().numpy()
        print(cov_kernel)
        cov_sawyer = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                x_i, x_j = self.x[i].cpu().detach().numpy(), self.x[j].cpu().detach().numpy()
                cov_sawyer[i][j] = heat_kernel(x1=x_i, x2=x_j, t=self.lengthscale*self.lengthscale)
        print(cov_sawyer)
        self.assertTrue(np.allclose(cov_kernel, cov_sawyer))

import unittest
from parameterized import parameterized_class
import torch
import numpy as np
import scipy
from scipy import integrate
from lie_geom_kernel.spaces.spd import SymmetricPositiveDefiniteMatrices, SPDShiftExp
from lie_geom_kernel.spectral_kernel import RandomSpectralKernel
from lie_geom_kernel.prior_approximation import RandomFourierApproximation
from lie_geom_kernel.spectral_measure import SqExpSpectralMeasure
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.set_printoptions(precision=3)
#device = 'cpu'


def heat_kernel(x1, x2, t=1.0):
    def heat_kernel_unnorm(x1, x2, t=1.0):
        cl_1 = np.linalg.cholesky(x1)
        cl_2 = np.linalg.cholesky(x2)
        diff = np.linalg.inv(cl_2) @ cl_1
        _, singular_values, _ = np.linalg.svd(diff)
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


@parameterized_class([
    {'space': SymmetricPositiveDefiniteMatrices, 'n': 2, 'order': 10**4, 'dtype': torch.double},
    {'space': SymmetricPositiveDefiniteMatrices, 'n': 3, 'order': 10**4, 'dtype': torch.double},
    {'space': SymmetricPositiveDefiniteMatrices, 'n': 4, 'order': 10**4, 'dtype': torch.double},
], class_name_func=lambda cls, num, params_dict: f'Test_{params_dict["space"].__name__}.'
                                                 f'{params_dict["n"]}.{params_dict["order"]}')
class TestSPD(unittest.TestCase):

    def setUp(self) -> None:
        # self.n, self.order = 2, 10**4
        self.space = self.space(n=self.n, order=self.order)

        self.lengthscale, self.nu = 8.0, 5.0
        self.measure = SqExpSpectralMeasure(self.space.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

        self.kernel = RandomSpectralKernel(self.measure, self.space)
        self.sampler = RandomFourierApproximation(self.kernel)
        self.x_size, self.y_size = 5, 5
        self.x, self.y = self.space.rand(self.x_size), self.space.rand(self.y_size)

    def test_kernel(self):
        # print(self.space._dist_to_id(self.x))
        cov_kernel = self.kernel(self.x, self.x)
        cov_sampler = self.sampler._cov(self.x, self.x)
        # print(cov_sampler)
        # print(cov_kernel)
        print(torch.max(torch.abs(cov_sampler-cov_kernel)).item())
        self.assertTrue(torch.allclose(cov_sampler, cov_kernel, atol=5e-2))

    def test_spherical_function(self):
        # print(self.kernel.normalizer)
        shift = self.space.rand_phase(self.order)
        lmd = torch.randn(1, self.n, device=device, dtype=dtype).repeat(self.order, 1)
        exp = SPDShiftExp(lmd, shift, self.space)

        x_, y_ = self.space.to_group(self.x), self.space.to_group(self.y)
        x_embed, y_embed = exp(x_), exp(y_)
        cov1 = (x_embed @ (torch.conj(y_embed).T)).real/self.order

        x_yinv = self.space.pairwise_diff(x_, y_)
        x_yinv_embed = exp(x_yinv)  # (n*m,order)
        eye_embed = exp(self.space.id)  # (1, order)
        cov_flatten = x_yinv_embed @ (torch.conj(eye_embed).T)
        cov2 = cov_flatten.view(self.x.size()[0], self.y.size()[0]).real/self.order

        # print(cov1)
        # print(cov2)
        self.assertTrue(torch.allclose(cov1, cov2, atol=5e-2))

    # @unittest.skipUnless(self.n == 2, 'dim should be equal to 2')
    def test_compare_with_sawyer(self):
        if self.n != 2:
            self.skipTest('Only compare to Sawyer formula in dim 2')
        cov_kernel = self.kernel(self.x, self.x).cpu().detach().numpy()
        # print(cov_kernel)
        cov_sawyer = np.zeros((self.x_size, self.x_size))
        for i in range(self.x_size):
            for j in range(self.x_size):
                x_i, x_j = self.x[i].cpu().detach().numpy(), self.x[j].cpu().detach().numpy()
                cov_sawyer[i][j] = heat_kernel(x1=x_i, x2=x_j, t=self.lengthscale*self.lengthscale/2)
        # print(cov_sawyer)
        self.assertTrue(np.allclose(cov_kernel, cov_sawyer, atol=5e-2))


if __name__ == '__main__':
    unittest.main(verbosity=2)

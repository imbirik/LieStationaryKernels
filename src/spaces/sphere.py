import torch
import numpy as np
from scipy.special import loggamma

from src.space import HomogeneousSpace
import spherical_harmonics.torch
from spherical_harmonics.spherical_harmonics import SphericalHarmonicsLevel
from spherical_harmonics.fundamental_set import FundamentalSystemCache
from spherical_harmonics.spherical_harmonics import num_harmonics

dtype = torch.double


class Sphere(HomogeneousSpace):
    '''
    S^{dim} sphere is contained in R^{dim+1}
    '''

    def __init__(self, dim: int, order: int):
        '''
        :param dim: sphere dimension
        :param order: order of approximation. Number of eigenspaces under consideration.
        '''
        super(Sphere, self).__init__()
        self.dim = dim
        self.order = order

        fundamental_system = FundamentalSystemCache(self.dim + 1)

        self.eigenspaces = [NormalizedSphericalFunctions(self.dim, n, fundamental_system) for n in range(1, order + 1)]
        self.eigenfunctions = [ZonalSphericalFunction(self.dim, n) for n in range(1, self.order + 1)]
        self.eigenvalues = [n * (self.dim + n - 1) for n in range(1, self.order + 1)]
        self.eigenspaces_dims = [num_harmonics(self.dim + 1, n) for n in range(1, order + 1)]

    def dist(self, x, y):
        return torch.arccos(torch.dot(x, y))

    def rand(self, n=1):
        if n == 0:
            return None
        x = torch.randn(n, self.dim + 1, dtype=dtype)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x


class NormalizedSphericalFunctions(torch.nn.Module):
    def __init__(self, dimension, degree, fundamental_system):
        super(NormalizedSphericalFunctions, self).__init__()
        self.spherical_functions = SphericalHarmonicsLevel(dimension + 1, degree, fundamental_system)
        # 2 * S_{dim}/dim^2
        self.const = np.sqrt(2/(dimension+1)) *\
                     np.exp((np.log(np.pi) * (dimension + 1) / 2 - loggamma((dimension + 1) / 2)) / 2)

    def forward(self, x):
        return self.spherical_functions(x)


class ZonalSphericalFunction(torch.nn.Module):
    def __init__(self, dim, n):
        super(ZonalSphericalFunction, self).__init__()
        self.gegenbauer = GegenbauerPolynomials(alpha=(dim - 1) / 2., n=n)

        if n == 0:
            self.const = torch.tensor([1.])
        else:
            log_d_n = np.log(2*n+dim-1) + loggamma(n+dim-1) - loggamma(dim) - loggamma(n+1)
            self.const = torch.tensor([np.exp(log_d_n)/self.gegenbauer(1)])

    def forward(self, x, y):
        dist = torch.dot(x, y)
        return self.gegenbauer(dist) * self.const[0]


class GegenbauerPolynomials(torch.nn.Module):
    def __init__(self, alpha, n):
        super(GegenbauerPolynomials, self).__init__()
        self.alpha = alpha
        self.n = n
        self.coefficients = self.compute_coefficients()
        self.powers = torch.arange(0., self.n + 1.)

    def compute_coefficients(self):
        coefficients = torch.zeros(self.n + 1)
        # Two first polynomials is quite pretty
        # C_0 = 1, C_1 = 2\alpha*x
        if self.n == 0:
            coefficients[0] = 0
        if self.n == 1:
            coefficients[1] = 2 * self.alpha
        if self.n >= 2:
            # Other polynimials are given in Abramowitz & Stegun
            # c_{n-2k} = (-1)^k * 2^{n-2k} \Gamma(n-k+\alpha)/(\Gamma(\alpha)*k!(n-2k)!)
            for k in range(0, self.n // 2 + 1):
                sgn = (-1) ** k
                log_coeff = (self.n - 2 * k) * np.log(2) + loggamma(self.n - k + self.alpha) \
                            - loggamma(self.alpha) - loggamma(k + 1) - loggamma(self.n - 2 * k + 1)
                coeff = sgn * np.exp(log_coeff)
                coefficients[self.n - 2 * k] = coeff
        return coefficients

    def forward(self, x):
        # returns \sum c_i * x^i
        x_pows = torch.pow(x, self.powers)
        return torch.dot(x_pows, self.coefficients)

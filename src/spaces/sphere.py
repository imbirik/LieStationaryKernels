import torch
import numpy as np
from src.space import HomogeneousSpace
import spherical_harmonics.torch
from spherical_harmonics.spherical_harmonics import SphericalHarmonicsLevel
from spherical_harmonics.fundamental_set import FundamentalSystemCache


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

        fundamental_system = FundamentalSystemCache(self.dim+1)

        self.eigenspaces = [SphericalHarmonicsLevel(self.dim+1, n, fundamental_system) for n in range(order)]
        self.eigenfunctions = [ZonalSphericalHarmonic(self.dim, n) for n in range(self.order)]
        self.eigenvalues = [n * (self.dim + n - 1) for n in range(self.order)]

    def dist(self, x, y):
        return torch.arccos(torch.dot(x, y))


class ZonalSphericalHarmonic(torch.nn.Module):
    def __init__(self, dim, n):
        super(ZonalSphericalHarmonic, self).__init__()
        self.gegenbauer_polynomial = GegenbauerPolynomials(alpha=(dim - 1)/2., n=n)

    def forward(self, x, y):
        dist = torch.dot(x, y)
        return self.gegenbauer_polynomial(dist)


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
                sgn = (-1 ** k)
                log_coeff = (self.n - 2 * k) * np.log(2) + np.lgamma(self.n - k + self.alpha) \
                            - np.lgamma(self.alpha) - np.lgamma(k + 1) - np.lgamma(self.n - 2 * k + 1)
                coeff = sgn * np.exp(log_coeff)
                coefficients[self.n - 2 * k] = coeff
        return coefficients

    def forward(self, x):
        # returns \sum c_i * x^i
        x_pows = torch.pow(x, self.powers)
        return torch.dot(x_pows, self.coefficients)

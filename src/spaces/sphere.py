import torch
import numpy as np
from scipy.special import loggamma

from  src.utils import cartesian_prod
from src.space import AbstractManifold, LBEigenspaceWithSum, LBEigenspaceWithBasis
from geomstats.geometry.hypersphere import Hypersphere
import spherical_harmonics.torch
#from functorch import vmap
#from torch import vmap
from torch.autograd.functional import _vmap as vmap
from spherical_harmonics.spherical_harmonics import SphericalHarmonicsLevel
from spherical_harmonics.fundamental_set import FundamentalSystemCache
from spherical_harmonics.spherical_harmonics import num_harmonics

dtype = torch.double
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Sphere(AbstractManifold, Hypersphere):
    """
    S^{dim} sphere, in R^{dim+1}
    """
    def __init__(self, n: int, order: int):
        """
        :param dim: sphere dimension
        :param order: the order of approximation, the umber of Laplace-Beltrami eigenspaces under consideration.
        """
        self.n = n
        self.dim = n
        self.order = order
        AbstractManifold.__init__(self)
        Hypersphere.__init__(self, self.n)

        self.fundamental_system = FundamentalSystemCache(self.n + 1)
        self.lb_eigenspaces = [SphereLBEigenspace(index, manifold=self) for index in range(1, self.order + 1)]

    def dist(self, x, y):
        return torch.arccos(torch.dot(x, y))

    def rand(self, n=1):
        if n == 0:
            return None
        x = torch.randn(n, self.n + 1, dtype=dtype, device=device)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x

    def pairwise_diff(self, x, y):
        # x -- [n,d+1]
        # y -- [m, d+1]
        x_, y_ = cartesian_prod(x, y)
        x_flatten = torch.reshape(x_, (-1, self.n+1))
        y_flatten = torch.reshape(y_, (-1, self.n+1))
        return vmap(torch.dot)(x_flatten, y_flatten)


class SphereLBEigenspace(LBEigenspaceWithBasis):
    """The Laplace-Beltrami eigenspace for the sphere."""
    def __init__(self, index, *, manifold: Sphere):
        """
        :param index: the index of an eigenspace
        :param manifold: the "parent" manifold, an instance of Sphere
        """
        super().__init__(index, manifold=manifold)

    def compute_dimension(self):
        return num_harmonics(self.manifold.dim + 1, self.index)

    def compute_lb_eigenvalue(self):
        n = self.index
        return n * (self.manifold.dim + n - 1)

    def compute_basis_sum(self):
        return ZonalSphericalFunction(self.manifold.dim, self.index)

    def compute_basis(self):
        return NormalizedSphericalFunctions(self.manifold.dim, self.index, self.manifold.fundamental_system)


class NormalizedSphericalFunctions(torch.nn.Module):
    def __init__(self, dimension, degree, fundamental_system):
        super().__init__()
        self.spherical_functions = SphericalHarmonicsLevel(dimension + 1, degree, fundamental_system)
        # 2 * S_{dim}/dim^2
        self.const = np.sqrt(2/(dimension+1)) *\
                     np.exp((np.log(np.pi) * (dimension + 1) / 2 - loggamma((dimension + 1) / 2)) / 2)

    def forward(self, x):
        return self.spherical_functions(x.cpu()).to(device)


class ZonalSphericalFunction(torch.nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.gegenbauer = GegenbauerPolynomials(alpha=(dim - 1) / 2., n=n)
        self.forward = vmap(self._forward)

        if n == 0:
            self.const = torch.tensor([1.])
        else:
            log_d_n = np.log(2*n+dim-1) + loggamma(n+dim-1) - loggamma(dim) - loggamma(n+1)
            self.const = torch.tensor([np.exp(log_d_n)/self.gegenbauer(1.0)])

    def _forward(self, dist):
        return self.gegenbauer(dist) * self.const[0]


class GegenbauerPolynomials(torch.nn.Module):
    def __init__(self, alpha, n):
        super().__init__()
        self.alpha = alpha
        self.n = n
        self.coefficients = self.compute_coefficients()
        self.powers = torch.arange(0., self.n + 1., dtype=dtype, device=device)

    def compute_coefficients(self):
        coefficients = torch.zeros(self.n + 1, dtype=dtype, device=device)
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

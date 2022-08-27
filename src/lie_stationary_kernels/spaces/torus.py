import torch
from lie_stationary_kernels.space import CompactLieGroup, LBEigenspaceWithPhaseFunction, LieGroupCharacter
import math
import itertools
from lie_stationary_kernels.utils import cartesian_prod

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

j = torch.tensor([1j], device=device, dtype=torch.complex128).item()  # imaginary unit
pi = 2*torch.acos(torch.zeros(1)).item()


class Torus(CompactLieGroup):
    """Torus(dim), torus of degree dim represented as [0,1]^dim with glued boundary"""

    def __init__(self, n: int, order=20):
        """
        :param dim: dimension of the space
        :param order: the order of approximation, the number of representations calculated
        """
        self.n = n
        self.dim = n
        self.order = order
        self.Eigenspace = TorusLBEigenspace
        self.id = 0.5*torch.zeros(self.n, device=device, dtype=dtype).view(1, self.n)
        CompactLieGroup.__init__(self, order=order)

    def difference(self, x, y):
        return x - y

    def dist(self, x, y):
        """Batched geodesic distance"""
        diff = torch.abs(x-y)
        diff = torch.minimum(diff, 1 - diff)
        return torch.norm(diff, dim=1)

    def torus_representative(self, x):
        return x

    def pairwise_embed(self, x, y):
        x_, y_ = cartesian_prod(x, y)
        x_flatten = torch.reshape(x_, (-1, self.n))
        y_flatten = torch.reshape(y_, (-1, self.n))
        return x_flatten-y_flatten

    def pairwise_dist(self, x, y):
        """For n points x_i and m points y_j computed dist(x_i,y_j)"""
        diff = self.pairwise_embed(x, y)
        diff = torch.minimum(diff, 1-diff)
        dist = torch.norm(diff, dim=1).reshape(x.shape[0], y.shape[0])
        return dist

    def rand(self, n=1):
        return torch.rand((n, self.n), device=device, dtype=dtype)

    def generate_signatures(self, order):
        vals = range(-order//2, order//2+1)
        return list(itertools.product(vals, repeat=self.n))

    @staticmethod
    def inv(x: torch.Tensor):
        # (n, dim)
        return -x

class TorusLBEigenspace(LBEigenspaceWithPhaseFunction):
    """The Laplace-Beltrami eigenspace for the torus.
        This is \prod sin(k_ix_i) or sin"""
    def __init__(self, index, *, manifold: Torus):
        """
        :param signature: the signature of a representation
        :param manifold: the "parent" manifold, an instance of SO
        """
        self.index_ = torch.tensor(index, dtype=dtype, device=device)
        self.add = pi/2 * (self.index_ < 0)

        super().__init__(index, manifold=manifold)

    def compute_dimension(self):
        return 1

    def compute_lb_eigenvalue(self):
        return 4*math.pi*math.pi*sum([x**2 for x in self.index])

    def compute_basis_sum(self):
        return TorusCharacter(representation=self)


class TorusCharacter(LieGroupCharacter):
    """Representation character for torus"""
    def chi(self, x):
        # note sin(kx) = cos(pi/2 - kx)
        x_index = torch.sum(x * self.representation.index_[:, ...], dim=1)
        return torch.pow(torch.e, 2 * pi * j * x_index)
        #return #torch.prod(torch.cos(self.representation.add[..., :] +
                #                    2 * pi * x * self.representation.index_[..., :]), dim=1)


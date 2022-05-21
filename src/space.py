import torch
from abc import ABC, abstractmethod
import heapq
from src.utils import lazy_property


class AbstractManifold(torch.nn.Module, ABC):
    """Abstract base class for Compact Lie Group, Compact Homogeneous space or Symmetric Space"""
    def __init__(self):

        super().__init__()

        # self.dim = None
        # self.order = None

    @abstractmethod
    def dist(self, x, y):
        # compute distance between x and y
        pass

    # @abstractmethod
    # def difference(self, x, y):
    #     # Using group structure computes xy^{-1}
    #     pass

    @abstractmethod
    def rand(self, n=1):
        # returns random element with respect to haar measure
        pass


class LieGroup(AbstractManifold, ABC):
    """Lie group abstract base class"""
    def __init__(self, *, order: int):
        """
        Generate the list of signatures of representations and pick those with smallest LB eigenvalues.
        :param order: the order of approximation, the number of representations calculated
        """
        super().__init__()
        lb_eigenspaces = [self.Eigenspace(signature, manifold=self) for signature in self.generate_signatures(order)]
        self.lb_eigenspaces = heapq.nsmallest(order, lb_eigenspaces, key=lambda eig: eig.lb_eigenvalue)

    @abstractmethod
    def generate_signatures(self, order) -> list:
        """Generate signatures of representations to enumerate them."""
        raise NotImplementedError


class HomogeneousSpace(AbstractManifold, ABC):
    pass


class SymmetricSpace(AbstractManifold, ABC):
    pass


class LBEigenspace(ABC):
    """Laplace-Beltrami eigenspace abstract base class"""
    def __init__(self, index, *, manifold: AbstractManifold):
        """
        :param index: the index of an LB eigenspace
        :param manifold: the "parent" manifold
        """
        self.index = index
        self.manifold = manifold
        self.dimension = self.compute_dimension()
        self.lb_eigenvalue = self.compute_lb_eigenvalue()

    @abstractmethod
    def compute_dimension(self):
        """Compute the dimension of the Laplace-Beltrami eigenspace."""
        raise NotImplementedError

    @abstractmethod
    def compute_lb_eigenvalue(self):
        """Compute the Laplace-Beltrami eigenvalues of the eigenspace."""
        raise NotImplementedError


class LBEigenspaceWithBasis(LBEigenspace, ABC):
    """Laplace-Beltrami eigenspace ABC in case orthonormal basis is available"""
    @lazy_property
    # @property
    def basis(self):
        basis = self.compute_basis()
        return basis

    @abstractmethod
    def compute_basis(self):
        """Compute an orthonormal basis of the eigenspace."""
        raise NotImplementedError


class LBEigenspaceWithSum(LBEigenspace, ABC):
    """Laplace-Beltrami eigenspace ABC in case the sum function of an orthonormal basis paired products is available"""
    @lazy_property
    # @property
    def basis_sum(self):
        basis_sum = self.compute_basis_sum()
        return basis_sum

    @abstractmethod
    def compute_basis_sum(self):
        """Compute the sum of the orthonormal basis paired products."""
        raise NotImplementedError

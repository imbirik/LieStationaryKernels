import torch
from abc import ABC, abstractmethod


class AbstractSpace(torch.nn.Module, ABC):
    '''
    Abstract base class for Compact Lie Group, Compact Homogeneous space or Symmetric Space
    Contains
    '''
    def __init__(self):

        super().__init__()

        self.dim = None
        self.order = None

        self.lb_eigenbases = []
        self.lb_eigenbases_sums = []
        self.lb_eigenvalues = []
        self.lb_eigenspaces_dims = []

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

    # @abstractmethod
    # def rand_phase(self, n=1):
    #     pass


class LieGroup(AbstractSpace):
    pass


class HomogeneousSpace(AbstractSpace):
    pass


class SymmetricSpace(AbstractSpace):
    pass

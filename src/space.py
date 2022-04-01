import torch


class AbstractSpace(torch.nn.Module):
    '''
    Base class for Compact Lie Group, Compact Homogeneous space or Symmetric Space
    Contains
    '''
    def __init__(self):

        super(AbstractSpace, self).__init__()

        self.dim = None
        self.order = None

        self.eigenfunctions = []
        self.eigenspaces = []
        self.eigenvalues = []

        self.eigenspace_dims = []

    def dist(self, x, y):
        # compute distance between x and y
        pass

    def difference(self, x, y):
        # Using group structure computes xy^{-1}
        pass

    def rand(self, n=1):
        # returns random element with respect to haar measure
        pass

    def rand_phase(self, n=1):
        pass


class LieGroup(AbstractSpace):
    pass


class HomogeneousSpace(AbstractSpace):
    pass


class SymmetricSpace(AbstractSpace):
    pass

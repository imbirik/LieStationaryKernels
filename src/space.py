import torch


class AbstractSpace(torch.Module):
    '''
    Base class for Compact Lie Group, Compact Homogeneous space or Symmetric Space
    Contains
    '''
    def __init__(self):

        super(AbstractSpace, self).__init__()

        self.dim = None
        self.order = None

        self.eigenfunctions = []
        self.characters = []
        self.eigenvalues = []

    def dist(self, x, y):
        # compute distance between x and y
        pass

    def difference(self, x, y):
        # Using group structure computes xy^{-1}
        pass


class LieGroup(AbstractSpace):
    pass

class HomogeneousSpace(AbstractSpace):
    pass

class SymmetricSpace(AbstractSpace):
    pass

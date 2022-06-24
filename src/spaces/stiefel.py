import torch
import numpy as np
from src.spaces.so import SO, SOLBEigenspace
from src.space import HomogeneousSpace
from geomstats.geometry.stiefel import Stiefel as Stiefel_
#from functorch import vmap
from torch.autograd.functional import _vmap as vmap

dtype = torch.double
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Stiefel(HomogeneousSpace, Stiefel_):
    """Class for stiefel manifold represented as SO(n)/SO(m)"""
    """Elements represented as orthonormal frames"""
    def __init__(self, n, m, order=10, average_order=1000):
        assert n > m, "n should be greater than m"
        self.n, self.m = n, m
        G = SO(n, order=order)
        H = SO(m, order=1)
        HomogeneousSpace.__init__(self, G=G, H=H, average_order=average_order)
        Stiefel_.__init__(self, n, m)

    def embed_H_to_G(self, h):
        """Embed ortogonal matrix Q in the following way
            I 0
            0 Q
        """
        #h -- [b, m, m]
        zeros = torch.zeros((h.size()[0], self.n-self.m, self.m), device=device, dtype=dtype)
        zerosT = zeros.T
        eye = torch.eye(self.n-self.m, device=device, dtype=dtype).view(1, self.n).repeat(h.size()[0], 1)
        res = torch.stack([[eye, zeros], [zerosT, h]])
        return res

    def to_G(self, x):
        x_, _ = torch.linalg.qr(x, mode='complete')

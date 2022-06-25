import torch
import numpy as np
from src.spaces.so import SO, SOLBEigenspace
from src.space import HomogeneousSpace
from geomstats.geometry.stiefel import Stiefel as Stiefel_
#from functorch import vmap
from torch.autograd.functional import _vmap as vmap
from src.space import LBEigenspaceWithSum, LieGroupCharacter, AveragedLieGroupCharacter
dtype = torch.double
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Stiefel(HomogeneousSpace, Stiefel_):
    """Class for stiefel manifold represented as SO(n)/SO(m)"""
    """Elements represented as orthonormal frames"""
    def __init__(self, n, m, order=10, average_order=1000):
        assert n > m, "n should be greater than m"
        self.n, self.m = n, m
        self.n_m = n - m
        G = SO(n, order=order)
        H = SO(self.n_m, order=1)
        HomogeneousSpace.__init__(self, G=G, H=H, average_order=average_order)
        Stiefel_.__init__(self, n, m)

    def H_to_G(self, h):
        """Embed ortogonal matrix Q in the following way
            I 0
            0 Q
        """
        #h -- [b, n-m, n-m]
        zeros = torch.zeros((h.size()[0], self.m, self.n_m), device=device, dtype=dtype)
        zerosT = torch.transpose(zeros, dim0=-1, dim1=-2)
        eye = torch.eye(self.m, device=device, dtype=dtype).view(1, self.m, self.m).repeat(h.size()[0], 1, 1)
        l, r = torch.cat((eye, zerosT), dim=1), torch.cat((zeros, h), dim=-2)
        res = torch.cat((l, r), dim=-1)
        return res

    def M_to_G(self, x):
        g, _ = torch.linalg.qr(x, mode='complete')
        x_sign = torch.sign(x[:, :, 0])
        g_sign = torch.sign(g[:, :, 0])
        g = g * (x_sign * g_sign)[..., None]
        return g

    def G_to_M(self, g):
        # [b, n, n] -> [b,n, n-m]
        x = g[:, :, :self.m]
        return x

    def dist(self, x, y):
        raise NotImplementedError


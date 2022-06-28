import torch
import numpy as np
from src.spaces.so import SO, SOLBEigenspace
from src.spaces.stiefel import _SO
from src.space import HomogeneousSpace
from geomstats.geometry.stiefel import Stiefel as Stiefel_
from src.utils import hook_content_formula
import warnings
#from functorch import vmap
from torch.autograd.functional import _vmap as vmap
from src.space import LBEigenspaceWithSum, LieGroupCharacter, AveragedLieGroupCharacter
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class _SO_SO:
    """Helper class for sampling"""
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.SOn = _SO(n)
        self.SOm = _SO(m)
        self.dim = n*(n-1)//2 + m*(m-1)//2 -1

    def rand(self, n=1):
        h_u = self.SOn.rand(n)
        h_d = self.SOm.rand(n)
        zeros = torch.zeros((n, self.n, self.m), device=device, dtype=dtype)
        zerosT = torch.transpose(zeros, dim0=-1, dim1=-2)
        l, r = torch.cat((h_u, zerosT), dim=1), torch.cat((zeros, h_d), dim=-2)
        res = torch.cat((l, r), dim=-1)
        return res

class Grassmanian(HomogeneousSpace, Stiefel_):
    """Class for stiefel manifold represented as SO(n)/SO(m)xSO(n-m)"""
    """Elements represented as orthonormal frames of size m i.e. matrices nxm"""
    def __init__(self, n, m, order=10, average_order=1000):
        assert n > m, "n should be greater than m"
        self.n, self.m = n, m
        self.n_m = n - m
        G = SO(n, order=order)
        H = _SO_SO(self.m, self.n_m)
        HomogeneousSpace.__init__(self, G=G, H=H, average_order=average_order)
        Stiefel_.__init__(self, n, m)
        self.id = torch.zeros((self.n, self.m), device=device, dtype=dtype).fill_diagonal_(1.0)

    def H_to_G(self, h):
        return h

    def M_to_G(self, x):
        g, r = torch.linalg.qr(x, mode='complete')
        r_diag = torch.diagonal(r, dim1=-2, dim2=-1)
        r_diag = torch.cat((r_diag, torch.ones((x.shape[0], self.n_m), dtype=dtype, device=device)), dim=1)
        g = g * r_diag[:, None]
        diff = 2*(torch.all(torch.isclose(g[:, :, :self.m], x), dim=-1).type(dtype)-0.5)
        g = g * diff[..., None]
        det_sign_g = torch.sign(torch.det(g))
        g[:, :, -1] *= det_sign_g[:, None]
        assert torch.allclose(x, g[:, :, :x.shape[-1]])
        return g

    def G_to_M(self, g):
        # [b, n, n] -> [b,n, n-m]
        x = g[:, :, :self.m]
        return x

    def dist(self, x, y):
        raise NotImplementedError

    def close_to_id(self, x):
        x_ = x[:, :self.m, :self.m].reshape(x.shape[:-2] + (-1,))
        return torch.all(torch.isclose(x_, torch.zeros_like(x_)), dim=-1)

    def compute_inv_dimension(self, signature):
        return 1
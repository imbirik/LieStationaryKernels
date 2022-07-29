import torch
from src.spaces.so import SO
from src.space import HomogeneousSpace
from geomstats.geometry.stiefel import Stiefel as Stiefel_
from src.utils import hook_content_formula

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class _SO:
    """Helper class for sampling"""
    def __init__(self, n):
        self.n = n
        self.dim = n*(n-1)//2

    def rand(self, n=1):
        h = torch.randn((n, self.n, self.n), device=device, dtype=dtype)
        q, r = torch.linalg.qr(h)
        r_diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        q *= r_diag_sign[:, None]
        q_det_sign = torch.sign(torch.det(q))
        q[:, :, 0] *= q_det_sign[:, None]
        return q


class Stiefel(HomogeneousSpace, Stiefel_):
    """Class for Stiefel manifold represented as SO(n)/SO(m)"""
    """Elements represented as orthonormal frames"""
    def __init__(self, n, m, order=10, average_order=30):
        assert n > m, "n should be greater than m"
        self.n, self.m = n, m
        self.n_m = n - m
        g = SO(self.n, order=order)
        h = SO(self.n_m, order=0)
        HomogeneousSpace.__init__(self, g=g, h=h, average_order=average_order)
        Stiefel_.__init__(self, n, m)
        self.id = torch.zeros((self.n, self.m), device=device, dtype=dtype).fill_diagonal_(1.0)

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
        x_ = x[:, :, :self.m].reshape(x.shape[:-2] + (-1,))
        id_ = self.id.reshape((-1,))
        return torch.all(torch.isclose(x_, id_[None, ...]), dim=-1)

    def compute_inv_dimension(self, signature):
        m_ = min(self.m, self.n_m)
        if m_ < self.g.rank and signature[m_] > 0:
                return 0
        signature_abs = tuple(abs(x) for x in signature)
        inv_dimension = hook_content_formula(signature_abs, m_)
        return inv_dimension

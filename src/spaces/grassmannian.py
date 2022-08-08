import torch
from src.spaces.so import SO
from src.space import HomogeneousSpace
from geomstats.geometry.grassmannian import Grassmannian as Grassmannian_
from src.utils import cartesian_prod


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


class _SOxSO:
    """Helper class for sampling, represents SO(n) x SO(m)"""
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.so_n = _SO(n)
        self.so_m = _SO(m)
        self.dim = n*(n-1)//2 + m*(m-1)//2

    def rand(self, n=1):
        h_u = self.so_n.rand(n)
        h_d = self.so_m.rand(n)
        zeros = torch.zeros((n, self.n, self.m), device=device, dtype=dtype)
        zeros_t = torch.transpose(zeros, dim0=-1, dim1=-2)
        l, r = torch.cat((h_u, zeros_t), dim=1), torch.cat((zeros, h_d), dim=-2)
        res = torch.cat((l, r), dim=-1)
        return res

class _S_OxO:
    """Helper class for sampling, represents S( O(n) x O(m) )"""
    def __init__(self, n, m):
        self.n, self.m = n, m
        self.so_n = _SO(n)
        self.so_m = _SO(m)
        self.dim = n*(n-1)//2 + m*(m-1)//2 -1

    def block_diag(self, h_u, h_d):
        zeros = torch.zeros((h_u.shape[0], self.n, self.m), device=device, dtype=dtype)
        zeros_t = torch.transpose(zeros, dim0=-1, dim1=-2)
        l, r = torch.cat((h_u, zeros_t), dim=1), torch.cat((zeros, h_d), dim=-2)
        concatenated = torch.cat((l, r), dim=-1)
        return concatenated

    def rand(self, n=1):
        assert n % 2 == 0, "number of samples must be even."
        h_u = self.so_n.rand(n // 2)
        h_d = self.so_m.rand(n // 2)
        res_1 = self.block_diag(h_u, h_d).clone()

        h_u[:, :, -1] = h_u[:, :, -1] * -1
        h_d[:, :, -1] = h_d[:, :, -1] * -1
        res_2 = self.block_diag(h_u, h_d)

        res = torch.cat((res_1, res_2), dim=0)
        return res

class OrientedGrassmannian(HomogeneousSpace, Grassmannian_):
    """Class for oriented Grassmannian manifold represented as SO(n)/SO(m)xSO(n-m)"""
    """Elements represented as orthonormal frames of size m i.e. matrices nxm"""
    def __init__(self, n, m, order=10, average_order=50):
        assert n > m, "n should be greater than m"
        self.n, self.m = n, m
        self.n_m = n - m
        g = SO(n, order=order)
        h = _SOxSO(self.m, self.n_m)
        HomogeneousSpace.__init__(self, g=g, h=h, average_order=average_order)
        Grassmannian_.__init__(self, n, m)
        self.id = torch.zeros((self.n, self.m), device=device, dtype=dtype)\
            .fill_diagonal_(1.0).view((1, self.n, self.m))

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
        assert torch.allclose(torch.det(g), torch.ones((g.shape[0],), dtype=dtype, device=device))
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


class Grassmannian(HomogeneousSpace, Grassmannian_):
    """Class for Grassmannian manifold represented as SO(n)/S(O(m)xO(n-m))"""
    """Elements represented as orthonormal frames of size m i.e. matrices nxm"""
    def __init__(self, n, m, order=10, average_order=30):
        assert n > m, "n should be greater than m"
        self.n, self.m = n, m
        self.n_m = n - m
        g = SO(n, order=order)
        h = _S_OxO(self.m, self.n_m)
        HomogeneousSpace.__init__(self, g=g, h=h, average_order=average_order)
        Grassmannian_.__init__(self, n, m)
        self.id = torch.zeros((self.n, self.m), device=device, dtype=dtype).fill_diagonal_(1.0)
        self.id = self.id.view((1, self.n, self.m))

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
        """from https://pymanopt.org/docs/latest/_modules/pymanopt/manifolds/grassmann.html"""
        _, s, _ = torch.linalg.svd(torch.bmm(torch.transpose(x, dim0=-2, dim1=-1), y))
        s[s > 1] = 1
        s = torch.arccos(s)
        return torch.linalg.norm(s, dim=1)

    def pairwise_dist(self, x, y):
        x_, y_ = cartesian_prod(x, y)
        x_flatten = x_.reshape((-1, self.n, self.m))
        y_flatten = y_.reshape((-1, self.n, self.m))
        return self.dist(x_flatten, y_flatten).reshape(x.shape[0], y.shape[0])

    def close_to_id(self, x):
        x_ = x[:, :self.m, :self.m].reshape(x.shape[:-2] + (-1,))
        return torch.all(torch.isclose(x_, torch.zeros_like(x_)), dim=-1)

    def compute_inv_dimension(self, signature):
        return 1

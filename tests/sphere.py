import unittest
import torch
import functorch
from src.spaces.sphere import Sphere
from src.spectral_kernel import EigenFunctionKernel, EigenSpaceKernel
from src.spectral_measure import SqExpSpectralMeasure, MaternSpectralMeasure

dtype = torch.double

class TestSphere(unittest.TestCase):
    def test_sq_exp_kernel(self) -> None:
        dim = 8
        order = 5
        space = Sphere(dim, order=order)

        lengthscale = 1.0
        measure = SqExpSpectralMeasure(dim, lengthscale)

        sq_exp_func_kernel = EigenFunctionKernel(measure=measure, space=space)
        sq_exp_space_kernel = EigenSpaceKernel(measure=measure, space=space)

        n, m = 10, 20
        x, y = torch.randn(n, dim+1, dtype=dtype), torch.randn(m, dim+1, dtype=dtype)
        x, y = x/torch.norm(x, dim=1, keepdim=True), y/torch.norm(y, dim=1, keepdim=True)
        cov_func = sq_exp_func_kernel(x, y)
        cov_space = sq_exp_space_kernel(x, y)
        self.assertTrue(torch.allclose(cov_space, cov_func))

    def test_vmap(self):
        class Dot(torch.nn.Module):
            def __init__(self):
                super(Dot, self).__init__()

            def forward(self, x, y):
                return torch.dot(x, y)

        batched_dot = functorch.vmap(functorch.vmap(Dot()))
        x, y = torch.randn(10, 3, 5), torch.randn(10, 3, 5)
        self.assertTrue(torch.allclose(torch.einsum('abi,abi->ab', x, y), batched_dot(x, y)))



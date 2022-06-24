import lab as B
import numpy as np
import torch
import unittest
import geometric_kernels.torch  # noqa
from src.spaces.sphere import Sphere
from src.spaces.so import SO
from src.spectral_measure import SqExpSpectralMeasure
from src.spectral_kernel import EigenbasisSumKernel
from geometric_kernels.utils.manifold_utils import manifold_laplacian

_TRUNCATION_LEVEL = 15
_NU = 2.5


class TestHeatSphere(unittest.TestCase):

    def test_sphere_heat_kernel(self):
        # Parameters
        grid_size = 4
        nb_samples = 10
        n = 5

        # Create manifold
        #space = SO(n=n, order=_TRUNCATION_LEVEL)#
        space = Sphere(n=n, order=_TRUNCATION_LEVEL)
        # Generate samples
        ts = torch.linspace(0.1, 1, grid_size, requires_grad=True)
        xs = space.rand(nb_samples).requires_grad_(True)
        ys = xs

        # Define kernel
        measure = SqExpSpectralMeasure(space.dim, 1.0)
        kernel = EigenbasisSumKernel(measure=measure, manifold=space)

        # Define heat kernel function
        def heat_kernel(t, x, y):
            kernel.measure.lengthscale = torch.sqrt(2 * t.view(1))
            kernel.normalizer = 1
            return kernel(x, y)

        for t in ts:
            for x in xs:
                for y in ys:
                    # Compute the derivative of the kernel function wrt t
                    dfdt, _, _ = torch.autograd.grad(
                        heat_kernel(t, x[None], y[None]), (t, x, y), allow_unused=True
                    )
                    # Compute the Laplacian of the kernel on the manifold
                    egrad = lambda u: torch.autograd.grad(  # noqa
                        heat_kernel(t, u[None], y[None]), (t, u, y)
                    )[
                        1
                    ]  # noqa
                    fx = lambda u: heat_kernel(t, u[None], y[None])  # noqa
                    ehess = lambda u, h: torch.autograd.functional.hvp(fx, u, h)[1]  # noqa
                    lapf = manifold_laplacian(x, space, egrad, ehess)
                    # Check that they match
                    self.assertTrue(np.isclose(dfdt.detach().numpy(), lapf, atol=1.0e-5))
                    print("passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
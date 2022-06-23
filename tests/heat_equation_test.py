import lab as B
import numpy as np
import torch
import unittest
import geometric_kernels.torch  # noqa
from geometric_kernels.kernels.geometric_kernels import MaternKarhunenLoeveKernel
from geometric_kernels.spaces.hypersphere import Hypersphere
from geometric_kernels.utils.manifold_utils import manifold_laplacian

_TRUNCATION_LEVEL = 10
_NU = 2.5


class TestSPD(unittest.TestCase):

    def setUp(self) -> None:
        self.dim, self.order = 2, 100000
        self.space = SymmetricPositiveDefiniteMatrices(dim=self.dim, order=self.order)

        self.lengthscale, self.nu = 2.0, 5.0
        self.measure = SqExpSpectralMeasure(self.dim, self.lengthscale)
        #self.measure = MaternSpectralMeasure(self.dim, self.lengthscale, self.nu)

        self.kernel = RandomFourierFeaturesKernel(self.measure, self.space)
        self.sampler = RandomFourierApproximation(self.kernel)
        self.n, self.m = 5, 5
        self.x, self.y = self.space.rand(self.n), self.space.rand(self.m)

def test_sphere_heat_kernel():
    # Parameters
    grid_size = 4
    nb_samples = 10
    dimension = 3

    # Create manifold
    hypersphere = Hypersphere(dim=dimension)

    # Generate samples
    ts = torch.linspace(0.1, 1, grid_size, requires_grad=True)
    xs = torch.tensor(
        np.array(hypersphere.random_point(nb_samples)), requires_grad=True
    )
    ys = xs

    # Define kernel
    kernel = MaternKarhunenLoeveKernel(hypersphere, _TRUNCATION_LEVEL)
    params, state = kernel.init_params_and_state()
    params["nu"] = torch.tensor(torch.inf)

    # Define heat kernel function
    def heat_kernel(t, x, y):
        params["lengthscale"] = B.sqrt(2 * t)
        return kernel.K(params, state, x, y)

    for t in ts:
        for x in xs:
            for y in ys:
                # Compute the derivative of the kernel function wrt t
                dfdt, _, _ = torch.autograd.grad(
                    heat_kernel(t, x[None], y[None]), (t, x, y)
                )
                # Compute the Laplacian of the kernel on the manifold
                egrad = lambda u: torch.autograd.grad(  # noqa
                    heat_kernel(t, u[None], y[None]), (t, u, y)
                )[
                    1
                ]  # noqa
                fx = lambda u: heat_kernel(t, u[None], y[None])  # noqa
                ehess = lambda u, h: torch.autograd.functional.hvp(fx, u, h)[1]  # noqa
                lapf = manifold_laplacian(x, hypersphere, egrad, ehess)

                # Check that they match
                assert np.isclose(dfdt.detach().numpy(), lapf, atol=1.0e-6)

def manifold_laplacian(x: B.Numeric, manifold, egrad, ehess):
    r"""
    Computes the manifold Laplacian of a given function at a given point x.
    The manifold Laplacian equals the trace of the manifold Hessian, i.e.,
    :math:`\Delta_M f(x) = \sum_{i=0}^{D-1} \nabla^2 f(x_i, x_i)`,
    where :math:`[x_i]_{i=0}^{D-1}` is an orthonormal basis of the tangent
    space at x.
    :param x: point on the manifold at which to compute the Laplacian
    :param manifold: manifold space, based on geomstats
    :param egrad: Euclidean gradient of the function
    :param ehess: Euclidean Hessian of the function
    :return: manifold Laplacian
    References:
        [1] J. Jost.
            Riemannian geometry and geometric analysis. Springer, 2017.
            Chapter 3.1.
    """
    dim = manifold.dim

    onb = tangent_onb(manifold, B.to_numpy(x))
    result = 0.0
    for j in range(dim):
        cur_vec = onb[:, j]
        egrad_x = torch.to_numpy(egrad(x))
        ehess_x = torch.to_numpy(ehess(x, torch.tensor(x, cur_vec)))
        hess_vec_prod = manifold.ehess2rhess(B.to_numpy(x), egrad_x, ehess_x, cur_vec)
        result += manifold.metric.inner_product(
            hess_vec_prod, cur_vec, base_point=B.to_numpy(x)
        )

    return result


def tangent_onb(manifold, x):
    r"""
    Computes an orthonormal basis on the tangent space at x.
    :param manifold: manifold space, based on geomstats
    :param x: point on the manifold
    :return: [num, num] array containing the orthonormal basis
    """
    ambient_dim = manifold.dim + 1
    manifold_dim = manifold.dim
    ambient_onb = np.eye(ambient_dim)

    projected_onb = manifold.to_tangent(ambient_onb, base_point=x)

    projected_onb_eigvals, projected_onb_eigvecs = np.linalg.eigh(projected_onb)

    # Getting rid of the zero eigenvalues:
    projected_onb_eigvals = projected_onb_eigvals[ambient_dim - manifold_dim :]
    projected_onb_eigvecs = projected_onb_eigvecs[:, ambient_dim - manifold_dim :]

    assert np.all(np.isclose(projected_onb_eigvals, 1.0))

    return projected_onb_eigvecs
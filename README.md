Stationary Kernels and Gaussian Processes  
on Lie Groups and their Homogeneous Spaces
==========================================

This is a prototypical implementation for the methods described in `Stationary Kernels and Gaussian Processes on Lie Groups
and their Homogeneous Spaces`([part I](https://arxiv.org/abs/2208.14960), [part II](https://arxiv.org/abs/2301.13088)), a two-part series of papers by I. Azangulov, A. Smolensky, A. Terenin and V. Borovitskiy.

The library features (approximate) computational techniques for heat and Matérn kernels on compact Lie groups, their homogeneous spaces and non-compact symmetric spaces. It allows approximate kernel evaluation and differentiation, with positive semidefiniteness guarantees, and efficient sampling of the corresponding Gaussian process.

**Example.** Samples from a Gaussian process with heat kernel covariance on the torus $\mathbb{T}^2$, on the real projective plane $\mathrm{RP}^2$ and on the sphere $\mathbb{S}^2$:
<p align="center">
  <img src="/plots/torus_heat_kernel_sample_colors.png" width="250" />
  <img src="/plots/projective_space_heat_kernel_sample_colors.png" width="250" /> 
  <img src="/plots/sphere_heat_kernel_sample_colors.png" width="250" />
</p>

## Spaces of interest
The following spaces are implemented:
- Special orthogonal group $\mathrm{SO}(n)$ (*n*-by-*n* orthogonal matrices of determinant 1).
- Special unitary group $\mathrm{SU}(n)$ (*n*-by-*n* unitary matrices of determinant 1).
- Stiefel manifold $\mathrm{V}(n,m)$ (collections of *m* orthonormal vectors in the *n*-space), including hypersphere `S^n`.
- Grassmann manifold $\mathrm{Gr}(n,m)$ (*m*-dimensional subspaces in the *n*-space), including projective spaces `P^n` and oriented Grassmannians.
- Hyperbolic space $\mathbb{H}^n$.
- Symmetric positive-definite matrices $\mathrm{SPD}(n)$.

## Showcase

```python
from lie_stationary_kernels.spaces import Grassmannian
from lie_stationary_kernels.spectral_kernel import RandomPhaseKernel
from lie_stationary_kernels.spectral_measure import MaternSpectralMeasure
from lie_stationary_kernels.prior_approximation import RandomPhaseApproximation

# First of all let us choose a space
space = Grassmannian(n, m)
# Then select a spectral measure
measure = MaternSpectralMeasure(space.dim, lengthscale, nu, variance)
# Finally we create kernel and sampler
kernel = RandomPhaseKernel(measure, space)
sampler = RandomPhaseApproximation(kernel, phase_order)
# Create two sets of random points
x = space.rand(10)
y = space.rand(20)
# Then
cov = kernel(x,y) # is 10x20 matrix --- covariance matrix 
sample = sampler(x) # is 10x1 vector --- random realization at x
```

### Correspondence between spaces and kernels/samplers
Kernels:

1. With ```EigenSumKernel``` the covariance is computed exactly up to truncation using manifold Fourier features. Works with ```CompactLieGroup```. 

2. With ```RandomPhaseKernel``` the covariance is computed using generalized random phase Fourier features. Works with ```CompactLieGroup``` and ```СompactHomogeneousSpace```.

3. With ```RandomFourierKernel``` the covariance is computed using symmetric space random Fourier features. Works with ```NonCompactSymmetricSpace```.

Samplers:

1. ```RandomPhaseApproximation``` is used for compact spaces (```CompactHomogeneousSpace```, ```CompactLieGroup```).

2. ```RandomFourierApproximation``` is used for non-compact spaces (```NonCompactSymmetricSpace```).

## Installation and dependencies

1. [Optionally] Create virtual environment.

2. Install [PyTorch](https://pytorch.org/get-started/locally/).

3. [Optionally] To use the sphere and projective space, install [SphericalHarmonics](https://github.com/vdutor/SphericalHarmonics) following the instructions.

4. Install the library by running
```
pip install git+https://github.com/imbirik/LieStationaryKernel.git
```

5. To install in developer mode, clone the repository, enter its directory and run
```
pip install -e ./
```


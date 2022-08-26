# LieGeomKernel

`LieGeomKernel` is a library that implements calculations with heat and Matérn kernels
on compact Lie groups and their homogeneous spaces as well as non-compact symmetric spaces.

This is a prototypical implementation for the methods described in `Stationary Kernels and Gaussian Processes on Lie Groups
and their Homogeneous Spaces`, a two-part paper by I. Azangulov, A. Smolensky, A. Terenin and V. Borovitskiy.

## Spaces of interest
The following spaces are implemented:
- Special orthogonal group `SO(n)` (*n*-by-*n* orthogonal matrices of determinant 1),
- Special unitary group `SU(n)` (*n*-by-*n* unitary matrices of determinant 1),
- Stiefel manifold `S(n,m)` (collections of *m* orthonormal vectors in the *n*-space), including hypersphere `S^n`,
- Grassmann manifold `Gr(n,m)` (*m*-dimensional subspaces in the *n*-space), including projective spaces `P^n` and oriented Grassmannians,
- Hyperbolic space `H^n`,
- Symmetric positive-definite matrices `SPD(n)`.

## Showcase

Alas, it's only the code for now...
```
# Some imports
import torch
from src.spaces import Grassmanian
from src.spectral_kernel import RandomPhaseKernel
from src.spectral_measure import MaternSpectralMeasure
from src.prior_approximation import RandomPhaseApproximation

# First of all let us choose a space
space = Grassmanian(n, m)
# Then select a spectral measure
measure = MaternSpectralMeasure(space.dim, lengthscale, nu, variance)
# Finally we create kernel and sampler
kernel = RandomPhaseKernel(measure, space)
sampler = RandomPhaseApproximation(kernel, phase_order)
# Create two sets of random points
x = space.rand(10)
y = space.rand(20)
# Then
Cov = kernel(x,y) # is 10x20 matrix --- covariance matrix 
sample = sampler(x) # is 10x1 vector --- random realization at x.
```

### Correspondence between spaces and kernels/samplers
Kernels:

1. With ```EigenSumKernel``` the covariance is computed precisely, but works only for ```CompactLieGroup```. 

2. With ```RandomPhaseKernel``` the covariance is computed using low-rank approximation, it is suitable for ```ompactHomogeneousSpace``` and ```CompactLieGroup```.

3. With ```RandomFourierKernel``` the covariance is computed using low-rank approximation, it is suitable for ```NonCompactSymmetricSpace```

Samplers:

1. ```RandomPhaseApproximation``` is used in compact case (```CompactHomogeneousSpace```, ```CompactLieGroup```)

2. ```RandomFourierApproximation``` is used in non-compact case (```NonCompactSymmetricSpace```)

## Installation and dependencies

Under construction...

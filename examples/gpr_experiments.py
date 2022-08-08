#%%

from src.spaces import Grassmannian, OrientedGrassmannian, HyperbolicSpace, SO, \
    SymmetricPositiveDefiniteMatrices, Sphere, Stiefel, SU
from src.spectral_kernel import RandomSpectralKernel, EigenbasisSumKernel, RandomFourierFeatureKernel, RandomPhaseKernel
from src.prior_approximation import RandomPhaseApproximation, RandomFourierApproximation
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
from src.space import CompactLieGroup, HomogeneousSpace
from examples.gpr_model import ExactGPModel, train
from torch.nn import MSELoss
import gpytorch
import torch
import sys
import pandas as pd
import os
#%%
sys.setrecursionlimit(2000)
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
#let's choose some space
#%%

def main(space, n, m):
    print(type(space).__name__, n, m)

    lengthscale, nu, variance = 0.5, 1.5, 4.0
    # measure = SqExpSpectralMeasure(space.dim, lengthscale, variance=variance)
    _measure = MaternSpectralMeasure(space.dim, lengthscale, nu, variance)

    if isinstance(space, CompactLieGroup) or isinstance(space, Sphere):
        _kernel = EigenbasisSumKernel(_measure, space)
        f = RandomPhaseApproximation(_kernel)

    elif isinstance(space, HomogeneousSpace):
        _kernel = RandomPhaseKernel(_measure, space, phase_order=25)
        f = RandomPhaseApproximation(_kernel)

    else:
        _kernel = RandomFourierFeatureKernel(_measure, space)
        f = RandomFourierApproximation(_kernel)

    # rand_points = space.rand(3)
    # def f(x):
    #     dists = space.pairwise_dist(x, rand_points)
    #     mean_dist = torch.sum(torch.pow(dists, 4), dim=1)
    #     return mean_dist.real

    n_train, n_test = 30, 100
    train_x, test_x = space.rand(n_train), space.rand(n_test)
    train_y, test_y = f(train_x), f(test_x)
    train_x, test_x = train_x.reshape(n_train, -1), test_x.reshape(n_test, -1)
    #%%
    #configure kernel

    lengthscale, nu, variance = 1, 1.5, 1.0
    #measure = SqExpSpectralMeasure(space.dim, lengthscale, variance=variance)
    measure = MaternSpectralMeasure(space.dim, lengthscale, nu)

    if isinstance(space, CompactLieGroup) or isinstance(space, Sphere):
        kernel = EigenbasisSumKernel(measure, space)
    elif isinstance(space, HomogeneousSpace):
        kernel = RandomPhaseKernel(measure, space, phase_order=25)
    else:
        kernel = RandomFourierFeatureKernel(measure, space)

    #sampler = RandomPhaseApproximation(spectral_kernel)

    #%%

    #%%
    # train model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel, space, point_shape=(n, m)).to(device=device)
    train(model, train_x, train_y, 1500, 500, lr=0.1)

    model.eval()
    with torch.no_grad():
        pred_f = model(test_x)
    pred_y = pred_f.mean
    ms_error = MSELoss()(pred_y, test_y).item()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    mll_test = mll(pred_f, test_y).item()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #euclidean_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    euclidean_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(1.5))
    #euclidean_kernel = gpytorch.kernels.MaternKernel(1.5)

    euclidean_kernel.outputscale = 1
    euclidean_model = ExactGPModel(train_x, train_y, likelihood, euclidean_kernel, space).to(device=device)
    train(euclidean_model, train_x, train_y, 1500, 500)

    euclidean_model.eval()
    with torch.no_grad():
        euclidean_f = euclidean_model(test_x)
    euclidean_pred_y = euclidean_f.mean
    euclidean_ms_error = MSELoss()(euclidean_pred_y, test_y).item()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(euclidean_model.likelihood, euclidean_model)
    euclidean_mll_test = mll(euclidean_f, test_y).item()

    data_variance = torch.var(test_y, unbiased=False).item()
    print("data variance:", torch.var(test_y, unbiased=False).item())
    print("geometric mse and mll:", ms_error, mll_test)
    print("euclidean mse and mll:", euclidean_ms_error, euclidean_mll_test)
    print("#"*100)
    return data_variance, ms_error, mll_test, euclidean_ms_error, euclidean_mll_test


if __name__ == "__main__":
    groups = [(SU(2), 2, 2), (SU(3), 3, 3), (SU(4), 4, 4),
              (SO(3), 3, 3), (SO(4), 4, 4), (SO(5), 5, 5), (SO(6), 6, 6),
              (Grassmannian(3, 1), 3, 1), (Grassmannian(3, 2), 3, 2), (Grassmannian(4, 2), 4, 2),
              (Sphere(2), 3, 1), (Sphere(3), 4, 1), (Sphere(4), 5, 1),
              (HyperbolicSpace(2), 2, 1), (HyperbolicSpace(3), 3, 1), (HyperbolicSpace(4), 4, 1),
              (HyperbolicSpace(5), 5, 1), (HyperbolicSpace(6), 6, 1),
              (SymmetricPositiveDefiniteMatrices(2), 2, 2), (SymmetricPositiveDefiniteMatrices(3), 3, 3),
              (SymmetricPositiveDefiniteMatrices(4), 4, 4), (SymmetricPositiveDefiniteMatrices(5), 5, 5),
              ]
    results = []
    for args in groups:
        space, n, m = args
        name = type(space).__name__ + str(n) + str(m)
        results.append([name, *main(*args)])

    results = pd.DataFrame(results,
            columns=["space", "data_variance", "geometric_mse", "geometric_mll", "euclidean_mse", "euclidean_mll"])
    results.to_csv("gpr_results_matern32.csv")
    print(results)
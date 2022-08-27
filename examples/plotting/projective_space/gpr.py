#%%

import torch
import gpytorch
from src.spaces import ProjectiveSpace
from src.spectral_kernel import EigenbasisSumKernel
from src.prior_approximation import RandomPhaseApproximation
from src.lie_geom_kernel.spectral_measure import MaternSpectralMeasure
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from examples.plotting.drawing_utils import save_points
from examples.gpr_model import ExactGPModel, train
import math
import random
import pandas as pd

random.seed(1111)
torch.manual_seed(1111)

plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

space = ProjectiveSpace(n=2, order=20)

#%%

lengthscale, nu, variance = 0.5, 1.5, 1.0
#measure = SqExpSpectralMeasure(space.dim, lengthscale, variance=1.0)
measure = MaternSpectralMeasure(space.dim, lengthscale, nu, variance)

#%%

kernel = EigenbasisSumKernel(measure, space)

def sphere(r=1., nlats=301, nlons=301):
    phi, theta = np.mgrid[0:np.pi/2:nlats * 1j, 0:2 * np.pi:nlons * 1j]

    z = (r * np.sin(phi) * np.cos(theta)).reshape(-1, 1)
    y = (r * np.sin(phi) * np.sin(theta)).reshape(-1, 1)
    x = (r * np.cos(phi)).reshape(-1, 1)

    return np.concatenate((x, y, z), axis=1)

points = sphere()
boundary = set(np.nonzero(points[:, 0] < 0.01)[0].tolist())

rand_points = torch.tensor([[math.sqrt(1/4), math.sqrt(3/4), 0]],  # [-math.sqrt(1/3), math.sqrt(1/3), math.sqrt(1/3)]],
                           dtype=dtype, device=device)
def f(x):
    dists = space.pairwise_dist(x, rand_points)
    dists = torch.clip(dists+1e-9, min=0.0)
    mean_dist = torch.sum(torch.pow(dists, 1), dim=1)
    return mean_dist

n_train = 50
x_train = space.rand(n_train)
y_train = f(x_train)

x_train_numpy = x_train.cpu().detach().numpy()

x_train_grid = []
for x in x_train_numpy:
    if x[0] < 0:
        x = -x
    x_train_grid.append(np.argmin(np.linalg.norm(points-x, axis=1))+1)

pd.DataFrame(x_train_grid).to_csv("train_points_id.csv", index=False, header=False)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood, kernel, space, point_shape=(3,)).to(device=device)
train(model, x_train, y_train, 1800, 600, lr=1)
model.eval()
sampler = RandomPhaseApproximation(model.covar_module, phase_order=300)

ground_truth = []
posterior_mean = []
posterior_variance = []
posterior_sample = []

batch_size = 100

sigma2 = model.likelihood.noise.item()
eps_train = torch.normal(0.0, sigma2, size=(n_train,), dtype=dtype, device=device)
sample_train = sampler(x_train)
noise_cov = sigma2 * torch.eye(n_train, device=device, dtype=dtype)
cov_inv = torch.linalg.inv((model.covar_module(x_train, x_train) + noise_cov))

for i in tqdm(range(len(points)//batch_size+1)):
    if (i+1)*batch_size > len(points):
        x_slice = torch.tensor(points[i*batch_size:], device=device, dtype=dtype)
    else:
        x_slice = torch.tensor(points[i*batch_size: (i+1)*batch_size], dtype=dtype, device=device)
    x_slice = x_slice.view(-1, 3)
    ground_truth.append(f(x_slice).cpu().detach().numpy())
    with torch.no_grad():
        pred_x = model(x_slice)
        posterior_mean.append(pred_x.mean.cpu().detach().numpy())
        posterior_variance.append(pred_x.variance.cpu().detach().numpy())
        cov_with_train = model.covar_module(x_slice, x_train)
        sample = sampler(x_slice)
        posterior_sample_ = model.mean_module.constant + sample + \
                            cov_with_train @ (cov_inv @ (y_train - model.mean_module.constant - sample_train-eps_train))
        posterior_sample.append(posterior_sample_.cpu().detach().numpy())

results = {"ground_truth": ground_truth, "posterior_mean": posterior_mean,
           "posterior_variance": posterior_variance, "posterior_sample": posterior_sample
          }


for key, arr in results.items():
    arr = np.concatenate(arr)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter(points[:, 2], points[:, 1], points[:, 0], c=arr)
    fig.colorbar(p)
    plt.savefig(key+".png")
    plt.savefig("projective_space_sample.png")
    save_points(np.squeeze(points), arr, boundary_v=boundary, space_name="projective_space", color_source=key)

plt.pause(1000)
#%%

import torch
from src.spaces import Torus
from src.spectral_kernel import EigenbasisSumKernel
from src.prior_approximation import RandomPhaseApproximation
from src.lie_geom_kernel.spectral_measure import SqExpSpectralMeasure
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from examples.plotting.drawing_utils import save_points
from scipy.spatial import Delaunay
plt.ion()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pi = 2*torch.acos(torch.zeros(1)).item()

#%%

space = Torus(n=2, order=31)

#%%

lengthscale, nu = 0.05, 1.5
measure = SqExpSpectralMeasure(space.dim, lengthscale, variance=1.0)

#%%

kernel = EigenbasisSumKernel(measure, space)
sampler = RandomPhaseApproximation(kernel, phase_order=100)

lspace = np.linspace(0, 1, 200, endpoint=True).reshape(200)
points = np.dstack(np.meshgrid(lspace, lspace)).reshape(-1, 2)

kernel_values = []
samples = []
batch_size = 100

point = torch.tensor([[0.125, 0.625]], dtype=dtype, device=device)
for i in tqdm(range(len(points)//batch_size+1)):
    if (i+1)*batch_size > len(points):
        x_slice = torch.tensor(points[i*batch_size:], device=device, dtype=dtype)
    else:
        x_slice = torch.tensor(points[i*batch_size: (i+1)*batch_size], dtype=dtype, device=device)
    x_slice = x_slice.view(-1, 2)
    kernel_values.append(kernel(x_slice, point).cpu().detach().numpy())
    samples.append(sampler(x_slice).cpu().detach().numpy())
kernel_values = np.concatenate(kernel_values)
kernel_values = np.squeeze(kernel_values)

samples = np.concatenate(samples)
samples = np.squeeze(samples)

points = np.squeeze(points)


def to_3d_torus(points, a=0.6, c=0.2):
    points = 2*pi*points
    num_points = points.shape[0]
    x = ((a + c * np.cos(points[:, 0])) * np.cos(points[:, 1])).reshape(num_points, 1)
    y = ((a + c * np.cos(points[:, 0])) * np.sin(points[:, 1])).reshape(num_points, 1)
    z = c * (np.sin(points[:, 0]).reshape(num_points, 1))
    return np.concatenate([x, y, z], axis=1)


triangulation = Delaunay(points)
points = to_3d_torus(points)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=kernel_values)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.savefig("torus_heat_kernel_value.png")
save_points(points, kernel_values, "torus", "heat_kernel_value", triangulation=triangulation)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=samples)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.savefig("torus_heat_kernel_sample.png")
save_points(points, samples, "torus", "heat_kernel_sample", triangulation=triangulation)

#plt.pause(100)
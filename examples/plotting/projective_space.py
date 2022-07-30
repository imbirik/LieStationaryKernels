#%%

import torch
from src.spaces.grassmannian import Grassmannian
from src.spectral_kernel import RandomPhaseKernel
from src.prior_approximation import RandomPhaseApproximation
from src.spectral_measure import SqExpSpectralMeasure
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from examples.plotting.drawing_utils import save_points, save_points_with_color

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

space = Grassmannian(n=3, m=1, order=20, average_order=100)

#%%

lengthscale, nu = 0.25, 5.0 + space.dim
measure = SqExpSpectralMeasure(space.dim, lengthscale)
#self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

#%%

kernel = RandomPhaseKernel(measure, space, 100)
sampler = RandomPhaseApproximation(kernel, phase_order=100)


def sphere(r=1., nlats=101, nlons=101):
    phi, theta = np.mgrid[0:np.pi/2:nlats * 1j, 0:2 * np.pi:nlons * 1j]

    x = (r * np.sin(phi) * np.cos(theta)).reshape(-1, 1)
    y = (r * np.sin(phi) * np.sin(theta)).reshape(-1, 1)
    z = (r * np.cos(phi)).reshape(-1, 1)

    return np.concatenate((x, y, z), axis=1)

points = sphere()
print(points.shape)

indices = np.nonzero(points[:, 2] < 0)[0]
points[indices] = -1 * points[indices]
x_boundary = set(np.nonzero(points[:, 2] < 0.01)[0].tolist())
samples = []
batch_size = 500
for i in tqdm(range(len(points)//batch_size+1)):
    if (i+1)*batch_size > len(points):
        x_slice = torch.tensor(points[i*batch_size:], device=device, dtype=dtype)
    else:
        x_slice = torch.tensor(points[i*batch_size: (i+1)*batch_size], dtype=dtype, device=device)
    x_slice = x_slice.view(-1, 3, 1)
    samples.append(sampler(x_slice).cpu().detach().numpy())

samples = np.concatenate(samples)

#%%
samples = np.squeeze(samples)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=samples)
#ax.scatter(y_boundary[:, 0], y_boundary[:, 1], y_boundary[:, 2], c='red')
plt.show()

save_points(np.squeeze(points), samples, boundary_v=x_boundary, filename="projective_sample")

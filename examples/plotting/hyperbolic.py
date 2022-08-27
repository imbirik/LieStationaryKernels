#%%

import torch
from src.spaces.hyperbolic import HyperbolicSpace
from src.spectral_kernel import RandomSpectralKernel
from src.prior_approximation import RandomFourierApproximation
from src.lie_geom_kernel.spectral_measure import SqExpSpectralMeasure
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from examples.plotting.drawing_utils import save_points, save_points_with_color

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

order = 10**3
space = HyperbolicSpace(n=2, order=order)

#%%

lengthscale, nu = 0.2, 5.0 + space.dim
measure = SqExpSpectralMeasure(space.dim, lengthscale)
#self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

#%%

kernel = RandomSpectralKernel(measure, space)
sampler = RandomFourierApproximation(kernel)

#%%

def disk_to_hyperboloid(x):
    x_sq_norm = np.sum(x * x, axis=1)
    t = ((x_sq_norm + 1)/(1-x_sq_norm))
    y = x * (1 + t)[:, None]
    return np.concatenate((y, t.reshape(-1, 1)), axis=1)

def hyperboloid_to_disk(y):
    x = y[:, :2]/((1+y[:, 2])[:, None])
    return x


lspace = np.linspace(-1, 1, num=500)

#lspace = torch.sign(lspace) * torch.square(torch.abs(lspace))
x = np.dstack(np.meshgrid(lspace, lspace)).reshape(-1, 2)
x = x[np.linalg.norm(x, axis=1) < 1]

samples = []
batch_size = 500
for i in tqdm(range(len(x)//batch_size+1)):
    if (i+1)*batch_size > len(x):
        x_slice = torch.tensor(x[i*batch_size:], device=device, dtype=dtype)
    else:
        x_slice = torch.tensor(x[i*batch_size: (i+1)*batch_size], dtype=dtype, device=device)
    samples.append(sampler(x_slice).cpu().detach().numpy())

samples = np.concatenate(samples)
plt.scatter(x[:, 0], x[:, 1], c=samples)
plt.show()

#%%
y = disk_to_hyperboloid(x)
indices = np.nonzero(y[:, 2] < 3)[0]

y = np.squeeze(y[indices])
y_boundary = set(np.nonzero(y[:, 2] > 2.99)[0].tolist())
samples = np.squeeze(samples[indices])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=samples)
#ax.scatter(y_boundary[:, 0], y_boundary[:, 1], y_boundary[:, 2], c='red')
plt.show()

save_points_with_color(y, samples, y_boundary, "hyperboloid_sample")
save_points(y, samples, y_boundary, "hyperboloid_sample")

save_points(x, samples, filename="poincare_disk")

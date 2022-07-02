#%%

import torch
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from src.spaces.hyperbolic import HyperbolicSpace
from src.spectral_kernel import RandomSpectralKernel, RandomFourierFeatureKernel
from src.prior_approximation import RandomFourierApproximation
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
import matplotlib.pyplot as plt
import gpytorch
import os
import sys
sys.setrecursionlimit(2000)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.autograd.set_detect_anomaly(True)
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):  # pylint: disable=arguments-differ
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#%%

def train(model, train_x, train_y):
    training_iter = 10
    # Find optimal model hyperparameters
    model.train()
    model.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    scheduler = StepLR(optimizer, step_size=300, gamma=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        #print(scheduler.get_lr())
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 30 == 0:
            try:
                lengthscale = model.covar_module.base_kernel.lengthscale.item()
                variance = model.covar_module.outputscale
            except:
                lengthscale = model.covar_module.kernel.measure.lengthscale.item()
                variance = model.covar_module.kernel.measure.variance.item()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f variance: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                lengthscale,
                variance,
                model.likelihood.noise.item()
            ))

#%%

order = 10**4
space = HyperbolicSpace(n=2, order=order)

#%%

def plot_disc(x, y):
    fig, ax = plt.subplots()
    plt.scatter(x.detach().cpu()[:, 0], x.detach().cpu()[:, 1], c=y.detach().cpu())
    circle = plt.Circle((0, 0), 1, facecolor="none", edgecolor="black")
    ax.add_patch(circle)
    plt.show()

#%%

def f(x):
    x_norm = x/torch.norm(x, dim=1, keepdim=True)
    angle_ = torch.arccos(x_norm[:, 0])
    #return torch.sin(4*space._dist_to_id(x)+4*angle_)
    return 1/(1 + torch.square(space._dist_to_id(x)))


lspace = torch.linspace(-1, 1, 50, device=device, dtype=dtype)
test_x = torch.cartesian_prod(lspace, lspace)
test_x = test_x[torch.norm(test_x, dim=1) < 0.99]

train_x = space.rand(200)
train_y, test_y = f(train_x), f(test_x)
print("test variance: ", torch.var(test_y))

#%%

def hyperboloid_to_disk(x_hyp):
    x = x_hyp[:, 1:]/((1+x_hyp[:,0])[:, None])
    return x

lspace = torch.linspace(-1, 1, 100, device='cpu', dtype=dtype)
x = torch.cartesian_prod(lspace, lspace)
x = x[torch.norm(x, dim=1) < 1]
plot_disc(x, f(x))

#%%

lengthscale, nu = 0.1, 5.0 + space.dim
measure = SqExpSpectralMeasure(space.dim, lengthscale)
#self.measure = MaternSpectralMeasure(self.space.dim, self.lengthscale, self.nu)

#%%

likelihood = gpytorch.likelihoods.GaussianLikelihood()
euclidean_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
euclidean_model = ExactGPModel(train_x, train_y, likelihood, euclidean_kernel).to(device=device)
train(euclidean_model, train_x, train_y)

euclidean_model.eval()
with torch.no_grad(), gpytorch.settings.skip_posterior_variances(state=True):
    euclidean_f = euclidean_model(test_x)
euclidean_pred_y = euclidean_f.mean
plot_disc(test_x, euclidean_pred_y)
error = MSELoss()(euclidean_pred_y, test_y)
print("euclidean error:", error.detach().cpu())

likelihood = gpytorch.likelihoods.GaussianLikelihood()
geometric_spectral_kernel = RandomSpectralKernel(measure, space)
geometric_rff_kernel = RandomFourierFeatureKernel(geometric_spectral_kernel)
geometric_sampler = RandomFourierApproximation(geometric_spectral_kernel)
geometric_model = ExactGPModel(train_x, train_y, likelihood, geometric_rff_kernel).to(device=device)
train(geometric_model, train_x, train_y)

#%%

geometric_model.eval()
with torch.no_grad(), gpytorch.settings.skip_posterior_variances(state=True):
    euclidean_f = geometric_model(test_x)
geometric_f = geometric_model(test_x)
geometric_pred_y = geometric_f.mean
plot_disc(test_x, geometric_pred_y)
error = MSELoss()(geometric_pred_y, test_y)
print("geometric error:", error.detach().cpu())

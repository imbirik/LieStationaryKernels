import gpytorch
import torch
from torch.optim.lr_scheduler import StepLR
from src.spaces import Grassmannian, OrientedGrassmannian, HyperbolicSpace, SO, \
    SymmetricPositiveDefiniteMatrices, Sphere, Stiefel, SU
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gpytorch.settings.cholesky_jitter(double=1e-4)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, manifold, point_shape=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.mean_module = gpytorch.means.ZeroMean()
        #self.mean = torch.mean(train_y)

        self.covar_module = kernel
        self.n = manifold.n
        self.point_shape = point_shape

    def forward(self, x):
        #mean_x = self.mean_module(x) + self.mean
        mean_x = self.mean_module(x)
        if self.point_shape is not None:
            data = x.view(*x.shape[:-1], *self.point_shape)
        else:
            if torch.is_complex(x):
                data = torch.view_as_real(x).reshape(x.shape[0], -1)
            else:
                data = x
        covar_x = self.covar_module(data) + 1e-6*torch.eye(x.shape[0], device=device, dtype=dtype)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#%%

def train(model: ExactGPModel, train_x, train_y, training_iter=900, lr_scheduler_step=300, lr=0.1):
    training_iter = training_iter
    # Find optimal model hyperparameters
    model.train()
    model.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters
    scheduler = StepLR(optimizer, step_size=lr_scheduler_step, gamma=0.1)
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
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        if i % 100 == 99:
            try:
                lengthscale = model.covar_module.base_kernel.lengthscale.item()
                variance = model.covar_module.outputscale
            except:
                lengthscale = model.covar_module.measure.lengthscale.item()
                variance = model.covar_module.measure.variance.item()
            # mean = model.mean
            mean = model.mean_module.constant
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f variance: %.3f   noise: %.3f, mean: %3f' % (
                i + 1, training_iter, loss.item(),
                lengthscale,
                variance,
                model.likelihood.noise.item(),
                mean.item()
            ))
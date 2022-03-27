import torch
import gpytorch

from functorch import vmap
from src.spectral_measure import AbstractSpectralMeasure
from src.space import AbstractSpace
from src.utils import cartesian_prod



class AbstractSpectralKernel(torch.nn.Module):
    def __init__(self, measure: AbstractSpectralMeasure, space: AbstractSpace):
        super(AbstractSpectralKernel, self).__init__()
        self.measure = measure
        self.space = space

    def forward(self):
        pass


class EigenFunctionKernel(AbstractSpectralKernel):
    def __init__(self, measure, space):
        super(EigenFunctionKernel, self).__init__(measure, space)

    def forward(self, x, y):
        x1, y1 = cartesian_prod(x, y)
        cov = torch.zeros(len(x), len(y))
        for lmd, f in zip(self.space.eigenvalues, self.space.eigenfunctions):
            cov += self.measure(lmd) * vmap(vmap(f))(x1, y1)
        return cov


class EigenSpaceKernel(AbstractSpectralKernel):
    def __init__(self, measure, space):
        super(EigenSpaceKernel, self).__init__(measure, space)

    def forward(self, x, y):
        cov = torch.zeros(len(x), len(y))
        for lmd, f in zip(self.space.eigenvalues, self.space.eigenspaces):
            f_x, f_y = f(x).T, f(y).T
            normalizer = f(x).shape[1]
            cov += self.measure(lmd) * (f_x @ f_y.T)/normalizer

        return cov

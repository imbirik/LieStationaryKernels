import torch
import gpytorch

from src.spectral_measure import AbstractSpectralMeasure
from src.space import AbstractSpace
from src.utils import cartesian_prod

dtype = torch.complex128


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

        point = space.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y, normalize=True):
        x1, y1 = cartesian_prod(x, y)
        cov = torch.zeros(len(x), len(y), dtype=dtype)
        for lmd, f in zip(self.space.eigenvalues, self.space.eigenfunctions):
            cov += self.measure(lmd) * f(x1, y1)
        if normalize:
            return cov.real/self.normalizer
        else:
            return cov.real


class EigenSpaceKernel(AbstractSpectralKernel):
    def __init__(self, measure, space):
        super(EigenSpaceKernel, self).__init__(measure, space)

        point = space.rand()
        self.normalizer = self.forward(point, point, normalize=False)[0, 0]

    def forward(self, x, y, normalize=True):
        cov = torch.zeros(len(x), len(y))
        for lmd, f in zip(self.space.eigenvalues, self.space.eigenspaces):
            f_x, f_y = f(x).T, f(y).T
            cov += self.measure(lmd) * (f_x @ f_y.T)
        if normalize:
            return cov/self.normalizer
        else:
            return cov

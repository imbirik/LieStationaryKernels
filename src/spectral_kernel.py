import torch
from src.spectral_measure import AbstractSpectralMeasure
from src.space import AbstractSpace


class AbstractSpectralKernel(torch.Module):
    def __init__(self, measure: AbstractSpectralMeasure, space: AbstractSpace):
        super(AbstractSpectralKernel, self).__init__()
        self.measure = measure
        self.space = space

    def forward(self):
        pass


class EigenFunctionKernel(AbstractSpectralKernel):
    def __init__(self, *args):
        super(EigenFunctionKernel, self).__init__(*args)

    def forward(self, x, y):
        cov = 0
        for lmd, f in zip(self.space.eigenfunctions, self.space.eigenvalues):
            cov += self.measure(lmd) * f(x, y)
        return cov


class EigenSpaceKernel(AbstractSpectralKernel):
    pass
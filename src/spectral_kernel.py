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
    pass

class EigenSpaceKernel(AbstractSpectralKernel):
    pass
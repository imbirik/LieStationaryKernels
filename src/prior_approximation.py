import torch
from src.spectral_kernel import AbstractSpectralKernel


class AbstractSpectralApproximation(torch.Module):
    def __init__(self, kernel: AbstractSpectralKernel):
        super(AbstractSpectralApproximation, self).__init__()
        self.weights = None
        self.phase = None

    def sample_weights(self):
        pass

    def sample_phase(self):
        pass

    def resample(self):
        pass

    def forward(self, x, y):
        pass
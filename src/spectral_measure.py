import torch
#import functorch


class AbstractSpectralMeasure(torch.nn.Module):
    def __init__(self, dim):
        super(AbstractSpectralMeasure, self).__init__()
        self.dim = dim

    def forward(self, eigenvalues):
        pass


class AbstractSpectralMeasure(torch.nn.Module):
    def __init__(self, dim):
        super(AbstractSpectralMeasure, self).__init__()
        self.dim = dim


class MaternSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale, nu):
        super(MaternSpectralMeasure, self).__init__(dim)
        self.lengthscale = lengthscale
        self.nu = nu

    def forward(self, eigenvalues):
        return torch.pow(self.nu/(self.lengthscale ** 2) + eigenvalues, -self.nu - self.dim/4)


class SqExpSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale, nu):
        super(MaternSpectralMeasure, self).__init__(dim)
        self.lengthscale = lengthscale

    def forward(self, eigenvalues):
        return torch.exp((-self.lengthscale ** 2)/2 * eigenvalues)


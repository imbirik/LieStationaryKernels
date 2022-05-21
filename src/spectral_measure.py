import torch
#import functorch
from torch.nn import Parameter


class AbstractSpectralMeasure(torch.nn.Module):
    def __init__(self, dim):
        super(AbstractSpectralMeasure, self).__init__()
        self.dim = dim

    def forward(self, eigenvalues):
        pass


class MaternSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale, nu):
        super(MaternSpectralMeasure, self).__init__(dim)
        self.lengthscale = Parameter(torch.tensor([lengthscale]))
        self.nu = Parameter(torch.tensor([nu]))

    def forward(self, eigenvalue):
        return torch.pow(self.nu[0]/(self.lengthscale[0] ** 2) + eigenvalue, -self.nu[0] - self.dim/4)


class SqExpSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale):
        super(SqExpSpectralMeasure, self).__init__(dim)
        self.lengthscale = Parameter(torch.tensor([lengthscale]))

    def forward(self, eigenvalues):
        return torch.exp((-self.lengthscale[0] ** 2)/2 * eigenvalues)

import torch
#import functorch
from torch.nn import Parameter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class AbstractSpectralMeasure(torch.nn.Module):
    def __init__(self, dim):
        super(AbstractSpectralMeasure, self).__init__()
        self.dim = dim

    def forward(self, eigenvalues):
        pass


class MaternSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale, nu):
        super(MaternSpectralMeasure, self).__init__(dim)
        self.lengthscale = torch.tensor([lengthscale], device=device)
        self.nu = torch.tensor([nu], device=device)

    def forward(self, eigenvalue):
        return torch.pow(self.nu[0]/(self.lengthscale[0] ** 2) + eigenvalue, -self.nu[0] - self.dim/4)


class SqExpSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale):
        super(SqExpSpectralMeasure, self).__init__(dim)
        self.lengthscale = torch.tensor([lengthscale], device=device)

    def forward(self, eigenvalues):
        return torch.exp((-self.lengthscale[0] ** 2)/2 * eigenvalues)

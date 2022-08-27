import torch
from torch.nn import Parameter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64


class AbstractSpectralMeasure(torch.nn.Module):
    def __init__(self, dim):
        super(AbstractSpectralMeasure, self).__init__()
        self.dim = dim

    def forward(self, eigenvalues):
        pass


class MaternSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale, nu, variance=1.0):
        super(MaternSpectralMeasure, self).__init__(dim)
        self.lengthscale = Parameter(torch.tensor([lengthscale], device=device, dtype=dtype, requires_grad=True))
        self.variance = Parameter(torch.tensor([variance], device=device, dtype=dtype, requires_grad=True))
        self.nu = torch.tensor([nu], device=device, dtype=dtype)

    def forward(self, eigenvalue):
        return torch.pow(self.nu/(self.lengthscale.clone() * self.lengthscale.clone() / 2.0) + eigenvalue,
                         -self.nu - self.dim/2)


class SqExpSpectralMeasure(AbstractSpectralMeasure):
    def __init__(self, dim, lengthscale, variance=1.0):
        super(SqExpSpectralMeasure, self).__init__(dim)
        self.lengthscale = Parameter(torch.tensor([lengthscale], device=device, dtype=dtype, requires_grad=True))
        self.variance = Parameter(torch.tensor([variance], device=device, dtype=dtype, requires_grad=True))

    def forward(self, eigenvalues):
        return torch.exp(-self.lengthscale.clone() * self.lengthscale.clone()/2.0 * eigenvalues)

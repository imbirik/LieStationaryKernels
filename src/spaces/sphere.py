import torch
from src.space import HomogeneousSpace

class Sphere(HomogeneousSpace):
    def __init__(self, n, order):
        super(Sphere, self).__init__()
        self.dim = n
        self.order = order
        self.characters = [GegenbauerPolynomials(self.n, (self.dim - 1) / 2) for n in range(self.order)]

    def dist(self, x, y):
        return torch.arccos(torch.dot(x, y))

    def eigenfunctions(self):
        pass
        #spherical harmonics


class GegenbauerPolynomials(torch.nn.Module):
    def __init__(self, alpha, n):
        super(GegenbauerPolynomials, self).__init__()
        self.alpha = alpha
        self.n = n
        self.compute_coefficients()
        self.powers = torch.arange(0., self.n + 1.)

    def compute_coefficients(self):
        self.coefficients = torch.zeros(self.n+1)
        #Two first polynomials is quite pretty
        # C_0 = 1, C_1 = 2\alpha*x
        if self.n == 0:
            self.coefficients[0][0] = 0
        if self.n == 1:
            self.coefficients[1][1] = 2*self.alpha
        if self.n >= 2:
        # Other polynimials are given in Abramowitz & Stegun
        # c_{n-2k} = (-1)^k * 2^{n-2k} \Gamma(n-k+\alpha)/(\Gamma(\alpha)*k!(n-2k)!)
            for k in range(0, self.n//2+1):
                sgn = (-1**k)
                log_coeff = (self.n - 2 * k)*torch.log(2) + torch.lgamma(self.n - k + self.alpha) \
                            - torch.lgamma(self.alpha) - torch.lgamma(k + 1) - torch.lgamma(self.n - 2 * k + 1)
                coeff = sgn*torch.exp(log_coeff)
                self.coefficients[self.n-2*k] = coeff
    
    def forward(self, x):
        # returns \sum c_i * x^i
        x_pows = torch.pow(x, self.powers)
        return torch.dot(x_pows, self.coefficients)



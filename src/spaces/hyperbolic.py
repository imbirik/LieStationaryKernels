import torch
from src.space import NonCompactSymmetricSpace
from src.spectral_measure import MaternSpectralMeasure, SqExpSpectralMeasure
from src.utils import cartesian_prod
from math import sqrt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

dtype = torch.float64
j = torch.tensor([1j], device=device).item()  # imaginary unit
pi = 2*torch.acos(torch.zeros(1)).item()


class HyperbolicSpace(NonCompactSymmetricSpace):
    """Ball model for Hyperbolic space formulas are taken from
     https://www.ams.org/journals/proc/1994-121-02/S0002-9939-1994-1186137-8/S0002-9939-1994-1186137-8.pdf"""

    def __init__(self, n: int, order: int):
        super(HyperbolicSpace, self).__init__()
        self.n = n
        self.dim = n
        self.order = order
        self.id = torch.zeros(self.n, device=device, dtype=dtype).view(1, self.n)

        #self.lb_eigenspaces = None

    def generate_lb_eigenspaces(self, measure):
        shift = self.rand_phase(self.order)
        if isinstance(measure, MaternSpectralMeasure):
            nu, lengthscale = measure.nu[0], measure.lengthscale[0]
            scale = 1.0/torch.sqrt(nu/(nu-1)) * torch.sqrt(1/4 + 2 * nu / lengthscale)
            student_samples = torch.distributions.StudentT(df=measure.nu-1, scale=scale).rsample((self.order,))
            lmd = torch.abs(student_samples)
        elif isinstance(measure, SqExpSpectralMeasure):
            scale = 1.0/measure.lengthscale[0]
            normal_samples = torch.distributions.Normal(0, torch.abs(scale)).rsample((self.order,))
            lmd = torch.abs(normal_samples)
        else:
            return NotImplementedError
        lmd = torch.squeeze(lmd)
        self.lb_eigenspaces = HypShiftedNormailizedExp(lmd, shift, self)

    def to_group(self, x):
        return x

    def pairwise_diff(self, x, y):
        """for x of size n and y of size m computes dist(x_i-y_j) and represent as array [n*m,...]"""
        """dist(x,y) = arccosh(1+2|x-y|^2/(1-|x|^2)(1-|y|^2)"""
        x_, y_ = cartesian_prod(x, y) # [n,m,d] and [n,m,d]

        x_flatten = torch.reshape(x_, (-1, self.n))
        y_flatten = torch.reshape(y_, (-1, self.n))
        xy_l2 = torch.sum(torch.square(x_flatten-y_flatten), dim=1)
        x_l2, y_l2 = torch.sum((torch.square(x_flatten)), dim=1), torch.sum((torch.square(y_flatten)), dim=1)
        xy_dist = torch.arccosh(1 + 2 * xy_l2/(1-x_l2)/(1-y_l2))
        ones = torch.ones((xy_dist.size()[0], self.n), device=device, dtype=dtype)/sqrt(self.n)
        xy_diff = ones * torch.tanh(xy_dist/2)[:, None].clone()
        return xy_diff

    def rand_phase(self, n=1):
        """Random point on hypersphere S^{n-1}"""
        if n == 0:
            return None
        x = torch.randn(n, self.n, device=device, dtype=dtype)
        x = x / torch.norm(x, dim=1, keepdim=True)
        return x

    def rand(self, n=1):
        """Note, there is no standard method to sample from Hyperbolic space since Haar measure is infinite.
           We will sample from unit ball uniformly. """
        sphere = self.rand_phase(n)
        r = torch.pow(torch.rand(n, device=device, dtype=dtype), 1/self.n) #torch.pow((torch.rand(n, device=device, dtype=dtype)), self.n)
        return sphere * r[:, None].clone()

    def inv(self, x):
        # TODO: CHECK
        return -x

    def _dist_to_id(self, x):
        """d(0,x) = log[(1+|x|)/(1-|x|)]"""
        eucl_dist = torch.sqrt(torch.sum(torch.square(x), dim=1))
        exp_dist = (1+eucl_dist)/(1-eucl_dist)
        return torch.log(exp_dist)

class HypShiftExp(torch.nn.Module):
    """We use explicit formula for exponents in terms of hyperbolic space,
        therefore we don't need an abstract description of them."""
    def __init__(self, lmd, shift, manifold):
        super().__init__()
        self.lmd = lmd
        self.shift = shift
        self.manifold = manifold
        self.rho = torch.tensor([(self.manifold.n-1)/2], device=device, dtype=dtype)[0]

    def forward(self, x):
        """e^{(-i\lambda+(n-1)/2)<x,b>} = ((1-|x|^2)/|x-b|^2)^{-i\lambda+(n-1)/2}"""
        """x --- [n,dim], shift --- [m, dim], lmd --- [m]"""
        x_, shift_ = cartesian_prod(x, self.shift)  # both [n, m, dim]
        x_flatten = x_.reshape((-1, self.manifold.n))
        shift_flatten = shift_.reshape((-1, self.manifold.n))
        x_shift_norm = torch.sum(torch.square(x_flatten - shift_flatten), dim=1)  # [n*m]
        denominator = torch.log(x_shift_norm).reshape(x.size()[0], -1)  # log(|x_i-b_j|^2) --- [n,m]
        numerator = torch.log(1-torch.sum(torch.square(x), dim=1))  # [n]
        log_xb = numerator[:, None].clone() - denominator  # [n,m]
        inner_prod = torch.einsum('nm,m-> nm', log_xb, -j * self.lmd + self.rho)  #[n,m]
        return torch.exp(inner_prod)


class HypShiftedNormailizedExp(torch.nn.Module):
    def __init__(self, lmd, shift, manifold):
        super().__init__()
        self.n = manifold.n
        if self.n % 2 == 0:
            self.adds = torch.tensor([(2 * i + 1) ** 2 / 4 for i in range(self.n // 2 - 1)],
                                     dtype=dtype, device=device)
        else:
            self.adds = torch.tensor([i ** 2 for i in range(self.n // 2)],
                                     dtype=dtype, device=device)
        self.exp = HypShiftExp(lmd, shift, manifold)
        self.coeff = self.c_function(lmd)  # (m,)

    def c_function(self, lmd):
        lmd_sq = torch.square(lmd)  # (m, )
        log_c = torch.log(lmd_sq[:, None].clone() + self.adds[None, :].clone())
        log_c = torch.sum(log_c, dim=1)
        if self.n % 2 == 0:
            log_c = log_c + torch.log(lmd) + torch.log(torch.tanh(pi * lmd))
        return torch.squeeze(torch.exp(log_c/2))

    def forward(self, x):
        # x has shape (n, dim, dim)

        exp = self.exp(x)  # (n, m)
        return torch.einsum('nm,m->nm', exp, self.coeff)  # (n, m)


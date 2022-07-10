import torch

# from pyro.distributions.torch_distribution import TorchDistribution
# from torch.distributions.utils import broadcast_all
# from torch.distributions import constraints
# from pyro.distributions.rejector import Rejector
dtype = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


def GOE_sampler(num_samples, n):
    samples = torch.randn(num_samples, n, n, device=device, dtype=dtype)
    samples = (samples + torch.transpose(samples, -2, -1))/2
    eigenvalues = torch.linalg.eigvalsh(samples, UPLO='U')
    return eigenvalues


def triu_ind(m, n, offset):
    a = torch.ones(m, n, n)
    triu_indices = a.triu(diagonal=offset).nonzero().transpose(0, 1)
    return triu_indices[0], triu_indices[1], triu_indices[2]


def vander_det(x):
    """ computing of \prod_{i < j} x_i - x_j"""
    dim = x.size()[1]
    x_ = (x[:, :, None] - x[:, None, :])[triu_ind(x.size()[0], dim, 1)].reshape(-1, dim * (dim - 1) // 2)
    res = torch.prod(x_, dim=1)
    return res


def vander_det2(x):
    d = x.unsqueeze(-2) - x.unsqueeze(-1)
    return (torch.ones_like(d).triu() + d.tril(-1)).prod(-1).prod(-1)


def poly_eval_tensor(x, coeffs):
    acc = torch.zeros_like(x)
    for i in range(len(coeffs) - 1):
        acc = (acc + coeffs[i]) * x
    return acc + coeffs[-1]


def cartesian_prod(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    '''
    Cartesian product of two tensors
    :param x1: tensor of the shape [a,x1,x2,...]
    :param x2: tensor of the shape [b,x1,x2,...]
    :return: 2 tensors of the shape [a,b,x1,x2,...]
    '''
    x1_shape = [1, x2.shape[0]] + [1] * (len(x1.shape)-1)
    x2_shape = [x1.shape[0], 1] + [1] * (len(x2.shape)-1)
    x1_ = torch.tile(x1[:, None, ...], x1_shape)  # (N, M, dim+1)
    x2_ = torch.tile(x2[None, :, ...], x2_shape)  # (N, M, dim+1)

    return x1_, x2_


def fixed_length_partitions(n, L):
    """
    https://www.ics.uci.edu/~eppstein/PADS/IntegerPartitions.py
    Integer partitions of n into L parts, in colex order.
    The algorithm follows Knuth v4 fasc3 p38 in rough outline;
    Knuth credits it
     to Hindenburg, 1779.
    """

    # guard against special cases
    if L == 0:
        if n == 0:
            yield []
        return
    if L == 1:
        if n > 0:
            yield [n]
        return
    if n < L:
        return

    partition = [n - L + 1] + (L - 1) * [1]
    while True:
        yield partition.copy()
        if partition[0] - 1 > partition[1]:
            partition[0] -= 1
            partition[1] += 1
            continue
        j = 2
        s = partition[0] + partition[1] - 1
        while j < L and partition[j] >= partition[0] - 1:
            s += partition[j]
            j += 1
        if j >= L:
            return
        partition[j] = x = partition[j] + 1
        j -= 1
        while j > 0:
            partition[j] = x
            s -= x
            j -= 1
        partition[0] = s


def hook_content_formula(lmd, n):
    numer = 1
    denom = 1

    l_cols = [sum([row_l >= i+1 for row_l in lmd]) for i in range(lmd[0])]
    for id_row, l_row in enumerate(lmd):
        for id_col in range(l_row):
            numer  *= (n + id_col - id_row)
            denom *= l_cols[id_col] + l_row-id_row - id_col - 1

    return numer/denom


def partition_dominance_cone(partition):
    '''
    Calculates partitions dominated by a given one
    and having the same number of parts (including zero parts of the original)
    '''
    cone = {partition}
    new_partitions = {0}
    prev_partitions = cone
    while new_partitions:
        new_partitions = set()
        for partition in prev_partitions:
            for i in range(len(partition) - 1):
                if partition[i] > partition[i + 1]:
                    for j in range(i + 1, len(partition)):
                        if partition[i] > partition[j] + 1 and partition[j] < partition[j - 1]:
                            new_partition = list(partition)
                            new_partition[i] -= 1
                            new_partition[j] += 1
                            new_partition = tuple(new_partition)
                            if new_partition not in cone:
                                new_partitions.add(new_partition)
        cone.update(new_partitions)
        prev_partitions = new_partitions
    return cone


def partition_dominance_or_subpartition_cone(partition):
    '''
        Calculates subpartitions and partitions dominated by a given one
        and having the same number of parts (including zero parts of the original)
        '''
    cone = {partition}
    new_partitions = {0}
    prev_partitions = cone
    while new_partitions:
        new_partitions = set()
        for partition in prev_partitions:
            for i in range(len(partition) - 1):
                if partition[i] > partition[i + 1]:
                    new_partition = list(partition)
                    new_partition[i] -= 1
                    new_partition = tuple(new_partition)
                    if new_partition not in cone:
                        new_partitions.add(new_partition)
                    for j in range(i + 1, len(partition)):
                        if partition[i] > partition[j] + 1 and partition[j] < partition[j - 1]:
                            new_partition = list(partition)
                            new_partition[i] -= 1
                            new_partition[j] += 1
                            new_partition = tuple(new_partition)
                            if new_partition not in cone:
                                new_partitions.add(new_partition)
        cone.update(new_partitions)
        prev_partitions = new_partitions
    return cone


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


# class GOE(TorchDistribution):
#     """Samples from distribution with pdf e^{-t*\lambda^2}\prod_{i<j} |\lambda_i-\lambda_j|"""
#     arg_constraints = {'scale': constraints.positive}
#     has_rsample = True
#
#     def __init__(self, dim, scale=1.0):
#         self.dim = dim
#         self.scale = broadcast_all(scale)[0]
#         super(GOE, self).__init__(event_shape=torch.Size([dim]))
#
#     def rsample(self, sample_shape=torch.Size()):
#         shape = self._extended_shape(sample_shape=sample_shape)
#         X = torch.randn(torch.Size((*shape, self.dim)), dtype=dtype, device=device)
#         M = (X + torch.transpose(X, -2, -1)) / sqrt(2)
#         eigenvalues = torch.linalg.eigvalsh(M)
#         return eigenvalues/self.scale
#
#     def log_prob(self, value):
#         value_diff = (value[:, None, :] - value[:, :, None])[triu_ind(value.size()[0], self.dim, 1)].\
#             reshape(-1, self.dim * (self.dim - 1) // 2)
#         log_prob = torch.sum(torch.log(torch.abs(value_diff))) - self.scale * torch.square(torch.linalg.norm(value))
#         return log_prob
#
#
# class StudentGOE(TorchDistribution):
#     """Samples from distribution with pdf (nu/scale^2+|\lambda|^2 + c)^{-\nu} \prod_{i < j} |\lambda_i-\lambda_j|"""
#     arg_constraints = {'scale': constraints.positive, 'nu': constraints.positive}
#     has_rsample = True
#
#     def __init__(self, dim, scale, nu, c):
#         self.dim = dim
#         self.scale, self.nu = broadcast_all(scale, nu)
#         self.shift = 1/torch.sqrt(c + 2 * self.nu / self.scale)
#
#         super(StudentGOE, self).__init__(event_shape=torch.Size([dim]))
#
#     def rsample(self, sample_shape=torch.Size()):
#         goe_samples = GOE(self.dim, self.shift).rsample(sample_shape)
#         chi2_samples = torch.distributions.chi2.Chi2(2 * self.nu).rsample(sample_shape)
#         chi_samples = torch.sqrt(chi2_samples)
#         return goe_samples / chi_samples[:, None]
#
#     def log_prob(self, value):
#         value_diff = (value[:, None, :] - value[:, :, None])[triu_ind(value.size()[0], self.dim, 1)].\
#             reshape(-1, self.dim * (self.dim - 1) // 2)
#         log_prob = torch.sum(torch.log(torch.abs(value_diff))) - \
#                    torch.log(torch.square(torch.linalg.norm(value))+self.shift) - self.nu
#         return log_prob
#
#

# if __name__ == "__main__":
#     dim = 5
#     def c_function_tanh(lmd):
#         lmd_ = (lmd[:, None, :] - lmd[:, :, None])[triu_ind(lmd.size()[0], dim, 1)].reshape(-1,
#                                                                                        dim * (dim - 1) // 2)
#         lmd_ = torch.pi * torch.abs(lmd_)
#         lmd_ = torch.tanh(lmd_)
#         c_function_tanh = torch.sum(torch.log(lmd_), dim=1)
#
#         return c_function_tanh
#
#     sampler = Rejector(GOE(dim=5), c_function_tanh, 0)
#     print(sampler.rsample((10,)))
#


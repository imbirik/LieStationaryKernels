import torch
dtype = torch.double


def GOE_sampler(num_samples, n):
    samples = torch.randn(num_samples, n, n, dtype=dtype)
    samples = (samples + torch.transpose(samples, -2, -1))/2
    eigenvalues = torch.linalg.eigvalsh(samples, UPLO='U')
    return eigenvalues


def triu_ind(m, n, offset):
    a = torch.ones(m, n, n)
    triu_indices = a.triu(diagonal=offset).nonzero().transpose(0, 1)
    return triu_indices[0], triu_indices[1], triu_indices[2]


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


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


if __name__ == "__main__":
    x = torch.reshape(torch.arange(2 * 3 * 4), (2, 3, 4))
    y = torch.reshape(torch.arange(3 * 3 * 4), (3, 3, 4))
    x1, y1 = cartesian_prod(x, y)
    print(x1, x1.shape)
    print(y1, y1.shape)
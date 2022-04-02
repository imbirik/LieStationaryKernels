import torch


def cartesian_prod(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    '''
    Cartesian product of two tensors
    :param x1: tensor of the shape [a,x]
    :param x2: tensor of the shape [b,x]
    :return: tensor of the shape [a,b,2,x]
    '''

    x1_ = torch.tile(x1[..., None, :], (1, x2.shape[0], 1))  # (N, M, dim+1)
    x2_ = torch.tile(x2[None], (x1.shape[0], 1, 1))  # (N, M, dim+1)

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

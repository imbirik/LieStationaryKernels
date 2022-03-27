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

if __name__ == "__main__":
    x1 = torch.reshape(torch.arange(18), (6, 3))
    x2 = torch.reshape(torch.arange(12), (4, 3))
    cartesian_prod(x1, x2)
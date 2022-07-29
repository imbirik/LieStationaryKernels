import torch
from abc import ABC, abstractmethod
import heapq
from src.utils import lazy_property
from src.utils import cartesian_prod

j = torch.tensor([1j]).item()  # imaginary unit
pi = 2*torch.acos(torch.zeros(1)).item()


class AbstractManifold(ABC):
    """Abstract base class for Compact Lie Group, Compact Homogeneous space or Symmetric Space"""
    def __init__(self):
        ABC.__init__(self)

    @abstractmethod
    def rand(self, n=1):
        # returns random element with respect to some fixed measure
        raise NotImplementedError

    def pairwise_diff(self, x, y):
        raise NotImplementedError


class CompactLieGroup(AbstractManifold, ABC):
    """Lie group abstract base class"""
    def __init__(self, *, order: int):
        """
        Generate the list of signatures of representations and pick those with smallest LB eigenvalues.
        :param order: the order of approximation, the number of representations calculated
        """
        super().__init__()
        if order:
            lb_eigenspaces = [self.Eigenspace(signature, manifold=self) for signature in self.generate_signatures(order)]
            self.lb_eigenspaces = heapq.nsmallest(order, lb_eigenspaces, key=lambda eig: eig.lb_eigenvalue)
        else:
            self.lb_eigenspaces = []

    @abstractmethod
    def generate_signatures(self, order) -> list:
        """Generate signatures of representations to enumerate them."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def inv(x):
        """Calculates the group inverse of a batch of group elements"""
        raise NotImplementedError

    @abstractmethod
    def torus_representative(self, x):
        """Return the corresponding element of the standard maximal torus"""
        raise NotImplementedError

    def pairwise_diff(self, x, y, reverse=False):
        """If reverse is False for x of size n and y of size n computes x_i*y_j^{-1} and represent as array [n*m,...]
            if reverse is True then compute same but in this case for x_i^{-1}*y_j"""
        if not reverse:
            # computes xy^{-1}
            y_inv = self.inv(y)
            x_, y_ = cartesian_prod(x, y_inv)  # [n,m,d,d] and [n,m,d,d]
        else:
            # computes x^{-1}y
            x_inv = self.inv(x)
            x_, y_ = cartesian_prod(x_inv, y)  # [n,m,d,d] and [n,m,d,d]

        x_flatten = torch.reshape(x_, (-1, self.n, self.n))
        y_flatten = torch.reshape(y_, (-1, self.n, self.n))

        x_y_ = torch.bmm(x_flatten, y_flatten)  # [n*m, ...]
        return x_y_

    def pairwise_embed(self, x, y):
        """for x of size n and y of size n computes the torus representatives corresponding to all pairs x_i and y_j
         as an array [n*m,...]"""
        x_y_ = self.pairwise_diff(x, y)
        x_y_gammas = self.torus_representative(x_y_)
        return x_y_gammas


class HomogeneousSpace(AbstractManifold, ABC):
    """Homogeneous space of form M=G/H"""
    def __init__(self, g: CompactLieGroup, h, average_order):
        # H is not typed to CompactLieGroup because it is often given as an incomplete realization
        AbstractManifold.__init__(self)
        self.g, self.h = g, h
        self.dim = self.g.dim - self.h.dim
        self.order, self.average_order = self.g.order, average_order

        self.h_samples = self.sample_H(self.average_order)

        self.lb_eigenspaces = [AveragedLBEigenspace(representation, self) for
                               representation in self.g.lb_eigenspaces]

    @abstractmethod
    def H_to_G(self, h):
        """Implements inclusion H<G"""
        raise NotImplementedError

    @abstractmethod
    def M_to_G(self, x):
        """Implements lifting M->G"""
        raise NotImplementedError

    @abstractmethod
    def G_to_M(self, g):
        """Implements a canonical projection G->M"""
        raise NotImplementedError

    def sample_H(self, n):
        raw_samples = self.h.rand(n)
        return self.H_to_G(raw_samples)

    def rand(self, n=1):
        raw_samples = self.g.rand(n)
        return self.G_to_M(raw_samples)

    def pairwise_diff(self, x, y):
        """For arrays of form x_iH, y_jH computes difference Hx_i^{-1}y_jH """
        x_, y_ = self.M_to_G(x), self.M_to_G(y)
        diff = self.g.pairwise_diff(x_, y_, reverse=True)
        return diff

    def pairwise_embed(self, x, y):
        """For arrays of form x_iH, y_jH computes embedding corresponding to x_i, y_j
        i.e. flattened array of form G.embed(x_i^{-1}y_jh_k)"""
        x_y_ = self.pairwise_diff(x, y)
        return self.g.pairwise_embed(x_y_, self.h_samples)

    @abstractmethod
    def dist(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def compute_inv_dimension(self, signature):
        raise NotImplementedError


class LBEigenfunction(ABC):
    """Laplace-Beltrami eigenfunction abstract base class"""
    def __init__(self, index, *, manifold: AbstractManifold):
        """
        :param index: the index of an LB eigenspace
        :param manifold: the "parent" manifold
        """
        self.index = index
        self.manifold = manifold
        self.lb_eigenvalue = self.compute_lb_eigenvalue()

    @abstractmethod
    def compute_lb_eigenvalue(self):
        """Compute the Laplace-Beltrami eigenvalues of the eigenfunction."""
        raise NotImplementedError


class LBEigenspace(ABC):
    """Laplace-Beltrami eigenspace abstract base class"""
    def __init__(self, index, *, manifold: AbstractManifold):
        """
        :param index: the index of an LB eigenspace
        :param manifold: the "parent" manifold
        """
        self.index = index
        self.manifold = manifold
        self.dimension = self.compute_dimension()
        self.lb_eigenvalue = self.compute_lb_eigenvalue()

    @abstractmethod
    def compute_dimension(self):
        """Compute the dimension of the Laplace-Beltrami eigenspace."""
        raise NotImplementedError

    @abstractmethod
    def compute_lb_eigenvalue(self):
        """Compute the Laplace-Beltrami eigenvalues of the eigenspace."""
        raise NotImplementedError


class LBEigenspaceWithSum(LBEigenspace, ABC):
    """Laplace-Beltrami eigenspace ABC in case the sum function of an orthonormal basis paired products is available"""
    @lazy_property
    def basis_sum(self):
        basis_sum = self.compute_basis_sum()
        return basis_sum

    @abstractmethod
    def compute_basis_sum(self):
        """Compute the sum of the orthonormal basis paired products."""
        raise NotImplementedError


class LBEigenspaceWithBasis(LBEigenspaceWithSum, ABC):
    """Laplace-Beltrami eigenspace ABC in case orthonormal basis is available"""
    @lazy_property
    def basis(self):
        basis = self.compute_basis()
        return basis

    @abstractmethod
    def compute_basis(self):
        """Compute an orthonormal basis of the eigenspace."""
        raise NotImplementedError


class AveragedLBEigenspace(LBEigenspaceWithSum):
    """The Laplace-Beltrami eigenspace for homogeneous space of a compact Lie group,
    with the `basis sum` calculated via averaging the character"""
    def __init__(self, representaion: LBEigenspaceWithSum, manifold: HomogeneousSpace):
        """
        :param signature: the signature of a representation
        :param manifold: the "parent" manifold, an instance of SO
        """
        self.initial_representation = representaion
        super().__init__(representaion.index, manifold=manifold)
        self.inv_dimension = self.manifold.compute_inv_dimension(representaion.index)

    def compute_dimension(self):
        return self.initial_representation.compute_dimension()

    def compute_lb_eigenvalue(self):
        return self.initial_representation.compute_lb_eigenvalue()

    def compute_basis_sum(self):
        return AveragedLieGroupCharacter(self, self.manifold, self.initial_representation.compute_basis_sum())


class LieGroupCharacter(torch.nn.Module, ABC):
    """Lie group representation character abstract base class"""
    def __init__(self, *, representation: LBEigenspace):
        super().__init__()
        self.representation = representation

    @abstractmethod
    def chi(self, gammas):
        """Calculates the character value at an element of the maximal torus"""
        raise NotImplementedError

    def evaluate(self, x):
        """Calculates the character value at an element of the group"""
        gammas = self.representation.manifold.torus_embed(x)
        return self.chi(gammas)

    def forward(self, gammas):
        # [n, dim, dim]
        chi = self.representation.dimension * self.chi(gammas)  # [n]
        return chi


class AveragedLieGroupCharacter(torch.nn.Module):
    """Lie group character averaged over a subgroup"""
    def __init__(self, representation: AveragedLBEigenspace, space: HomogeneousSpace, chi: LieGroupCharacter):
        super().__init__()
        self.representation = representation
        self.space = space
        self.chi = chi

    def forward(self, gammas_x_h):
        chi_x_h = self.chi(gammas_x_h).reshape(-1, self.space.average_order)
        result = torch.mean(chi_x_h, dim=-1)

        #is_close_to_id = self.representation.manifold.close_to_id(x)  # [n]
        #rep_dim = self.representation.dimension * self.representation.inv_dimension
        #result = torch.where(is_close_to_id,  rep_dim * torch.ones_like(result), result)

        return result


class TranslatedCharactersBasis(torch.nn.Module):
    def __init__(self, *, representation: LBEigenspaceWithBasis):
        super().__init__()
        self.representation = representation
        dim = self.representation.dimension
        self.character = self.representation.basis_sum
        gram_is_spd = False
        attempts = 0
        while not gram_is_spd:
            attempts += 1
            self.translations = self.representation.manifold.rand(dim**2)
            ratios = self.representation.manifold.pairwise_diff(self.translations, self.translations)
            gram = self.character.forward(ratios).reshape(dim**2, dim**2)
            try:
                self.coeffs = torch.linalg.inv(torch.linalg.cholesky(gram))
                gram_is_spd = True
            except RuntimeError:
                if attempts >= 3:
                    raise

    def forward(self, x):
        x_unsq = x.unsqueeze(1)
        tr_unsq = self.translations.unsqueeze(0)
        x_translates = torch.matmul(tr_unsq, x_unsq)
        characters = self.character.forward(x_translates)
        return torch.matmul(self.coeffs, characters.T).T


class NonCompactSymmetricSpace(AbstractManifold, ABC):
    """Symmetric space of form G/H abstract class"""
    def __init__(self):
        super().__init__()
        #self.lb_eigenspaces = None  # will be generated with respect to spectral measure

    def dist(self, x, y):
        raise NotImplementedError

    def generate_lb_eigenspaces(self, measure):
        """Generates Eigenspaces with respect to the measure"""
        raise NotImplementedError

    def rand_factor(self, n=1):
        """ Generate elements from H with respect to Haar measure on H """
        raise NotImplementedError

    def inv(self, x):
        """ For element x in G calculates x^{-1}"""
        raise NotImplementedError

    def pairwise_diff(self, x, y):
        """for x of size n and y of size m computes x_i-y_j and represent as array [n*m,...]"""
        y_inv = self.inv(y)
        x_, y_ = cartesian_prod(x, y_inv)  # [n,m,d,d] and [n,m,d,d]
        x_flatten = torch.reshape(x_, (-1, self.n, self.n))
        y_flatten = torch.reshape(y_, (-1, self.n, self.n))

        x_y_ = torch.bmm(x_flatten, y_flatten)  # [n*m, ...]
        return x_y_


class NonCompactSymmetricSpaceExp(torch.nn.Module, ABC):
    """For x in G computes e^{(i*lmd+rho)a(xh^{-1})}"""
    def __init__(self, lmd, shift, manifold):
        super().__init__()
        self.lmd = lmd  # shape is (m, r)
        self.shift = shift  # shape is (m,...)
        self.manifold = manifold

        self.order = lmd.size()[0]
        self.n = self.manifold.n
        self.rho = self.compute_rho()  # shape is (r,)

    @abstractmethod
    def iwasawa_decomposition(self, x):
        """For x in G computes Iwasawa decomposition x = h(x)a(x)n(x)"""
        raise NotImplementedError

    @abstractmethod
    def compute_rho(self):
        raise NotImplementedError

    def forward(self, x):
        # shape x is (n, ...)
        n = x.shape[0]
        x_shift_flatten = self.manifold.pairwise_diff(x, self.shift) # (n * m, ...)
        _, a_flatten, _ = self.iwasawa_decomposition(x_shift_flatten)  # shape (n * m, rank)
        log_a_flatten = torch.log(a_flatten).type(torch.complex128)
        a = log_a_flatten.view(n, self.order, -1)

        lin_func = j*self.lmd + self.rho[None, :] # (m, rank)
        inner_prod = torch.einsum('nmr,mr->nm', a, lin_func)
        return torch.exp(inner_prod)  # shape (n, m)

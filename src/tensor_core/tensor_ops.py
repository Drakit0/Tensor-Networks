import numpy as np
from typing import List, Tuple, Optional
from scipy.linalg import svd


class Tensor:
    """
    Basic tensor class with support for graphical tensor network operations.
    
    Represents a tensor as a multidimensional array with labeled indices.
    Implements the graphical calculus described in Penrose notation.
    """
    
    def __init__(self, data: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """
        Initialize a tensor.
        
        Args:
            data: numpy array containing tensor components
            labels: optional list of index labels
        """
        self.data = np.array(data, dtype=complex)
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        
        if labels is None:
            self.labels = [f"i{k}" for k in range(self.ndim)]
        else:
            assert len(labels) == self.ndim, "Number of labels must match tensor order"
            self.labels = labels
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, labels={self.labels})"
    
    def contract(self, other: 'Tensor', indices: List[Tuple[int, int]]) -> 'Tensor':
        """
        Contract this tensor with another tensor.
        
        Implements the wire connection in tensor diagrams.
        
        Args:
            other: tensor to contract with
            indices: list of (self_idx, other_idx) pairs to contract
            
        Returns:
            Resulting contracted tensor
        """
        # Build axes for np.tensordot
        axes1 = [idx[0] for idx in indices]
        axes2 = [idx[1] for idx in indices]
        
        result = np.tensordot(self.data, other.data, axes=(axes1, axes2))
        
        # Determine remaining labels
        remaining_labels = []
        for i, label in enumerate(self.labels):
            if i not in axes1:
                remaining_labels.append(label)
        for i, label in enumerate(other.labels):
            if i not in axes2:
                remaining_labels.append(label)
        
        return Tensor(result, remaining_labels)
    
    def transpose(self, perm: List[int]) -> 'Tensor':
        """
        Permute tensor indices.
        
        Args:
            perm: permutation of indices
            
        Returns:
            Transposed tensor
        """
        new_data = np.transpose(self.data, perm)
        new_labels = [self.labels[i] for i in perm]
        return Tensor(new_data, new_labels)
    
    def trace(self, axis1: int, axis2: int) -> 'Tensor':
        """
        Take trace over two indices (contract with identity).
        
        Args:
            axis1: first index to trace
            axis2: second index to trace
            
        Returns:
            Traced tensor
        """
        result = np.trace(self.data, axis1=axis1, axis2=axis2)
        new_labels = [l for i, l in enumerate(self.labels) if i not in [axis1, axis2]]
        return Tensor(result, new_labels)
    
    def norm(self) -> float:
        """Compute Frobenius norm of the tensor."""
        return float(np.linalg.norm(self.data))
    
    def conjugate(self) -> 'Tensor':
        """Complex conjugate of tensor (flips diagram vertically)."""
        return Tensor(np.conjugate(self.data), self.labels.copy())


def identity(dim: int, label1: str = "i", label2: str = "j") -> Tensor:
    """
    Create identity tensor (wire in tensor diagrams).
    
    Args:
        dim: dimension of the identity
        label1: label for first index
        label2: label for second index
        
    Returns:
        Identity tensor
    """
    return Tensor(np.eye(dim, dtype=complex), [label1, label2])


def delta(dim: int, labels: List[str]) -> Tensor:
    """
    Create generalized Kronecker delta tensor.
    
    Used for index contraction (connecting wires).
    
    Args:
        dim: dimension
        labels: list of index labels
        
    Returns:
        Delta tensor
    """
    order = len(labels)
    shape = [dim] * order
    data = np.zeros(shape, dtype=complex)
    
    # Set delta_{i,i,...,i} = 1
    for i in range(dim):
        idx = tuple([i] * order)
        data[idx] = 1.0
    
    return Tensor(data, labels)


def cup(dim: int, label1: str = "i", label2: str = "j") -> Tensor:
    """
    Create cup tensor for raising indices.
    
    Corresponds to Bell state: |cup⟩ = Σ_k |kk⟩
    
    Args:
        dim: dimension
        label1: first index label
        label2: second index label
        
    Returns:
        Cup tensor
    """
    data = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        data[i, i] = 1.0
    return Tensor(data, [label1, label2])


def cap(dim: int, label1: str = "i", label2: str = "j") -> Tensor:
    """
    Create cap tensor for lowering indices.
    
    Corresponds to ⟨cap| = Σ_k ⟨kk|
    
    Args:
        dim: dimension
        label1: first index label
        label2: second index label
        
    Returns:
        Cap tensor
    """
    return cup(dim, label1, label2)  # Same as cup for finite dimensions


def swap(dim: int) -> Tensor:
    """
    Create SWAP gate tensor.
    
    Swaps two subsystems: SWAP|ij⟩ = |ji⟩
    
    Args:
        dim: dimension of each subsystem
        
    Returns:
        SWAP tensor
    """
    data = np.zeros((dim, dim, dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            data[j, i, i, j] = 1.0
    return Tensor(data, ["i", "j", "k", "l"])


def epsilon(dim: int = 2) -> Tensor:
    """
    Create fully antisymmetric epsilon tensor.
    
    For dim=2: epsilon_01 = -epsilon_10 = 1, others = 0
    For dim=3: Levi-Civita symbol
    
    Args:
        dim: dimension (2 or 3 supported)
        
    Returns:
        Epsilon tensor
    """
    if dim == 2:
        data = np.array([[0, 1], [-1, 0]], dtype=complex)
        return Tensor(data, ["i", "j"])
    elif dim == 3:
        data = np.zeros((3, 3, 3), dtype=complex)
        # Define Levi-Civita symbol
        data[0, 1, 2] = data[1, 2, 0] = data[2, 0, 1] = 1.0
        data[0, 2, 1] = data[2, 1, 0] = data[1, 0, 2] = -1.0
        return Tensor(data, ["i", "j", "k"])
    else:
        raise ValueError(f"Epsilon tensor not implemented for dim={dim}")


def tensor_svd(tensor: Tensor, left_indices: List[int]) -> Tuple[Tensor, np.ndarray, Tensor]:
    """
    Perform singular value decomposition on a tensor.
    
    Implements diagrammatic SVD from the paper (Section 4).
    Splits tensor into U, Sigma, V^dagger where Sigma contains singular values.
    
    Args:
        tensor: input tensor
        left_indices: indices to be grouped for left matrix
        
    Returns:
        Tuple of (U tensor, singular values array, V tensor)
    """
    # Reshape tensor into matrix
    right_indices = [i for i in range(tensor.ndim) if i not in left_indices]
    
    left_dims = [tensor.shape[i] for i in left_indices]
    right_dims = [tensor.shape[i] for i in right_indices]
    
    left_size = np.prod(left_dims) if left_dims else 1
    right_size = np.prod(right_dims) if right_dims else 1
    
    # Permute and reshape
    perm = left_indices + right_indices
    tensor_perm = tensor.transpose(perm)
    matrix = tensor_perm.data.reshape(left_size, right_size)
    
    # Perform SVD
    U, s, Vh = svd(matrix, full_matrices=False)
    
    # Reshape back to tensors
    chi = len(s)  # bond dimension
    U_shape = left_dims + [chi]
    V_shape = [chi] + right_dims
    
    U_tensor = Tensor(U.reshape(U_shape), 
                     [tensor.labels[i] for i in left_indices] + ["bond"])
    V_tensor = Tensor(Vh.reshape(V_shape),
                     ["bond"] + [tensor.labels[i] for i in right_indices])
    
    return U_tensor, s, V_tensor


def truncate_svd(U: Tensor, s: np.ndarray, V: Tensor, 
                 chi_max: Optional[int] = None,
                 epsilon: Optional[float] = None) -> Tuple[Tensor, np.ndarray, Tensor]:
    """
    Truncate SVD by keeping only the largest singular values.
    
    Implements the MPS approximation discussed in Example (Section 5).
    
    Args:
        U: left unitary tensor
        s: singular values (sorted descending)
        V: right unitary tensor
        chi_max: maximum bond dimension to keep
        epsilon: cutoff threshold for singular values
        
    Returns:
        Truncated (U, s, V)
    """
    chi: int = len(s)
    
    # Determine truncation
    if chi_max is not None:
        chi = min(chi, chi_max)
    
    if epsilon is not None:
        # Keep singular values >= epsilon
        chi = min(chi, int(np.sum(s >= epsilon)))
    
    if chi == 0:
        chi = 1  # Keep at least one singular value
    
    # Truncate
    s_trunc = s[:chi]
    
    U_data = U.data[..., :chi]
    U_trunc = Tensor(U_data, U.labels[:-1] + ["bond"])
    
    V_data = V.data[:chi, ...]
    V_trunc = Tensor(V_data, ["bond"] + V.labels[1:])
    
    return U_trunc, s_trunc, V_trunc

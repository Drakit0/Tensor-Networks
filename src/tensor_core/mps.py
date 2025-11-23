import numpy as np
from typing import List, Optional
from .tensor_ops import Tensor
from typing import Union

class MPS:
    """
    Matrix Product State representation.
    
    Represents a quantum state as a chain of tensors:
    |ψ⟩ = Σ A[1]_i A[2]_j ... A[n]_k |ijk...⟩
    
    This provides an efficient representation for states with limited entanglement.
    """
    
    def __init__(self, tensors: List[Tensor], bond_dims: Optional[List[int]] = None) -> None:
        """
        Initialize MPS.
        
        Args:
            tensors: list of MPS tensors (each has physical + bond indices)
            bond_dims: optional list of bond dimensions
        """
        self.tensors = tensors
        self.n_sites = len(tensors)
        
        if bond_dims is None:
            # Infer bond dimensions from tensor shapes
            self.bond_dims = []
            for t in tensors[:-1]:
                self.bond_dims.append(t.shape[-1])
        else:
            self.bond_dims = bond_dims
    
    def __repr__(self) -> str:
        return f"MPS(n_sites={self.n_sites}, bond_dims={self.bond_dims})"
    
    @classmethod
    def from_state_vector(cls, state: np.ndarray, 
                         d: int = 2,
                         chi_max: Optional[int] = None,
                         epsilon: Optional[float] = None) -> 'MPS':
        """
        Convert state vector to MPS using iterative SVD.
        
        Implements the MPS factorization from Section 5 of the paper.
        
        Args:
            state: state vector to decompose
            d: physical dimension at each site
            chi_max: maximum bond dimension
            epsilon: SVD truncation threshold
            
        Returns:
            MPS representation
        """
        n = int(np.log(len(state)) / np.log(d))
        assert d**n == len(state), "State dimension must be d^n"
        
        # Reshape state into tensor
        shape = [d] * n
        psi = state.reshape(shape)
        
        tensors = []
        bond_dims: List[int] = []
        
        # Iterative SVD from left to right
        for i in range(n - 1):
            # Reshape remaining dimensions
            current_shape = psi.shape
            left_size = current_shape[0]
            right_size = np.prod(current_shape[1:])
            
            matrix = psi.reshape(left_size, right_size)
            
            # SVD
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
            
            # Truncate if needed
            chi = len(s)
            if chi_max is not None:
                chi = min(chi, chi_max)
            if epsilon is not None:
                chi = min(chi, np.sum(s >= epsilon))
            chi = max(chi, 1)
            
            U = U[:, :chi]
            s = s[:chi]
            Vh = Vh[:chi, :]
            
            # Store left unitary as MPS tensor
            if i == 0:
                # First tensor: only physical and right bond
                tensors.append(Tensor(U, ["phys", "bond_r"]))
            else:
                # Middle tensors: left bond, physical, right bond
                U_reshaped = U.reshape(bond_dims[-1], d, chi)
                tensors.append(Tensor(U_reshaped, ["bond_l", "phys", "bond_r"]))
            
            bond_dims.append(chi)
            
            # Continue with S*Vh
            psi = (np.diag(s) @ Vh).reshape([chi] + list(current_shape[1:]))
        
        # Last tensor
        if n > 1:
            psi_reshaped = psi.reshape(bond_dims[-1], d)
            tensors.append(Tensor(psi_reshaped, ["bond_l", "phys"]))
        else:
            tensors.append(Tensor(psi.reshape(d, 1), ["phys", "bond_r"]))
        
        return cls(tensors, bond_dims)
    
    def to_state_vector(self) -> np.ndarray:
        """
        Convert MPS back to full state vector.
        
        Returns:
            State vector
        """
        # Contract all tensors
        result = self.tensors[0].data
        
        for i in range(1, self.n_sites):
            tensor = self.tensors[i].data
            result = np.tensordot(result, tensor, axes=([-1], [0]))
        
        # Flatten to vector
        return result.flatten()
    
    def norm(self) -> float:
        """Compute norm of the MPS."""
        psi = self.to_state_vector()
        return float(np.linalg.norm(psi))
    
    def normalize(self) -> None:
        """Normalize the MPS in place."""
        norm = self.norm()
        if norm > 0:
            # Normalize first tensor
            self.tensors[0].data /= norm
    
    def bond_dimension(self, site: int) -> int:
        """Get bond dimension at given site."""
        if site < len(self.bond_dims):
            return self.bond_dims[site]
        return 1
    
    def entanglement_entropy(self, site: int) -> float:
        """
        Compute von Neumann entanglement entropy across bond.
        
        Implements Definition (entropies) from Section 5.
        
        Args:
            site: bond position (between site and site+1)
            
        Returns:
            Von Neumann entropy
        """
        # Contract left part
        left = self.tensors[0].data
        for i in range(1, site + 1):
            left = np.tensordot(left, self.tensors[i].data, axes=([-1], [0]))
        
        # Contract right part
        right = self.tensors[site + 1].data
        for i in range(site + 2, self.n_sites):
            right = np.tensordot(right, self.tensors[i].data, axes=([-1], [0]))
        
        # Compute reduced density matrix
        left_flat = left.reshape(-1, left.shape[-1])
        right_flat = right.reshape(right.shape[0], -1)
        
        # Form state
        psi = left_flat @ right_flat
        rho = psi @ psi.conj().T
        
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(rho)
        eigvals_filtered = eigvals[eigvals > 1e-12]  # Remove numerical zeros
        
        # Von Neumann entropy
        return float(-np.sum(eigvals_filtered * np.log(eigvals_filtered)))
    
    def expect_operator(self, operator: np.ndarray, site: int) -> complex:
        """
        Compute expectation value of local operator.
        
        Args:
            operator: operator matrix
            site: site to apply operator
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        psi = self.to_state_vector()
        d = self.tensors[0].shape[0]  # physical dimension
        
        # Build full operator (identity on other sites)
        full_op: Union[float, np.ndarray] = 1.0
        for i in range(self.n_sites):
            if i == site:
                full_op = np.kron(full_op, operator)
            else:
                full_op = np.kron(full_op, np.eye(d))
        
        return complex(np.vdot(psi, full_op @ psi))


def create_ghz_mps(n: int) -> MPS:
    """
    Create GHZ state in MPS form (Example from Section 5).
    
    |GHZ⟩ = (|00...0⟩ + |11...1⟩) / 2
    
    Args:
        n: number of qubits
        
    Returns:
        MPS representation of GHZ state
    """
    tensors = []
    
    # First tensor
    first = np.zeros((2, 2), dtype=complex)
    first[0, 0] = 1.0
    first[1, 1] = 1.0
    first /= np.sqrt(2)
    tensors.append(Tensor(first, ["phys", "bond"]))
    
    # Middle tensors (COPY tensors)
    for i in range(n - 2):
        middle = np.zeros((2, 2, 2), dtype=complex)
        middle[0, 0, 0] = 1.0
        middle[1, 1, 1] = 1.0
        tensors.append(Tensor(middle, ["bond_l", "phys", "bond_r"]))
    
    # Last tensor
    last = np.zeros((2, 2), dtype=complex)
    last[0, 0] = 1.0
    last[1, 1] = 1.0
    tensors.append(Tensor(last, ["bond", "phys"]))
    
    return MPS(tensors, [2] * (n - 1))


def create_w_mps(n: int) -> MPS:
    """
    Create W state in MPS form (Example from Section 5).
    
    |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩) / n
    
    Args:
        n: number of qubits
        
    Returns:
        MPS representation of W state
    """
    tensors = []
    norm = 1.0 / np.sqrt(n)
    
    # First tensor: [physical] -> [bond]
    first = np.zeros((2, 2), dtype=complex)
    first[1, 0] = norm  # |1⟩ state
    first[0, 1] = norm  # Continue chain
    tensors.append(Tensor(first, ["phys", "bond"]))
    
    # Middle tensors: [bond_l] -> [physical, bond_r]
    for i in range(n - 2):
        middle = np.zeros((2, 2, 2), dtype=complex)
        middle[0, 0, 0] = 1.0    # |0⟩ with no excitation
        middle[1, 1, 0] = 1.0    # |1⟩ with incoming excitation
        middle[1, 0, 1] = 1.0    # |0⟩ passing excitation forward
        tensors.append(Tensor(middle, ["bond_l", "phys", "bond_r"]))
    
    # Last tensor: [bond] -> [physical]
    last = np.zeros((2, 2), dtype=complex)
    last[0, 1] = 1.0  # |0⟩ with no excitation
    last[1, 0] = 1.0  # |1⟩ at the end
    tensors.append(Tensor(last, ["bond", "phys"]))
    
    return MPS(tensors, [2] * (n - 1))

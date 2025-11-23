import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensor_core.tensor_ops import Tensor 


class QuantumGate:
    def __init__(self, matrix: NDArray[np.complex128], name: str) -> None:
        self.matrix: NDArray[np.complex128] = matrix
        self.name: str = name
        self.n_qubits: int = int(np.log2(matrix.shape[0]))
    
    def as_tensor(self) -> Tensor:
        d: int = 2
        shape = [d] * (2 * self.n_qubits)  # n inputs, n outputs
        data = self.matrix.reshape(shape)
        
        labels = [f"out{i}" for i in range(self.n_qubits)] + \
                [f"in{i}" for i in range(self.n_qubits)]
        
        return Tensor(data, labels)
    
    def __matmul__(self, other: object) -> 'QuantumGate':
        if isinstance(other, QuantumGate):
            return QuantumGate(self.matrix @ other.matrix, 
                             f"{self.name}·{other.name}")
        return NotImplemented


# Single-qubit gates

def hadamard() -> QuantumGate:
    H: NDArray[np.complex128] = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    return QuantumGate(H, "H")


def pauli_x() -> QuantumGate:
    X: NDArray[np.complex128] = np.array([[0, 1], [1, 0]], dtype=complex)
    return QuantumGate(X, "X")


def pauli_y() -> QuantumGate:
    Y: NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return QuantumGate(Y, "Y")


def pauli_z() -> QuantumGate:
    Z: NDArray[np.complex128] = np.array([[1, 0], [0, -1]], dtype=complex)
    return QuantumGate(Z, "Z")


def phase(theta: float) -> QuantumGate:
    P: NDArray[np.complex128] = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
    return QuantumGate(P, f"P({theta:.2f})")


# Two-qubit gates

def cnot() -> QuantumGate:
    CNOT: NDArray[np.complex128] = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    return QuantumGate(CNOT, "CNOT")


def swap_gate() -> QuantumGate:
    SWAP: NDArray[np.complex128] = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    return QuantumGate(SWAP, "SWAP")


def cz() -> QuantumGate:
    CZ: NDArray[np.complex128] = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=complex)
    return QuantumGate(CZ, "CZ")


# Special tensor operations

def copy_tensor() -> Tensor:
    data: NDArray[np.complex128] = np.zeros((2, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                data[i, j, k] = (1-i)*(1-j)*(1-k) + i*j*k
    return Tensor(data, ["out1", "out2", "in"])


def xor_tensor() -> Tensor:
    data: NDArray[np.complex128] = np.zeros((2, 2, 2), dtype=complex)
    for q in range(2):
        for r in range(2):
            for s in range(2):
                data[q, r, s] = 1 - (q+r+s) + 2*(q*r + q*s + s*r) - 4*q*r*s
    return Tensor(data, ["out", "in1", "in2"])


# Bell states and entanglement

def bell_state(which: str = "phi+") -> NDArray[np.float64]:
    states: Dict[str, NDArray[np.float64]] = {
        "phi+": np.array([1, 0, 0, 1]) / np.sqrt(2),   # (|00⟩ + |11⟩)/2
        "phi-": np.array([1, 0, 0, -1]) / np.sqrt(2),  # (|00⟩ - |11⟩)/2
        "psi+": np.array([0, 1, 1, 0]) / np.sqrt(2),   # (|01⟩ + |10⟩)/2
        "psi-": np.array([0, 1, -1, 0]) / np.sqrt(2),  # (|01⟩ - |10⟩)/2 (singlet)
    }
    return states[which]


def create_bell_state(which: str = "phi+") -> NDArray[np.complex128]:
    psi: NDArray[np.complex128] = np.array([1, 0, 0, 0], dtype=complex)
    
    # Apply HI
    H = hadamard().matrix
    I = np.eye(2)
    HI = np.kron(H, I)
    psi = HI @ psi
    
    # Apply CNOT
    CNOT = cnot().matrix
    psi = CNOT @ psi
    
    return psi


def concurrence(psi: NDArray[np.complex128]) -> float:
    psi_matrix: NDArray[np.complex128] = psi.reshape(2, 2)
    C: float = float(2 * np.abs(np.linalg.det(psi_matrix)))
    
    return C


class QuantumCircuit:
    def __init__(self, n_qubits: int) -> None:
        self.n_qubits: int = n_qubits
        self.state: NDArray[np.complex128] = np.zeros(2**n_qubits, dtype=complex)
        self.state[0] = 1.0
        self.gates: List[Tuple[str, List[int]]] = []
    
    def apply_gate(self, gate: QuantumGate, qubits: List[int]) -> None:
        assert len(qubits) == gate.n_qubits
        
        # Build full operator
        full_op = self._expand_gate(gate.matrix, qubits)
        
        # Apply to state
        self.state = full_op @ self.state
        
        self.gates.append((gate.name, qubits))
    
    def _expand_gate(self, gate: NDArray[np.complex128], qubits: List[int]) -> NDArray[np.complex128]:
        n: int = gate.shape[0]
        d: int = int(np.log2(n))
        
        # Create operator acting on full space
        ops: List[NDArray[np.complex128]] = []
        qubit_set = set(qubits)
        
        for i in range(self.n_qubits):
            if i in qubit_set:
                # Find position in qubit list
                pos = qubits.index(i)
                # Extract corresponding 2x2 block if multi-qubit gate
                if d == 1:
                    ops.append(gate)
                else:
                    # This is simplified; proper implementation would handle arbitrary gates
                    ops.append(np.eye(2, dtype=complex))
            else:
                ops.append(np.eye(2, dtype=complex))
        
        # Kronecker product
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        
        # For multi-qubit gates, need to properly expand
        if d > 1:
            # Simplified: just use the gate directly on specified qubits
            # Full implementation would permute and expand properly
            if set(qubits) == {0, 1} and self.n_qubits == 2:
                result = gate
        
        return result
    
    def measure_all(self) -> NDArray[np.float64]:
        return np.abs(self.state)**2  # type: ignore[return-value]
    
    def get_state(self) -> NDArray[np.complex128]:
        return self.state.copy()
    
    def expectation(self, operator: NDArray[np.complex128]) -> complex:
        return complex(np.vdot(self.state, operator @ self.state))

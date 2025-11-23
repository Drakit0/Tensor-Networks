import numpy as np
import time
from typing import Dict, Any
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator, StatevectorSimulator
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.tensor_core.mps import create_ghz_mps
    
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, parent_dir)
    from src.tensor_core.mps import create_ghz_mps


def create_ghz_circuit(n_qubits: int) -> Any:  # QuantumCircuit
    """
    Create GHZ state circuit: (|000...0⟩ + |111...1⟩) / 2
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Quantum circuit
    """
    qc = QuantumCircuit(n_qubits)
    
    # Apply H to first qubit
    qc.h(0)
    
    # Apply CNOT chain
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    
    return qc


def create_w_circuit(n_qubits: int) -> Any:  # QuantumCircuit
    """
    Create W state circuit: (|1000...⟩ + |0100...⟩ + ... + |000...1⟩) / n
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Quantum circuit
    """
    qc = QuantumCircuit(n_qubits)
    
    # Recursive construction of W state
    # This is a simplified version
    qc.ry(2 * np.arccos(np.sqrt((n_qubits - 1) / n_qubits)), 0)
    
    for i in range(1, n_qubits):
        angle = 2 * np.arccos(np.sqrt((n_qubits - i - 1) / (n_qubits - i)))
        qc.cry(angle, i - 1, i)
    
    qc.x(0)
    for i in range(1, n_qubits):
        qc.cx(i - 1, i)
    
    return qc


def create_random_circuit(n_qubits: int, depth: int, seed: int = 42) -> Any:  # QuantumCircuit
    """
    Create random quantum circuit.
    
    Args:
        n_qubits: Number of qubits
        depth: Circuit depth
        seed: Random seed
        
    Returns:
        Random quantum circuit
    """
    np.random.seed(seed)
    qc = QuantumCircuit(n_qubits)
    
    gates = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']
    
    for _ in range(depth):
        for qubit in range(n_qubits):
            gate = np.random.choice(gates)
            
            if gate == 'h':
                qc.h(qubit)
            elif gate == 'x':
                qc.x(qubit)
            elif gate == 'y':
                qc.y(qubit)
            elif gate == 'z':
                qc.z(qubit)
            elif gate in ['rx', 'ry', 'rz']:
                angle = np.random.uniform(0, 2 * np.pi)
                if gate == 'rx':
                    qc.rx(angle, qubit)
                elif gate == 'ry':
                    qc.ry(angle, qubit)
                else:
                    qc.rz(angle, qubit)
            elif gate == 'cx' and n_qubits > 1:
                target = np.random.choice([q for q in range(n_qubits) if q != qubit])
                qc.cx(qubit, target)
    
    return qc


def benchmark_qiskit_simulation(circuit: Any,  # QuantumCircuit
                                method: str = 'statevector') -> Dict[str, Any]:
    """
    Benchmark Qiskit simulation.
    
    Args:
        circuit: Quantum circuit to simulate
        method: Simulation method ('statevector', 'matrix_product_state')
        
    Returns:
        Dictionary with timing and memory info
    """
    n_qubits = circuit.num_qubits
    
    # Create simulator
    simulator: Any
    try:
        if method == 'statevector':
            simulator = StatevectorSimulator()
        elif method == 'matrix_product_state':
            simulator = AerSimulator(method='matrix_product_state')
        else:
            simulator = AerSimulator(method=method)
    except Exception as e:
        # Fallback to basic Aer simulator
        simulator = AerSimulator()
        if method == 'statevector':
            simulator.set_options(method='statevector')
        elif method == 'matrix_product_state':
            simulator.set_options(method='matrix_product_state')
    
    # Add measurements
    qc: Any = circuit.copy()
    qc.measure_all()
    
    # Transpile
    qc_transpiled: Any = transpile(qc, simulator)
    
    # Benchmark execution
    start_time: float = time.time()
    try:
        result: Any = simulator.run(qc_transpiled, shots=1).result()
    except Exception as e:
        print(f"  Warning: Simulation failed for {n_qubits} qubits with {method}: {e}")
        raise
    end_time: float = time.time()
    
    execution_time: float = (end_time - start_time) * 1000  # ms
    
    # Estimate memory (statevector requires 2^n complex numbers)
    memory_mb: float | None
    if method == 'statevector':
        memory_mb = (2**n_qubits * 16) / (1024 * 1024)  # 16 bytes per complex128
    else:
        memory_mb = None  # MPS memory depends on entanglement
    
    return {
        'method': method,
        'n_qubits': n_qubits,
        'execution_time_ms': execution_time,
        'memory_mb': memory_mb,
        'result': result
    }


def compare_simulation_methods(n_qubits_range: range, circuit_type: str = 'ghz') -> None:
    """
    Compare different simulation methods across qubit counts.
    
    Args:
        n_qubits_range: Range of qubit counts
        circuit_type: Type of circuit ('ghz', 'w', 'random')
    """
    print(f"\n{'='*80}")
    print(f"  QUANTUM SIMULATION BENCHMARK: {circuit_type.upper()} States")
    print(f"{'='*80}\n")
    
    print(f"{'Qubits':<10} {'Statevector (ms)':<20} {'MPS (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for n in n_qubits_range:
        # Create circuit
        qc: Any
        if circuit_type == 'ghz':
            qc = create_ghz_circuit(n)
        elif circuit_type == 'w':
            qc = create_w_circuit(n)
        elif circuit_type == 'random':
            qc = create_random_circuit(n, depth=10)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Benchmark statevector (only for small n)
        sv_time: float | None
        memory: float
        if n <= 20:  # Limit statevector to prevent memory issues
            try:
                sv_result: Dict[str, Any] = benchmark_qiskit_simulation(qc, 'statevector')
                sv_time = sv_result['execution_time_ms']
                memory = sv_result['memory_mb']
            except Exception as e:
                sv_time = None
                memory = 2**n * 16 / (1024 * 1024)  # Estimated
        else:
            sv_time = None
            memory = 2**n * 16 / (1024 * 1024)  # Estimated
        
        # Benchmark MPS
        mps_time: float | None
        try:
            mps_result: Dict[str, Any] = benchmark_qiskit_simulation(qc, 'matrix_product_state')
            mps_time = mps_result['execution_time_ms']
        except Exception as e:
            mps_time = None
        
        # Print results
        if sv_time and mps_time:
            speedup = sv_time / mps_time
            print(f"{n:<10} {sv_time:<20.2f} {mps_time:<15.2f} {memory:<15.1f} {speedup:<10.2f}x")
        elif not sv_time and mps_time:
            print(f"{n:<10} {'TOO LARGE':<20} {mps_time:<15.2f} {memory:<15.1f} {'':<10}")
        elif sv_time and not mps_time:
            print(f"{n:<10} {sv_time:<20.2f} {'FAILED':<15} {memory:<15.1f} {'N/A':<10}")
        else:
            print(f"{n:<10} {'TOO LARGE':<20} {'TOO LARGE':<15} {memory:<15.1f} {'N/A':<10}")


def demonstrate_entanglement_structure() -> None:
    """
    Demonstrate how entanglement structure affects MPS efficiency.
    """
    print(f"\n{'='*80}")
    print(f"  ENTANGLEMENT STRUCTURE AND MPS EFFICIENCY")
    print(f"{'='*80}\n")
    
    n_qubits = 20
    
    print(f"Comparing states with {n_qubits} qubits:\n")
    
    # 1. GHZ state (low entanglement, MPS-friendly)
    print("1. GHZ State (|00...0⟩ + |11...1⟩) / 2")
    print("   - Entanglement: Bounded across all bipartitions")
    print("   - MPS bond dimension: χ = 2")
    
    ghz_circuit: Any = create_ghz_circuit(n_qubits)
    ghz_result: Dict[str, Any] = benchmark_qiskit_simulation(ghz_circuit, 'matrix_product_state')
    print(f"   - Simulation time: {ghz_result['execution_time_ms']:.2f} ms")
    
    # Create MPS directly
    mps_ghz: Any = create_ghz_mps(n_qubits)
    full_size: int = 2**n_qubits
    mps_size: int = sum(t.data.size for t in mps_ghz.tensors)
    print(f"   - Compression: {full_size / mps_size:.0f}x ({mps_size} vs {full_size:,} parameters)")
    
    # 2. W state (moderate entanglement)
    print("\n2. W State (|100...⟩ + |010...⟩ + ... + |...001⟩) / n")
    print("   - Entanglement: Moderate")
    print("   - MPS bond dimension: χ = 2")
    
    w_circuit: Any = create_w_circuit(n_qubits)
    w_result: Dict[str, Any] = benchmark_qiskit_simulation(w_circuit, 'matrix_product_state')
    print(f"   - Simulation time: {w_result['execution_time_ms']:.2f} ms")
    
    # 3. Random circuit (high entanglement)
    print("\n3. Random Circuit (High Entanglement)")
    print("   - Entanglement: High, grows with depth")
    print("   - MPS bond dimension: χ grows exponentially")
    
    random_circuit: Any = create_random_circuit(n_qubits, depth=5)
    try:
        random_result: Dict[str, Any] = benchmark_qiskit_simulation(random_circuit, 'matrix_product_state')
        print(f"   - Simulation time: {random_result['execution_time_ms']:.2f} ms")
    except Exception as e:
        print(f"   - Simulation: Failed (too entangled)")
    
    print(f"\n{'='*80}")
    print("  💡 KEY INSIGHT:")
    print("  Tensor networks (MPS) excel at low-entanglement states,")
    print("  enabling simulation of 50+ qubits that would be impossible")
    print("  with standard methods (requiring petabytes of memory).")
    print(f"{'='*80}\n")


def quantum_advantage_demonstration() -> None:
    """
    Demonstrate where tensor networks provide quantum advantage.
    """
    print(f"\n{'='*80}")
    print(f"  QUANTUM COMPUTATIONAL ADVANTAGE WITH TENSOR NETWORKS")
    print(f"{'='*80}\n")
    
    scenarios = [
        {
            'name': '1D Quantum System (Nearest-neighbor)',
            'qubits': [10, 20, 30, 40, 50],
            'circuit_type': 'ghz',
            'description': 'Perfect for MPS - bounded entanglement'
        },
        {
            'name': 'Random Quantum Circuit',
            'qubits': [10, 15, 20, 25],
            'circuit_type': 'random',
            'description': 'Difficult for MPS - high entanglement'
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"  {scenario['description']}")
        print("-" * 80)
        
        qubits_list: Any = scenario['qubits']
        for n in qubits_list:
            qc: Any
            if scenario['circuit_type'] == 'ghz':
                qc = create_ghz_circuit(n)
            else:
                qc = create_random_circuit(n, depth=10)
            
            # MPS simulation
            try:
                result: Dict[str, Any] = benchmark_qiskit_simulation(qc, 'matrix_product_state')
                time_ms: float = result['execution_time_ms']
                
                # Estimate statevector memory
                sv_memory_gb: float = (2**int(n) * 16) / (1024**3)
                
                status: str
                if sv_memory_gb < 1:
                    status = " Feasible"
                elif sv_memory_gb < 100:
                    status = " Challenging"
                else:
                    status = " Impossible"
                
                print(f"  {n:2d} qubits: {time_ms:>8.2f} ms | "
                      f"Statevector would need {sv_memory_gb:>8.2f} GB {status}")
            except Exception as e:
                print(f"  {n:2d} qubits: Failed (too entangled)")
    
    print(f"\n{'='*80}")
    print("  🚀 BREAKTHROUGH:")
    print("   GHZ-type states: Can simulate 50+ qubits effortlessly")
    print("   Standard method: Limited to ~30 qubits (requires 16 GB RAM)")
    print("   Tensor networks turn 'impossible' into 'routine'!")
    print(f"{'='*80}\n")


def main() -> None:
    """Run comprehensive quantum simulation demonstrations."""
    print(f"\n{'='*80}")
    print("  QUANTUM SIMULATION WITH QISKIT & TENSOR NETWORKS")
    print(f"{'='*80}\n")
    
    print("Checking Qiskit installation...")
    try:
        import qiskit
        print(f" Qiskit version: {qiskit.__version__}")
    except ImportError:
        print(" Qiskit not installed. Run: pip install qiskit qiskit-aer")
        return
    
    try:
        import qiskit_aer
        print(f" Qiskit Aer version: {qiskit_aer.__version__}")
    except ImportError:
        print(" Qiskit Aer not found. Some features may be limited.")
    
    print()
    
    try:
        # 1. Compare simulation methods for GHZ states
        print("\n[1/3] Comparing simulation methods...")
        compare_simulation_methods(range(10, 21, 5), 'ghz')
        
        # 2. Demonstrate entanglement structure
        print("\n[2/3] Demonstrating entanglement structure...")
        demonstrate_entanglement_structure()
        
        # 3. Show quantum advantage
        print("\n[3/3] Showing quantum advantage...")
        quantum_advantage_demonstration()
        
        print(f"\n{'='*80}")
        print("   ALL QUANTUM SIMULATIONS COMPLETE!")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n\n Interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("  1. Make sure qiskit and qiskit-aer are installed:")
        print("     pip install qiskit qiskit-aer")
        print("  2. Check that you have enough memory for large simulations")
        print("  3. Try reducing the number of qubits")


if __name__ == "__main__":
    main()

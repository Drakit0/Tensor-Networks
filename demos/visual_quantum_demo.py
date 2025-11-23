import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import sys
import os
from matplotlib import cm
from qiskit import QuantumCircuit
from typing import Dict, Any, List, Optional

# Add parent directory to path to access src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Create output directory for generated images (in parent directory)
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), '..', 'generated_images', 'quantum')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.quantum.qiskit_simulation import (
    create_ghz_circuit, create_w_circuit, create_random_circuit,
    benchmark_qiskit_simulation
)
from src.tensor_core.mps import create_ghz_mps, MPS
from qiskit.visualization import circuit_drawer

def visualize_quantum_state_comparison(n_qubits=10) -> None:
    """
    Visualize the difference between full statevector and MPS representations.
    """
    print("\n" + "="*80)
    print("  VISUALIZATION 1: QUANTUM STATE REPRESENTATIONS")
    print("="*80)
    
    fig: Figure
    axes: NDArray[Any]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Quantum State Representations for {n_qubits} Qubits', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 1. Full statevector size
    ax1: Axes = axes[0]
    statevector_size: int = 2**n_qubits
    complexity_range: List[int] = [2**i for i in range(1, n_qubits+1)]
    
    ax1.semilogy(range(1, n_qubits+1), complexity_range, 'ro-', linewidth=2, markersize=8)
    ax1.fill_between(range(1, n_qubits+1), complexity_range, alpha=0.3, color='red')
    ax1.set_xlabel('Number of Qubits', fontsize=11)
    ax1.set_ylabel('Statevector Size', fontsize=11)
    ax1.set_title('Exponential Growth\n(Standard Method)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1e6, color='orange', linestyle='--', label='1 Million params')
    ax1.legend()
    
    # 2. MPS tensor structure
    ax2: Axes = axes[1]
    # Draw MPS structure
    mps: MPS = create_ghz_mps(n_qubits)
    tensor_sizes: List[int] = [t.data.size for t in mps.tensors]
    positions: NDArray[Any] = np.arange(len(tensor_sizes))
    
    colors: NDArray[Any] = cm.get_cmap('viridis')(np.linspace(0.2, 0.8, len(tensor_sizes)))
    ax2.bar(positions, tensor_sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Tensor Position', fontsize=11)
    ax2.set_ylabel('Tensor Size', fontsize=11)
    ax2.set_title(f'MPS Representation\n(Total: {sum(tensor_sizes)} params)', 
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Compression ratio
    ax3: Axes = axes[2]
    qubit_range: range = range(5, min(n_qubits + 5, 31))
    compression_ratios: List[float] = []
    
    for n in qubit_range:
        full: int = 2**n
        # Estimate MPS size for GHZ (very efficient)
        mps_est: int = n * 8  # Rough estimate: n tensors, ~8 params each for GHZ
        compression_ratios.append(full / mps_est)
    
    ax3.semilogy(list(qubit_range), compression_ratios, 'go-', linewidth=2, markersize=8)
    ax3.fill_between(list(qubit_range), compression_ratios, alpha=0.3, color='green')
    ax3.set_xlabel('Number of Qubits', fontsize=11)
    ax3.set_ylabel('Compression Ratio', fontsize=11)
    ax3.set_title('MPS Compression Advantage\n(GHZ State)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path: str = os.path.join(OUTPUT_DIR, 'quantum_state_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Visualization saved: {output_path}")
    plt.show()
    
    print(f"\n📊 Key Insights:")
    print(f"   Full statevector: {statevector_size:,} complex numbers")
    print(f"   MPS representation: {sum(tensor_sizes)} parameters")
    print(f"   Compression ratio: {statevector_size / sum(tensor_sizes):.0f}x")
    print(f"   Memory saved: {(1 - sum(tensor_sizes)/statevector_size)*100:.1f}%\n")


def visualize_entanglement_growth() -> None:
    """
    Visualize how entanglement grows in different quantum circuits.
    """
    print("\n" + "="*80)
    print("  VISUALIZATION 2: ENTANGLEMENT GROWTH")
    print("="*80)
    
    n_qubits: int = 15
    depths: range = range(1, 11)
    
    fig: Figure
    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Entanglement Growth for {n_qubits} Qubits', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Simulate circuits of different depths
    ghz_times: List[Optional[float]] = []
    random_times: List[Optional[float]] = []
    
    print("\n📊 Simulating circuits with different depths...\n")
    
    for depth in depths:
        # GHZ-like circuit (low entanglement)
        ghz_circuit: QuantumCircuit = create_ghz_circuit(n_qubits)
        result: Dict[str, Any]
        try:
            result = benchmark_qiskit_simulation(ghz_circuit, 'matrix_product_state')
            ghz_times.append(result['execution_time_ms'])
        except:
            ghz_times.append(None)
        
        # Random circuit (high entanglement grows with depth)
        random_circuit: QuantumCircuit = create_random_circuit(n_qubits, depth=depth)
        try:
            result = benchmark_qiskit_simulation(random_circuit, 'matrix_product_state')
            random_times.append(result['execution_time_ms'])
        except:
            random_times.append(None)
        
        print(f"  Depth {depth}: GHZ={ghz_times[-1]:.1f}ms, Random={random_times[-1]:.1f}ms" 
              if ghz_times[-1] and random_times[-1] else f"  Depth {depth}: Failed")
    
    # Plot 1: Execution time vs depth
    ghz_times_clean: List[float] = [t for t in ghz_times if t is not None]
    random_times_clean: List[float] = [t for t in random_times if t is not None]
    ax1.plot(list(depths)[:len(ghz_times_clean)], ghz_times_clean, 'b-o', linewidth=2, markersize=8, label='GHZ State (Low Entanglement)')
    ax1.plot(list(depths)[:len(random_times_clean)], random_times_clean, 'r-s', linewidth=2, markersize=8, label='Random Circuit (High Entanglement)')
    ax1.set_xlabel('Circuit Depth', fontsize=11)
    ax1.set_ylabel('Execution Time (ms)', fontsize=11)
    ax1.set_title('MPS Simulation Time vs Circuit Depth', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entanglement structure illustration
    ax2.axis('off')
    
    # Draw two quantum circuits side by side
    y_ghz: float = 0.7
    y_random: float = 0.3
    
    # GHZ structure
    ax2.text(0.5, y_ghz + 0.15, 'GHZ State Structure', 
            ha='center', fontsize=12, fontweight='bold', color='blue')
    for i in range(n_qubits):
        x = 0.1 + (i * 0.8 / n_qubits)
        ax2.add_patch(Rectangle((x, y_ghz), 0.05, 0.08, facecolor='lightblue', edgecolor='blue', linewidth=2))
        if i < n_qubits - 1:
            ax2.plot([x + 0.025, x + 0.8/n_qubits + 0.025], [y_ghz + 0.04, y_ghz + 0.04], 
                    'b-', linewidth=2)
    ax2.text(0.5, y_ghz - 0.1, ' Linear entanglement chain', ha='center', fontsize=10, style='italic')
    
    # Random structure
    ax2.text(0.5, y_random + 0.15, 'Random Circuit Structure', 
            ha='center', fontsize=12, fontweight='bold', color='red')
    for i in range(n_qubits):
        x_pos: float = 0.1 + (i * 0.8 / n_qubits)
        ax2.add_patch(Rectangle((x_pos, y_random), 0.05, 0.08, facecolor='lightcoral', edgecolor='red', linewidth=2))
        # Random connections
        if i < n_qubits - 2 and i % 2 == 0:
            target_x_pos: float = 0.1 + ((i + 2) * 0.8 / n_qubits) + 0.025
            ax2.plot([x_pos + 0.025, target_x_pos], [y_random + 0.04, y_random + 0.04],
                    'r--', linewidth=1.5, alpha=0.6)
    ax2.text(0.5, y_random - 0.1, ' Complex entanglement web', ha='center', fontsize=10, style='italic')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path: str = os.path.join(OUTPUT_DIR, 'entanglement_growth.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved: {output_path}")
    plt.show()
    print()


def visualize_memory_comparison() -> None:
    """
    Visualize memory requirements for different simulation methods.
    """
    print("\n" + "="*80)
    print("  VISUALIZATION 3: MEMORY REQUIREMENTS")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Memory Requirements: Statevector vs MPS', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    qubit_range: range = range(5, 41, 5)
    statevector_memory: List[float] = []
    mps_memory_ghz: List[float] = []
    
    for n in qubit_range:
        # Statevector: 2^n complex128 numbers
        sv_mem: float = (2**n * 16) / (1024**3)  # GB
        statevector_memory.append(sv_mem)
        
        # MPS for GHZ: very efficient
        mps_mem: float = (n * 8 * 16) / (1024**3)  # GB (rough estimate)
        mps_memory_ghz.append(mps_mem)
    
    # Plot 1: Absolute memory
    ax1.semilogy(list(qubit_range), statevector_memory, 'ro-', 
                linewidth=2.5, markersize=10, label='Statevector Method')
    ax1.semilogy(list(qubit_range), mps_memory_ghz, 'go-', 
                linewidth=2.5, markersize=10, label='MPS Method (GHZ)')
    
    # Add memory threshold lines
    ax1.axhline(y=16, color='orange', linestyle='--', linewidth=2, label='16 GB RAM (typical)')
    ax1.axhline(y=1, color='yellow', linestyle='--', linewidth=2, alpha=0.7, label='1 GB')
    
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Memory Required (GB)', fontsize=12)
    ax1.set_title('Absolute Memory Requirements', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-6, 1e6)
    
    # Plot 2: Memory ratio (how many times smaller MPS is)
    memory_ratios: List[float] = [sv / mps for sv, mps in zip(statevector_memory, mps_memory_ghz)]
    
    ax2.semilogy(list(qubit_range), memory_ratios, 'bo-', 
                linewidth=2.5, markersize=10)
    ax2.fill_between(list(qubit_range), memory_ratios, alpha=0.3, color='blue')
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('Memory Reduction Factor', fontsize=12)
    ax2.set_title('MPS Memory Advantage\n(Higher is Better)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (n, ratio) in enumerate(zip(qubit_range, memory_ratios)):
        if i % 2 == 0:  # Annotate every other point
            ax2.annotate(f'{ratio:.0f}x', 
                        xy=(n, ratio), 
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, 'memory_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f" Visualization saved: {output_path}")
    plt.show()
    
    print(f"\n📊 Memory Analysis:")
    print(f"  {'Qubits':<10} {'Statevector (GB)':<20} {'MPS (GB)':<15} {'Reduction':<15}")
    print("  " + "-"*60)
    for n, sv, mps in zip(qubit_range, statevector_memory, mps_memory_ghz):
        ratio = sv / mps
        print(f"  {n:<10} {sv:<20.2e} {mps:<15.2e} {ratio:<15.0f}x")
    print()


def benchmark_live_simulation() -> None:
    """
    Run live simulation benchmark with progress visualization.
    """
    print("\n" + "="*80)
    print("  VISUALIZATION 4: LIVE SIMULATION BENCHMARK")
    print("="*80)
    
    print("\n🔬 Running live simulations...\n")
    
    # Test different qubit counts
    qubit_counts: List[int] = [10, 15, 20, 25]
    results: List[Dict[str, Any]] = []
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Real-Time Quantum Simulation Benchmark', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for i, n in enumerate(qubit_counts):
        print(f"  [{i+1}/{len(qubit_counts)}] Simulating {n} qubits...")
        
        # Create GHZ circuit
        circuit: QuantumCircuit = create_ghz_circuit(n)
        
        # Benchmark
        try:
            # Statevector (only for smaller systems)
            sv_result: Dict[str, Any]
            sv_time: Optional[float]
            sv_memory: float
            if n <= 20:
                sv_result = benchmark_qiskit_simulation(circuit, 'statevector')
                sv_time = sv_result['execution_time_ms']
                sv_memory = sv_result['memory_mb']
            else:
                sv_time = None
                sv_memory = (2**n * 16) / (1024**2)
            
            # MPS
            mps_result: Dict[str, Any] = benchmark_qiskit_simulation(circuit, 'matrix_product_state')
            mps_time: float = mps_result['execution_time_ms']
            
            results.append({
                'n_qubits': n,
                'sv_time': sv_time,
                'mps_time': mps_time,
                'sv_memory': sv_memory,
                'speedup': sv_time / mps_time if sv_time else None
            })
            
            status = "" if sv_time else ""
            speedup_str = f"{sv_time/mps_time:.1f}x faster" if sv_time else " (SV impossible)"
            print(f"      {status} MPS: {mps_time:.2f}ms | {speedup_str}")
            
        except Exception as e:
            print(f"       Failed: {e}")
            results.append({
                'n_qubits': n,
                'sv_time': None,
                'mps_time': None,
                'sv_memory': (2**n * 16) / (1024**2),
                'speedup': None
            })
    
    # Plot results
    valid_results: List[Dict[str, Any]] = [r for r in results if r['mps_time'] is not None]
    
    if valid_results:
        qubits: List[int] = [r['n_qubits'] for r in valid_results]
        mps_times: List[float] = [r['mps_time'] for r in valid_results]
        sv_times: List[Any] = [r['sv_time'] if r['sv_time'] else np.nan for r in valid_results]
        
        # Plot 1: Execution times
        ax1.plot(qubits, mps_times, 'go-', linewidth=2.5, markersize=10, label='MPS Method')
        
        # Only plot statevector for small systems
        sv_qubits: List[int] = [r['n_qubits'] for r in valid_results if r['sv_time']]
        sv_times_valid: List[float] = [r['sv_time'] for r in valid_results if r['sv_time']]
        if sv_times_valid:
            ax1.plot(sv_qubits, sv_times_valid, 'ro-', linewidth=2.5, markersize=10, label='Statevector Method')
        
        ax1.set_xlabel('Number of Qubits', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title('Simulation Time Comparison', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Speedup
        speedups: List[float] = [r['speedup'] for r in valid_results if r['speedup']]
        speedup_qubits: List[int] = [r['n_qubits'] for r in valid_results if r['speedup']]
        
        if speedups:
            bars = ax2.bar(speedup_qubits, speedups, color='steelblue', 
                          edgecolor='black', linewidth=1.5, alpha=0.7)
            
            # Color bars by speedup magnitude
            for bar, speedup in zip(bars, speedups):
                if speedup > 100:
                    bar.set_color('darkgreen')
                elif speedup > 10:
                    bar.set_color('green')
                else:
                    bar.set_color('steelblue')
            
            ax2.set_xlabel('Number of Qubits', fontsize=12)
            ax2.set_ylabel('Speedup Factor', fontsize=12)
            ax2.set_title('MPS Speedup (Higher is Better)', fontsize=11, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{speedup:.1f}x',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, 'live_benchmark.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved: {output_path}")
    plt.show()
    print()


def visualize_circuit_types() -> None:
    """
    Visualize different types of quantum circuits and their MPS efficiency.
    """
    print("\n" + "="*80)
    print("  VISUALIZATION 5: CIRCUIT TYPES & MPS EFFICIENCY")
    print("="*80)
    
    n_qubits: int = 12
    
    circuits: Dict[str, QuantumCircuit] = {
        'GHZ State': create_ghz_circuit(n_qubits),
        'W State': create_w_circuit(n_qubits),
        'Random (Depth 5)': create_random_circuit(n_qubits, depth=5),
        'Random (Depth 10)': create_random_circuit(n_qubits, depth=10),
    }
    
    fig: Figure
    axes: NDArray[Any]
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f'Quantum Circuit Types - {n_qubits} Qubits', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    results: Dict[str, Optional[float]] = {}
    
    print(f"\n🔬 Simulating different circuit types ({n_qubits} qubits)...\n")
    
    for idx, (name, circuit) in enumerate(circuits.items()):
        ax: Axes = axes[idx // 2, idx % 2]
        
        print(f"  [{idx+1}/{len(circuits)}] {name}...")
        
        # Benchmark first
        result: Dict[str, Any]
        try:
            result = benchmark_qiskit_simulation(circuit, 'matrix_product_state')
            time_ms: float = result['execution_time_ms']
            results[name] = time_ms
            print(f"       Simulation time: {time_ms:.2f}ms")
        except Exception as e:
            results[name] = None
            print(f"       Failed: {e}")
        
        # Draw circuit diagram
        try:            
            # Clear the axis first
            ax.clear()
            
            # Draw the circuit directly on this axis
            circuit_drawer(circuit, output='mpl', ax=ax, style='iqp', fold=20, plot_barriers=False)
            
            # Add simulation time as text annotation
            if results[name] is not None:
                ax.text(0.50, 1.02, f'Time: {results[name]:.2f}ms',
                       transform=ax.transAxes,
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8,
                                edgecolor='black', linewidth=1.5))
            
            print(f"       Circuit drawn successfully")
            
        except Exception as e:
            # Fallback to text display if drawing fails
            print(f"       Circuit drawing failed: {e}")
            
            circuit_info: str = f'{name}\n\n'
            circuit_info += f'Qubits: {circuit.num_qubits}\n'
            circuit_info += f'Depth: {circuit.depth()}\n'
            circuit_info += f'Gates: {len(circuit.data)}'
            if results[name] is not None:
                circuit_info += f'\n\nSimulation: {results[name]:.2f}ms'
            
            ax.text(0.5, 0.5, circuit_info, 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.7))
            ax.set_title(f'{name}', fontsize=11, fontweight='bold', pad=10)
            ax.axis('off')
    
    plt.subplots_adjust(top=0.94, bottom=0.02, left=0.03, right=0.98, hspace=0.18, wspace=0.15)
    output_path = os.path.join(OUTPUT_DIR, 'circuit_types.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
    print(f"\n Visualization saved: {output_path}")
    plt.show()
    
    # Summary plot - horizontal bar chart
    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names: List[str] = [n for n, t in results.items() if t is not None]
        times: List[float] = [t for t in results.values() if t is not None]
        
        # Assign colors based on performance (faster = greener)
        max_time: float = max(times) if times else 1
        colors: List[str] = []
        for t in times:
            if t < max_time * 0.5:
                colors.append('darkgreen')
            elif t < max_time * 0.75:
                colors.append('lightgreen')
            elif t < max_time * 0.9:
                colors.append('orange')
            else:
                colors.append('coral')
        
        bars = ax.barh(names, times, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        ax.set_xlabel('Simulation Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'MPS Simulation Time by Circuit Type ({n_qubits} qubits)', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels inside bars
        for bar, time in zip(bars, times):
            width = bar.get_width()
            ax.text(width * 0.95, bar.get_y() + bar.get_height()/2., 
                   f'{time:.2f} ms',
                   ha='right', va='center', fontsize=10, fontweight='bold',
                   color='white', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
        
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'circuit_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Visualization saved: {output_path}")
        plt.show()
    
    print()


def main() -> None:
    """Run visual quantum simulation demo."""
    
    print("\n" + "="*80)
    print("  VISUAL QUANTUM SIMULATION DEMO")
    print("  See Tensor Networks in Action!")
    print("="*80)
    
    print("\n  This demo will create visual representations of:")
    print("    1. Quantum state representations (Statevector vs MPS)")
    print("    2. Entanglement growth in different circuits")
    print("    3. Memory requirements comparison")
    print("    4. Live simulation benchmarks")
    print("    5. Different circuit types and their efficiency")
    
    print("\n  Choose visualization mode:")
    print("    [1] Quick Demo (visualizations 1-3, ~2 min)")
    print("    [2] Full Demo (all visualizations, ~5 min)")
    print("    [3] Custom (choose specific visualizations)")
    print("    [Q] Quit")
    
    choice: str = input("\n  Your choice [1/2/3/Q]: ").strip().upper()
    
    try:
        if choice == '1':
            visualize_quantum_state_comparison(10)
            visualize_entanglement_growth()
            visualize_memory_comparison()
            
        elif choice == '2':
            visualize_quantum_state_comparison(10)
            visualize_entanglement_growth()
            visualize_memory_comparison()
            benchmark_live_simulation()
            visualize_circuit_types()
            
        elif choice == '3':
            print("\n  Select visualizations:")
            print("    [1] State representations")
            print("    [2] Entanglement growth")
            print("    [3] Memory comparison")
            print("    [4] Live benchmark")
            print("    [5] Circuit types")
            
            selections: List[str] = input("\n  Enter numbers separated by spaces (e.g., '1 3 4'): ").strip().split()
            
            for sel in selections:
                if sel == '1':
                    visualize_quantum_state_comparison(10)
                elif sel == '2':
                    visualize_entanglement_growth()
                elif sel == '3':
                    visualize_memory_comparison()
                elif sel == '4':
                    benchmark_live_simulation()
                elif sel == '5':
                    visualize_circuit_types()
                    
        elif choice == 'Q':
            print("\n  Exiting...")
            return
        else:
            print(f"\n  Invalid choice: {choice}")
            return
        
        print("\n" + "="*80)
        print("   VISUAL DEMO COMPLETE!")
        print("="*80)
        print(f"\n  📊 Generated visualizations saved to: {OUTPUT_DIR}/")
        print("     quantum_state_comparison.png")
        print("     entanglement_growth.png")
        print("     memory_comparison.png")
        if choice in ['2', '3']:
            print("     live_benchmark.png")
            print("     circuit_types.png")
            print("     circuit_comparison.png")
        print(f"\n  Check the '{OUTPUT_DIR}' folder to see the tensor network advantages!")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Make sure qiskit and matplotlib are installed:")
        print("     pip install qiskit qiskit-aer matplotlib")


if __name__ == "__main__":
    main()

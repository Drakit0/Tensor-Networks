import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision  
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from datetime import datetime
from typing import Tuple, Dict, Optional, Any, Union, TypeVar
import tensorly as tl
from tensorly.decomposition import tucker, parafac

tl.set_backend('pytorch')

# TypeVar for generic module type
T = TypeVar('T', bound=nn.Module)


class TuckerConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, ranks: Tuple[int, int, int, int],
                 stride: int = 1, padding: int = 0, bias: bool = True) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.ranks = ranks
        self.stride = stride
        self.padding = padding
        
        # Tucker decomposition layers
        # First: reduce input channels
        self.conv_in = nn.Conv2d(in_channels, ranks[1], kernel_size=1, bias=False)
        
        # Second: spatial convolution on reduced space
        self.conv_core = nn.Conv2d(ranks[1], ranks[0], kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=False)
        
        # Third: expand to output channels
        self.conv_out = nn.Conv2d(ranks[0], out_channels, kernel_size=1, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.conv_core(x)
        x = self.conv_out(x)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return x
    
    @staticmethod
    def from_conv2d(conv_layer: nn.Conv2d, rank_ratio: float = 0.5) -> 'TuckerConv2d':
        out_ch: int
        in_ch: int
        h: int
        w: int
        out_ch, in_ch, h, w = conv_layer.weight.shape
        
        ranks: Tuple[int, int, int, int] = (
            max(int(out_ch * rank_ratio), 1),
            max(int(in_ch * rank_ratio), 1),
            h,
            w
        )
        
        # Create decomposed layer
        stride_val = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
        padding_val = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
        tucker_layer = TuckerConv2d(
            in_ch, out_ch, h,
            ranks=ranks,
            stride=int(stride_val),
            padding=int(padding_val),
            bias=conv_layer.bias is not None
        )
        
        # Perform Tucker decomposition on weights
        device = conv_layer.weight.device
        weight = conv_layer.weight.detach().cpu()
        
        # Tucker decomposition
        core, factors = tucker(weight, rank=ranks)
        
        # Assign decomposed weights
        # factors[1] is (in_ch, rank_in) -> need (rank_in, in_ch, 1, 1) for conv_in
        # factors[0] is (out_ch, rank_out) -> need (out_ch, rank_out, 1, 1) for conv_out
        tucker_layer.conv_in.weight.data = factors[1].t().unsqueeze(2).unsqueeze(3).to(device)
        tucker_layer.conv_core.weight.data = core.to(device)
        tucker_layer.conv_out.weight.data = factors[0].unsqueeze(2).unsqueeze(3).to(device)
        
        if conv_layer.bias is not None:
            tucker_layer.bias.data = conv_layer.bias.detach()
        
        return tucker_layer.to(device)
    
    def count_parameters(self) -> int:
        """Count total parameters in decomposed layer."""
        return sum(p.numel() for p in self.parameters())


class CPFullyConnected(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # CP factors
        self.factor_in = nn.Linear(in_features, rank, bias=False)
        self.factor_out = nn.Linear(rank, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.factor_in(x)
        x = self.factor_out(x)
        return x
    
    @staticmethod
    def from_linear(linear_layer: nn.Linear, rank: int) -> 'CPFullyConnected':
        cp_layer: CPFullyConnected = CPFullyConnected(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            bias=linear_layer.bias is not None
        )
        
        # Initialize with decomposition
        device = linear_layer.weight.device
        weight = linear_layer.weight.detach().cpu().t()  # Transpose for decomposition
        
        # CP decomposition
        cp_decomp = parafac(weight, rank=rank)
        
        # Extract factors from CPTensor object
        # cp_decomp is a tuple: (weights, factors) where factors is a list
        if isinstance(cp_decomp, tuple):
            factors = cp_decomp[1]  # factors list
        else:
            factors = cp_decomp.factors  # CPTensor object
        
        # Assign weights correctly
        # After transposing weight to (in_features, out_features):
        # factors[0] is (in_features, rank) - use for factor_in
        # factors[1] is (out_features, rank) - use for factor_out
        cp_layer.factor_in.weight.data = factors[0].t().to(device)  # (rank, in_features)
        cp_layer.factor_out.weight.data = factors[1].to(device)     # (out_features, rank)
        
        if linear_layer.bias is not None:
            cp_layer.factor_out.bias.data = linear_layer.bias.detach()
        
        return cp_layer.to(device)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CompressedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, rank_ratio: float = 0.5, fc_rank: int = 64) -> None:
        super().__init__()
        # Note: We'll replace these with compressed versions after training
        self.conv1: Union[nn.Conv2d, TuckerConv2d] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2: Union[nn.Conv2d, TuckerConv2d] = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3: Union[nn.Conv2d, TuckerConv2d] = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1: Union[nn.Linear, CPFullyConnected] = nn.Linear(256 * 4 * 4, 512)
        self.fc2: Union[nn.Linear, CPFullyConnected] = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
        self.is_compressed = False
        self.rank_ratio = rank_ratio
        self.fc_rank = fc_rank
    
    def compress(self) -> None:
        if self.is_compressed:
            return
        
        print("Compressing CNN with tensor decomposition...")
        
        # Type assertions to ensure we're working with base types
        assert isinstance(self.conv1, nn.Conv2d)
        assert isinstance(self.conv2, nn.Conv2d)
        assert isinstance(self.conv3, nn.Conv2d)
        assert isinstance(self.fc1, nn.Linear)
        assert isinstance(self.fc2, nn.Linear)
        
        self.conv1 = TuckerConv2d.from_conv2d(self.conv1, self.rank_ratio)
        self.conv2 = TuckerConv2d.from_conv2d(self.conv2, self.rank_ratio)
        self.conv3 = TuckerConv2d.from_conv2d(self.conv3, self.rank_ratio)
        
        self.fc1 = CPFullyConnected.from_linear(self.fc1, self.fc_rank)
        self.fc2 = CPFullyConnected.from_linear(self.fc2, self.fc_rank // 2)
        
        self.is_compressed = True
        print("Compression complete!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def benchmark_inference_speed(model: nn.Module, input_shape: Tuple[int, ...], 
                              num_iterations: int = 100, device: str = 'cuda') -> float:
    """
    Benchmark inference speed of a model.
    
    Args:
        model: Model to benchmark
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        num_iterations: Number of iterations to average
        device: Device to run on
        
    Returns:
        Average inference time in milliseconds
    """
    model.eval()
    model.to(device)
    
    # Warm up
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    return float(np.mean(times) * 1000)  # Convert to milliseconds


def train_model(model: T, train_loader: DataLoader,  # type: ignore[type-arg]
                epochs: int = 5, device: str = 'cuda') -> T:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 50 == 49:
                print(f'  Epoch [{epoch+1}/{epochs}], Step [{i+1}], '
                      f'Loss: {running_loss/50:.3f}, Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
    
    print("Training complete!")
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cuda') -> float:  # type: ignore[type-arg]
    """
    Evaluate model accuracy.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Test accuracy (%)
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def save_model_with_metadata(model: nn.Module, model_type: str, accuracy: float,
                            training_time: float, epochs: int, metadata: Optional[Dict[str, Any]] = None,
                            save_dir: str = "models") -> str:
    """
    Save model with metadata.
    
    Args:
        model: Trained model to save
        model_type: Type of model ('baseline' or 'compressed')
        accuracy: Test accuracy (%)
        training_time: Total training time in seconds
        epochs: Number of training epochs
        metadata: Additional metadata to save
        save_dir: Directory to save models
        
    Returns:
        Path to saved model
    """
    # Create models directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Simple filename
    model_path = os.path.join(save_dir, f"{model_type}.pth")
    
    # Prepare metadata
    params_list = list(model.parameters())
    if params_list:
        first_param = params_list[0]
        device_str: str = first_param.device.type
    else:
        device_str = 'cpu'
    
    if hasattr(model, 'count_parameters'):
        param_count: int = model.count_parameters()  # type: ignore[operator]
    else:
        param_count = sum(p.numel() for p in params_list)
    
    model_metadata: Dict[str, Any] = {
        "model_type": model_type,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(accuracy),
        "training_time_seconds": float(training_time),
        "training_time_formatted": f"{training_time//60:.0f}m {training_time%60:.0f}s",
        "epochs": epochs,
        "num_parameters": param_count,
        "device": device_str,
    }
    
    # Add compression info if available
    if hasattr(model, 'is_compressed'):
        model_metadata["is_compressed"] = model.is_compressed
        model_metadata["rank_ratio"] = getattr(model, 'rank_ratio', None)
        model_metadata["fc_rank"] = getattr(model, 'fc_rank', None)
    
    # Add custom metadata
    if metadata:
        model_metadata.update(metadata)
    
    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': model_metadata,
    }, model_path)
    
    print(f"Model saved: {model_path}")
    print(f"  Accuracy: {accuracy:.2f}% | Training: {model_metadata['training_time_formatted']} | Params: {model_metadata['num_parameters']:,}")
    
    return model_path


def comprehensive_cnn_demo(use_cuda: bool = True) -> Dict[str, Any]:
    device: str = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*80}")
    print(f"  REAL CNN COMPRESSION WITH TENSOR NETWORKS")
    print(f"  Device: {device.upper()}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded: {len(trainset)} training, {len(testset)} test images\n")
    
    # Create and train baseline model
    print("="*80)
    print("  BASELINE MODEL (Standard CNN)")
    print("="*80)
    baseline: SimpleCNN = SimpleCNN().to(device)
    baseline_params: int = baseline.count_parameters()
    print(f"Parameters: {baseline_params:,}")
    
    epochs: int = 3
    baseline_train_start = time.time()
    baseline = train_model(baseline, train_loader, epochs=epochs, device=device)
    baseline_train_time = time.time() - baseline_train_start
    
    baseline_acc: float = evaluate_model(baseline, test_loader, device=device)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Training Time: {baseline_train_time//60:.0f}m {baseline_train_time%60:.0f}s")
    
    # Benchmark speed
    baseline_time: float = benchmark_inference_speed(baseline, (128, 3, 32, 32), device=device)
    print(f"Baseline Inference: {baseline_time:.2f} ms/batch (128 images)")
    
    print("\nSaving baseline model...")
    save_model_with_metadata(
        baseline, 
        model_type="baseline",
        accuracy=baseline_acc,
        training_time=baseline_train_time,
        epochs=epochs,
        metadata={
            "inference_time_ms": float(baseline_time),
            "dataset": "CIFAR-10",
            "batch_size": 128
        }
    )
    
    # Create compressed model
    print(f"\n{'='*80}")
    print("  COMPRESSED MODEL (Tensor Network Decomposition)")
    print("="*80)
    
    compressed: CompressedCNN = CompressedCNN(rank_ratio=0.5, fc_rank=64).to(device)
    compressed_train_start = time.time()
    compressed = train_model(compressed, train_loader, epochs=epochs, device=device)
    compressed_train_time = time.time() - compressed_train_start
    
    # Now compress it
    compressed.compress()
    compressed_params = compressed.count_parameters()
    print(f"Parameters: {compressed_params:,}")
    
    # Evaluate compressed model before fine-tuning
    compressed_acc_before: float = evaluate_model(compressed, test_loader, device=device)
    print(f"Compressed Accuracy (before fine-tuning): {compressed_acc_before:.2f}%")
    
    print(f"\nFine-tuning compressed model (1 epoch)...")
    compressed = train_model(compressed, train_loader, epochs=1, device=device)
    
    # Evaluate after fine-tuning
    compressed_acc: float = evaluate_model(compressed, test_loader, device=device)
    print(f"Compressed Accuracy (after fine-tuning): {compressed_acc:.2f}%")
    print(f"Training Time: {compressed_train_time//60:.0f}m {compressed_train_time%60:.0f}s")
    
    # Benchmark compressed speed
    compressed_time: float = benchmark_inference_speed(compressed, (128, 3, 32, 32), device=device)
    print(f"Compressed Inference: {compressed_time:.2f} ms/batch (128 images)")
    
    print("\nSaving compressed model...")
    save_model_with_metadata(
        compressed, 
        model_type="compressed",
        accuracy=compressed_acc,
        training_time=compressed_train_time,
        epochs=epochs,
        metadata={
            "inference_time_ms": float(compressed_time),
            "dataset": "CIFAR-10",
            "batch_size": 128,
            "compression_ratio": float(baseline_params / compressed_params),
            "speedup": float(baseline_time / compressed_time)
        }
    )
    
    print(f"\n{'='*80}")
    print("  COMPRESSION RESULTS")
    print("="*80)
    
    compression_ratio: float = baseline_params / compressed_params
    speedup: float = baseline_time / compressed_time
    accuracy_drop: float = baseline_acc - compressed_acc
    
    print(f"\n  Parameter Reduction:")
    print(f"    Before: {baseline_params:>12,} parameters")
    print(f"    After:  {compressed_params:>12,} parameters")
    print(f"    Ratio:  {compression_ratio:>12.2f}x compression")
    print(f"    Saved:  {(1 - compressed_params/baseline_params)*100:>11.1f}% reduction")
    
    print(f"\n  Inference Speed:")
    print(f"    Before: {baseline_time:>12.2f} ms")
    print(f"    After:  {compressed_time:>12.2f} ms")
    print(f"    Speedup: {speedup:>11.2f}x faster")
    
    print(f"\n  Accuracy:")
    print(f"    Before: {baseline_acc:>12.2f}%")
    print(f"    After:  {compressed_acc:>12.2f}%")
    print(f"    Drop:   {accuracy_drop:>12.2f}%")
    
    print(f"\n{'='*80}")
    print("  KEY INSIGHTS")
    print("="*80)
    print(f"  - Achieved {compression_ratio:.1f}x parameter compression")
    print(f"  - Got {speedup:.1f}x inference speedup on {device.upper()}")
    print(f"  - Accuracy loss only {abs(accuracy_drop):.2f}%")
    print(f"  - Enables deployment on mobile/edge devices")
    print(f"{'='*80}\n")
    
    return {
        'baseline_params': baseline_params,
        'compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'baseline_time': baseline_time,
        'compressed_time': compressed_time,
        'speedup': speedup,
        'baseline_acc': baseline_acc,
        'compressed_acc': compressed_acc,
        'accuracy_drop': accuracy_drop
    }


if __name__ == "__main__":
    # Run comprehensive demo
    results = comprehensive_cnn_demo(use_cuda=True)

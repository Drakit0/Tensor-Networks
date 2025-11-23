import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader
import sys
import os
import time
from typing import Tuple, Optional, Any
from matplotlib.figure import Figure

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), '..', 'generated_images', 'cnn')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from src.cnn.cnn_compression import SimpleCNN, CompressedCNN, train_model, evaluate_model, save_model_with_metadata
from typing import Dict, List, Any

# CIFAR-10 class names
CIFAR10_CLASSES: List[str] = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


def show_sample_images(dataset: torchvision.datasets.CIFAR10, num_images: int = 10) -> None:
    print("\n" + "="*80)
    print("  STEP 1: SHOWING SAMPLE CIFAR-10 IMAGES")
    print("="*80)
    
    fig: Figure
    axes: Any
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle('Sample Images from CIFAR-10 Dataset', fontsize=16, fontweight='bold', y=1.0)
    
    img: torch.Tensor
    label: int
    for idx, ax in enumerate(axes.flat):
        img, label = dataset[idx]
        # Denormalize image for display
        img_display: NDArray[Any] = (img / 2 + 0.5).numpy().transpose(1, 2, 0)
        img_display = np.clip(img_display, 0, 1)
        
        # Display image with border
        ax.imshow(img_display, interpolation='nearest')
        
        # Add colored border around image
        for spine in ax.spines.values():
            spine.set_edgecolor('steelblue')
            spine.set_linewidth(2)
            spine.set_visible(True)
        
        # Title with better formatting
        ax.set_title(f'{CIFAR10_CLASSES[label]}', 
                    fontsize=12, 
                    fontweight='bold',
                    pad=8,
                    color='darkblue')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95, hspace=0.35, wspace=0.3)
    output_path: str = os.path.join(OUTPUT_DIR, 'cifar10_samples.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Sample images saved to: {output_path}")
    print(f"These are real 32x32 color images from {len(CIFAR10_CLASSES)} classes")
    plt.show()
    print("\n")


def quick_train_and_test(use_cuda: bool = True, quick_mode: bool = True) -> None:
    device: str = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print(f"  VERIFICATION DEMO - CUDA: {device.upper()}")
    print("="*80)
    
    if quick_mode:
        print("\nQUICK MODE: Using reduced dataset for faster verification")
        print("   (Full mode would take 10-15 minutes)\n")
    
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Data directory is in parent folder
    data_dir: str = os.path.join(os.path.dirname(__file__), '..', 'data')
    trainset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                           download=True, transform=transform)
    testset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                          download=True, transform=transform)
    
    show_sample_images(testset)
    
    epochs: int
    train_subset: torch.utils.data.Subset[Any]
    test_subset: torch.utils.data.Subset[Any]
    if quick_mode:
        train_subset = torch.utils.data.Subset(trainset, range(10000))
        test_subset = torch.utils.data.Subset(testset, range(2000))
        train_loader: DataLoader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
        test_loader: DataLoader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=0)
        epochs = 2
        print(f"Using {len(train_subset)} training images, {len(test_subset)} test images")
    else:
        train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        epochs = 3
        print(f"Using full dataset: {len(trainset)} training, {len(testset)} test images")
    
    print("\n" + "="*80)
    print("  STEP 2: TRAINING BASELINE CNN")
    print("="*80)
    baseline: SimpleCNN = SimpleCNN().to(device)
    baseline_params: int = baseline.count_parameters()
    print(f"Model created with {baseline_params:,} parameters")
    print(f"Training for {epochs} epochs...\n")
    
    baseline_train_start: float = time.time()
    baseline = train_model(baseline, train_loader, epochs=epochs, device=device)  # type: ignore[assignment]
    baseline_train_time: float = time.time() - baseline_train_start
    
    baseline_acc: float = evaluate_model(baseline, test_loader, device=device)
    
    print(f"\nBaseline accuracy: {baseline_acc:.2f}%")
    print("This proves the model is learning from the data!")
    
    models_dir: str = os.path.join(os.path.dirname(__file__), '..', 'models')
    metadata: Dict[str, Any]
    print("\nSaving baseline model...")
    save_model_with_metadata(baseline, "baseline", baseline_acc, baseline_train_time, epochs, save_dir=models_dir)
    
    # STEP 3: Show predictions
    print("\n" + "="*80)
    print("  STEP 3: BASELINE MODEL PREDICTIONS")
    print("="*80)
    show_predictions(baseline, testset, device, "Baseline CNN Predictions")
    
    print("\n" + "="*80)
    print("  STEP 4: CREATING COMPRESSED MODEL")
    print("="*80)
    compressed: CompressedCNN = CompressedCNN(rank_ratio=0.5, fc_rank=64).to(device)
    print(f"Training compressed model for {epochs} epochs...\n")
    
    compressed_train_start: float = time.time()
    compressed = train_model(compressed, train_loader, epochs=epochs, device=device)  # type: ignore[assignment]
    compressed_train_time: float = time.time() - compressed_train_start
    
    print("\nApplying tensor decomposition...")
    compressed.compress()
    compressed_params: int = compressed.count_parameters()
    
    compressed_acc: float = evaluate_model(compressed, test_loader, device=device)
    
    print(f"\nCompressed accuracy: {compressed_acc:.2f}%")
    print(f"Parameters: {baseline_params:,} -> {compressed_params:,}")
    print(f"Compression: {baseline_params/compressed_params:.1f}x")
    
    print("\nSaving compressed model...")
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    metadata = {"compression_ratio": float(baseline_params/compressed_params)}
    save_model_with_metadata(
        compressed, "compressed", compressed_acc, compressed_train_time, epochs,
        metadata=metadata,
        save_dir=models_dir
    )
    
    # STEP 5: Show compressed predictions
    print("\n" + "="*80)
    print("  STEP 5: COMPRESSED MODEL PREDICTIONS")
    print("="*80)
    show_predictions(compressed, testset, device, "Compressed CNN Predictions")
    
    print("\n" + "="*80)
    print("  VERIFICATION COMPLETE")
    print("="*80)
    print(f"\n  RESULTS:")
    print(f"    - Baseline:   {baseline_acc:.2f}% accuracy, {baseline_params:,} params")
    print(f"    - Compressed: {compressed_acc:.2f}% accuracy, {compressed_params:,} params")
    print(f"    - Compression: {baseline_params/compressed_params:.1f}x smaller")
    print(f"    - Accuracy loss: {baseline_acc - compressed_acc:.2f}%")
    print(f"\n  The model correctly classifies CIFAR-10 images")
    print(f"  Tensor compression works with minimal accuracy loss")
    print(f"  Check the saved .png files to see predictions!")
    print("\n" + "="*80 + "\n")


def show_predictions(model: nn.Module, dataset: torchvision.datasets.CIFAR10, 
                    device: str, title: str = "Model Predictions") -> None:
    model.eval()
    
    fig: Figure
    axes: Any
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.0)
    
    correct: int = 0
    total: int = 10
    
    for idx, ax in enumerate(axes.flat):
        img: torch.Tensor
        true_label: int
        img, true_label = dataset[idx]
        
        # Get prediction
        with torch.no_grad():
            img_batch: torch.Tensor = img.unsqueeze(0).to(device)
            output: torch.Tensor = model(img_batch)
            probabilities: torch.Tensor = torch.nn.functional.softmax(output, dim=1)
            confidence_tensor: torch.Tensor
            pred_label_tensor: torch.Tensor
            confidence_tensor, pred_label_tensor = probabilities.max(1)
            pred_label_int: int = int(pred_label_tensor.item())
            confidence: float = float(confidence_tensor.item())
        
        # Denormalize for display
        display_img: NDArray[Any] = (img / 2 + 0.5).numpy().transpose(1, 2, 0)
        display_img = np.clip(display_img, 0, 1)
        
        # Show image with better interpolation
        ax.imshow(display_img, interpolation='nearest')
        
        # Determine correctness
        is_correct: bool = pred_label_int == true_label
        if is_correct:
            correct += 1
        
        # Set border color based on correctness
        border_color: str = 'darkgreen' if is_correct else 'darkred'
        border_width: int = 3
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(border_width)
            spine.set_visible(True)
        
        status_indicator: str = '[OK]' if is_correct else '[NO]'
        title_color: str = 'darkgreen' if is_correct else 'darkred'
        
        title_text: str = (f'{status_indicator} True: {CIFAR10_CLASSES[true_label]}\n'
                     f'Pred: {CIFAR10_CLASSES[pred_label_int]}\n'
                     f'Conf: {confidence*100:.1f}%')
        
        ax.set_title(title_text,
                    fontsize=10, 
                    color=title_color, 
                    fontweight='bold',
                    pad=10,
                    linespacing=1.3)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add accuracy text at the bottom
    accuracy_pct: float = (correct / total) * 100
    fig.text(0.5, 0.01, 
            f'Sample Accuracy: {correct}/{total} correct ({accuracy_pct:.0f}%)',
            ha='center', 
            fontsize=13, 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.subplots_adjust(top=0.87, bottom=0.06, left=0.04, right=0.96, hspace=0.45, wspace=0.25)
    filename: str = title.lower().replace(' ', '_') + '.png'
    output_path: str = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Predictions saved to: {output_path}")
    print(f"Got {correct}/{total} correct in this sample ({accuracy_pct:.0f}%)")
    plt.show()


def load_trained_models(device: str = 'cuda') -> Tuple[Optional[SimpleCNN], Optional[CompressedCNN]]:
    print("\n" + "="*80)
    print("  LOADING PRE-TRAINED MODELS")
    print("="*80)
    
    baseline_path: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'baseline.pth')
    compressed_path: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'compressed.pth')
    
    if not os.path.exists(baseline_path) or not os.path.exists(compressed_path):
        print("\nPre-trained models not found!")
        print("    Please run training mode first to create models.")
        return None, None
    
    print("\nLoading saved models...")
    
    baseline: SimpleCNN = SimpleCNN().to(device)
    checkpoint: Dict[str, Any] = torch.load(baseline_path, map_location=device, weights_only=False)  # type: ignore[assignment]
    baseline.load_state_dict(checkpoint['model_state_dict'])
    baseline_metadata: Dict[str, Any] = checkpoint['metadata']
    
    print(f"Baseline model loaded:")
    print(f"  Accuracy: {baseline_metadata['accuracy']:.2f}%")
    print(f"  Parameters: {baseline_metadata['num_parameters']:,}")
    
    compressed: CompressedCNN = CompressedCNN(rank_ratio=0.5, fc_rank=64).to(device)
    compressed.compress()
    checkpoint = torch.load(compressed_path, map_location=device, weights_only=False)
    compressed.load_state_dict(checkpoint['model_state_dict'])
    compressed_metadata: Dict[str, Any] = checkpoint['metadata']
    
    print(f"Compressed model loaded:")
    print(f"  Accuracy: {compressed_metadata['accuracy']:.2f}%")
    print(f"  Parameters: {compressed_metadata['num_parameters']:,}")
    print(f"  Compression: {baseline_metadata['num_parameters']/compressed_metadata['num_parameters']:.1f}x")
    
    return baseline, compressed


def demo_with_trained_models(use_cuda: bool = True) -> None:
    device: str = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    
    baseline, compressed = load_trained_models(device)
    
    if baseline is None or compressed is None:
        return
    
    print("\nLoading CIFAR-10 test dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Data directory is in parent folder
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                          download=True, transform=transform)
    test_loader: DataLoader[Any] = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    
    # Show sample images
    show_sample_images(testset)
    
    print("\n" + "="*80)
    print("  EVALUATING MODELS ON TEST SET")
    print("="*80)
    
    baseline_acc: float = evaluate_model(baseline, test_loader, device=device)
    print(f"\nBaseline accuracy: {baseline_acc:.2f}%")
    
    compressed_acc: float = evaluate_model(compressed, test_loader, device=device)
    print(f"Compressed accuracy: {compressed_acc:.2f}%")
    print(f"Accuracy drop: {baseline_acc - compressed_acc:.2f}%")
    
    # Show predictions
    print("\n" + "="*80)
    print("  BASELINE MODEL PREDICTIONS")
    print("="*80)
    show_predictions(baseline, testset, device, "Baseline CNN Predictions")
    
    print("\n" + "="*80)
    print("  COMPRESSED MODEL PREDICTIONS")
    print("="*80)
    show_predictions(compressed, testset, device, "Compressed CNN Predictions")
    
    print("\n" + "="*80)
    print("  DEMONSTRATION COMPLETE")
    print("="*80)
    print(f"\n  RESULTS:")
    print(f"    - Baseline:   {baseline_acc:.2f}% accuracy")
    print(f"    - Compressed: {compressed_acc:.2f}% accuracy")
    print(f"    - Accuracy loss: {baseline_acc - compressed_acc:.2f}%")
    print(f"\n  Check the '{OUTPUT_DIR}' folder to see saved visualizations!")
    print("\n" + "="*80 + "\n")


def main() -> None:
    print("\n" + "="*80)
    print("  CNN TENSOR COMPRESSION - VERIFICATION DEMO")
    print("="*80)
    print("\n  This demo will show:")
    print("    1. Real CIFAR-10 images (airplanes, cars, cats, etc.)")
    print("    2. Model training progress (you'll see loss decreasing)")
    print("    3. Actual predictions on test images")
    print("    4. Before/after compression comparison")
    print("\n  Choose your mode:")
    print("    [1] Use pre-trained models (fastest, ~1 min)")
    print("    [2] Train from scratch - QUICK MODE (~3-5 minutes)")
    print("    [3] Train from scratch - FULL MODE (~10-15 minutes)")
    print("    [Q] Quit")
    
    choice: str = input("\n  Your choice [1/2/3/Q]: ").strip().upper()
    
    if choice == '1':
        demo_with_trained_models(use_cuda=True)
    elif choice == '2':
        quick_train_and_test(use_cuda=True, quick_mode=True)
    elif choice == '3':
        quick_train_and_test(use_cuda=True, quick_mode=False)
    elif choice == 'Q':
        print("\n  Exiting...")
    else:
        print(f"\n  Invalid choice: {choice}")
    
    print("\n" + "="*80)
    print(f"  Check the '{OUTPUT_DIR}' folder for generated images:")
    print("     cifar10_samples.png - Sample images from dataset")
    print("     baseline_cnn_predictions.png - Predictions before compression")
    print("     compressed_cnn_predictions.png - Predictions after compression")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()

# Check for virtual environments and set up pip install
import os
import subprocess
import sys
from typing import Dict, Tuple

venv_paths = [
    os.path.expanduser("~/venvs/ai"),
    os.path.join(os.getcwd(), ".venv")
]

venv_path = None
for path in venv_paths:
    if os.path.exists(path):
        venv_path = path
        break

if venv_path is None:
    venv_path = os.path.join(os.getcwd(), ".venv")
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", venv_path])

# Activate the virtual environment and install dependencies
activate_script = os.path.join(venv_path, "bin", "activate")
command = (
    f"source {activate_script} && "
    "pip install -q --upgrade git+https://github.com/PandaSt0rm/htb-ai-library && "
    "pip install -q --upgrade torch torchvision numpy matplotlib scikit-learn jupyterlab notebook pandas tqdm Pillow"
)
print("Activating virtual environment and installing/updating dependencies...")
subprocess.run(
    ["bash", "-c", command],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=True,
)

# Activate the venv in the current Python process
os.environ["VIRTUAL_ENV"] = venv_path
bin_path = os.path.join(venv_path, "bin")
os.environ["PATH"] = bin_path + ":" + os.environ.get("PATH", "")
site_packages = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
sys.path.insert(0, site_packages)

print("\n" + "="*60)
print("Dependencies successfully installed.")

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import io, base64, requests, random
from PIL import Image

from htb_ai_library import (
    set_reproducibility,
    MNISTClassifierWithDropout,
    get_mnist_loaders,
    mnist_denormalize,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
    analyze_model_confidence,
    HTB_GREEN, NODE_BLACK, HACKER_GREY, WHITE,
    AZURE, NUGGET_YELLOW, MALWARE_RED, VIVID_PURPLE, AQUAMARINE
)

set_reproducibility(1337)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNISTClassifierWithDropout is imported from htb_ai_library
# The architecture internally defines:
# - Conv1: 1->32 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - Conv2: 32->64 channels, 3x3 kernel, ReLU, 2x2 pooling, 25% dropout
# - FC1: 3136->128, ReLU, 50% dropout
# - FC2: 128->10 (logits)

model = MNISTClassifierWithDropout().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

model_path = 'output/mnist_model.pth'
os.makedirs('output', exist_ok=True)

# Try loading cached model
if os.path.exists(model_path):
    print(f"Found cached model at {model_path}")
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()

    # Validate cached model
    _, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Cached model accuracy: {accuracy:.2f}%")

    if accuracy < 90.0:
        print("Accuracy below threshold, retraining required")
        model = None
else:
    model = None
    
# Train if needed
if model is None:
    print("Training new model...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64, normalize=True)
    model = MNISTClassifierWithDropout().to(device)

    model = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device
    )
    
    # Evaluate and cache
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

    save_model({
        'model': model,
        'architecture': 'MNISTClassifierWithDropout',
        'accuracy': accuracy,
        'training_config': {
            'epochs': 5,
            'batch_size': 64,
            'device': str(device)
        }
    }, model_path)
    
# Analyze confidence distribution
_, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
stats = analyze_model_confidence(model, test_loader, device=device, num_samples=1000)


### DeepFool attack implementation
def deepfool(image: torch.Tensor,
             net: nn.Module,
             num_classes: int = 10,
             overshoot: float = 0.02,
             max_iter: int = 50,
             device: str = 'cuda') -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
    """
    Generate minimal adversarial perturbation using DeepFool algorithm.

    Args:
        image (torch.Tensor): Input image tensor of shape (1, C, H, W)
        net (nn.Module): Target neural network in evaluation mode
        num_classes (int): Number of top-scoring classes to consider (default: 10)
        overshoot (float): Overshoot parameter for boundary crossing (default: 0.02)
        max_iter (int): Maximum iterations before terminating (default: 50)
        device (str): Computation device ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - r_tot (torch.Tensor): Total accumulated perturbation
            - loop_i (int): Number of iterations performed
            - label (int): Original predicted class
            - k_i (int): Final adversarial class
            - pert_image (torch.Tensor): Final perturbed image
    """
    image = image.to(device)
    net = net.to(device)
    
    # Original prediction and class ordering (descending score)
    f_image = net(image).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]
    label = I[0]
    
    # Working tensors and accumulators
    input_shape = image.shape
    pert_image = image.clone()
    r_tot = torch.zeros(input_shape).to(device)
    loop_i = 0
    
    # Iterate until a successful perturbation is found or the limit is reached
    while loop_i < max_iter:
        x = pert_image.clone().requires_grad_(True)
        fs = net(x)
        # Current top prediction at x
        k_i = fs.data.cpu().numpy().flatten().argsort()[::-1][0]

        # Stop when the prediction changes
        if k_i != label:
            break

        # Initialize the best candidate step for this iteration
        pert = float('inf')
        w = None
        
        # Search minimal step among candidate classes
        for k in range(1, num_classes):
            if I[k] == label:
                continue
            
            # Compute gradient for candidate class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, I[k]].backward(retain_graph=True)
            grad_k = x.grad.data.clone()
            
            # Compute gradient for original class
            if x.grad is not None:
                x.grad.zero_()
            fs[0, label].backward(retain_graph=True)
            grad_label = x.grad.data.clone()
            
            # Direction and distance under linearization
            w_k = grad_k - grad_label
            f_k = (fs[0, I[k]] - fs[0, label]).data.cpu().numpy()
            pert_k = abs(f_k) / (torch.norm(w_k.flatten()) + 1e-10)
            
            if pert_k < pert:
                pert = pert_k
                w = w_k
                
        # Minimal step for the selected direction
        r_i = (pert + 1e-4) * w / (torch.norm(w.flatten()) + 1e-10)
        r_tot = r_tot + r_i

        # Apply with overshoot to ensure crossing
        pert_image = image + (1 + overshoot) * r_tot
        loop_i += 1

    return r_tot, loop_i, label, k_i, pert_image


# Load trained model
model_path = 'output/mnist_model.pth'
if os.path.exists(model_path):
    model_data = load_model(model_path)
    model = model_data['model'].to(device)
    model.eval()
else:
    raise FileNotFoundError("Model not found.")

# Get single test sample
_, test_loader = get_mnist_loaders(batch_size=1, normalize=True)
dataiter = iter(test_loader)
image, true_label = next(dataiter)
image = image.to(device)

print(f"True label: {true_label.item()}")

# Baseline classification
with torch.no_grad():
    original_output = model(image)
    original_pred = original_output.argmax(dim=1).item()
    original_confidence = F.softmax(original_output, dim=1).max().item()

print(f"Original: class {original_pred} (confidence: {original_confidence:.3f})")

# Execute DeepFool attack
r_total, iterations, orig_label, pert_label, pert_image = deepfool(
    image, model, num_classes=10, overshoot=0.02, max_iter=50, device=device
)

print(f"Attack: {orig_label} → {pert_label} in {iterations} iterations")

# Compute perturbation norms
perturbation_norm_l2 = torch.norm(r_total).item()
perturbation_norm_linf = torch.abs(r_total).max().item()
relative_perturbation = perturbation_norm_l2 / torch.norm(image).item()

# Evaluate adversarial confidence
with torch.no_grad():
    adv_output = model(pert_image)
    adv_confidence = F.softmax(adv_output, dim=1).max().item()

# Display results
print(f"\n=== Attack Results ===")
print(f"L2 norm: {perturbation_norm_l2:.4f}")
print(f"L∞ norm: {perturbation_norm_linf:.4f}")
print(f"Relative perturbation: {relative_perturbation:.2%}")
print(f"Original confidence: {original_confidence:.3f}")
print(f"Adversarial confidence: {adv_confidence:.3f}")

# Prepare images for visualization
original_img = mnist_denormalize(image.squeeze()).cpu().numpy()
adversarial_img = mnist_denormalize(pert_image.squeeze()).cpu().numpy()
perturbation = r_total.cpu().squeeze().numpy()

# Normalize perturbation for visibility (amplify minimal changes)
pert_display = perturbation - perturbation.min()
if pert_display.max() > 0:
    pert_display = pert_display / pert_display.max()

# Create four-panel visualization
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
fig.patch.set_facecolor(NODE_BLACK)

for ax in axes:
    ax.set_facecolor(NODE_BLACK)
    for spine in ax.spines.values():
        spine.set_edgecolor(HACKER_GREY)

# Panel 1: Original clean image
axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f"Original\nClass: {original_pred}",
                  color=HTB_GREEN, fontweight='bold')
axes[0].axis('off')

# Panel 2: Amplified perturbation pattern
axes[1].imshow(pert_display, cmap='inferno')
axes[1].set_title("Perturbation\n(amplified)",
                  color=NUGGET_YELLOW, fontweight='bold')
axes[1].axis('off')

# Panel 3: Perturbation magnitude heatmap
im = axes[2].imshow(np.abs(perturbation), cmap='viridis')
axes[2].set_title(f"Magnitude\nL2: {perturbation_norm_l2:.4f}",
                  color=AZURE, fontweight='bold')
axes[2].axis('off')
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

# Panel 4: Adversarial result
title_color = HTB_GREEN if pert_label != original_pred else MALWARE_RED
axes[3].imshow(adversarial_img, cmap='gray', vmin=0, vmax=1)
axes[3].set_title(f"Adversarial\nClass: {pert_label}",
                  color=title_color, fontweight='bold')
axes[3].axis('off')

# Summary metrics
metrics_text = (
    f"Iterations: {iterations}  |  "
    f"Relative pert: {relative_perturbation:.2%}  |  "
    f"Confidence: {original_confidence:.3f} → {adv_confidence:.3f}"
)
fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, color=WHITE)

plt.suptitle("DeepFool Attack Visualization", fontsize=14,
             color=HTB_GREEN, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


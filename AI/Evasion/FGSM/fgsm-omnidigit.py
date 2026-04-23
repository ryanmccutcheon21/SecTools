# Check for virtual environments and set up pip install
import os
import subprocess
import sys
from typing import Dict

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

import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import io, base64, requests
from PIL import Image

# Import common utilities from HTB Evasion Library
from htb_ai_library import (
    set_reproducibility,
    SimpleCNN,
    get_mnist_loaders,
    mnist_denormalize,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
    analyze_model_confidence,
    HTB_GREEN, MALWARE_RED, AZURE, NUGGET_YELLOW, HACKER_GREY, WHITE, NODE_BLACK,
)


base64_images = [
"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"
]


# API Helpers and Challenge Fetching

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

def x01_from_b64_png(b64: str) -> np.ndarray:
    """Convert base64 PNG to [0,1] numpy array.

    Args:
        b64: Base64 encoded PNG string

    Returns:
        np.ndarray: Image as (28, 28) array in [0,1] range
    """
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    """Convert [0,1] array to base64 PNG.

    Args:
        x2d: Image array in [0,1] range

    Returns:
        str: Base64 encoded PNG string
    """
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def normalize_mnist_tensor(x: Tensor) -> Tensor:
    return (x - 0.1307) / 0.3081


def tensor_from_b64_png(b64: str, device: torch.device | None = None) -> Tensor:
    x01 = x01_from_b64_png(b64)
    x = torch.from_numpy(x01).unsqueeze(0).unsqueeze(0).to(torch.float32)
    x = normalize_mnist_tensor(x)
    if device is not None:
        x = x.to(device)
    return x


def tensor_from_np_arr(x01: np.ndarray, device: torch.device | None = None) -> Tensor:
    x = torch.from_numpy(x01.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    x = normalize_mnist_tensor(x)
    if device is not None:
        x = x.to(device)
    return x


def predict_digit(model: nn.Module, image: Tensor) -> int:
    model.eval()
    with torch.no_grad():
        return int(model(image).argmax(dim=1).item())


def split_base64_composite(b64: str, tile_count: int) -> list[np.ndarray]:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    width, height = img.size
    if height != 28 or width != 28 * tile_count:
        raise ValueError(f"Expected composite size {28*tile_count}x28")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return [arr[:, i * 28:(i + 1) * 28] for i in range(tile_count)]


def compose_horizontal_png(tiles: list[np.ndarray]) -> str:
    if any(tile.shape != (28, 28) for tile in tiles):
        raise ValueError("All tiles must be 28x28")
    composite = np.hstack(tiles)
    return b64_png_from_x01(composite)


def predict_sequence_from_composite(model: nn.Module, b64: str, device: torch.device, tile_count: int = 6) -> list[int]:
    tiles = split_base64_composite(b64, tile_count)
    return [predict_digit(model, tensor_from_np_arr(tile, device=device)) for tile in tiles]


def split_128x28_png_to_tiles(b64: str) -> list[np.ndarray]:
    return split_base64_composite(b64, tile_count=4)


def find_best_targeted_flip(model: nn.Module,
                             bank_tensors: list[Tensor],
                             target_digit: int,
                             used_indices: set[int],
                             device: torch.device,
                             epsilons: list[float] | None = None) -> tuple[int, Tensor, float]:
    if epsilons is None:
        epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.3]
    best = None
    for idx, tensor in enumerate(bank_tensors):
        if idx in used_indices:
            continue
        tensor = tensor.to(device)
        with torch.no_grad():
            src_pred = int(model(tensor).argmax(dim=1).item())
        if src_pred == target_digit:
            continue
        for eps in epsilons:
            adv = fgsm_attack(model, tensor, torch.tensor([target_digit], device=device), epsilon=eps, targeted=True)
            with torch.no_grad():
                tgt_pred = int(model(adv).argmax(dim=1).item())
            if tgt_pred == target_digit:
                perturbation = (adv - tensor).abs().amax().item()
                if best is None or eps < best[2] or (eps == best[2] and perturbation < best[3]):
                    best = (idx, adv.cpu(), eps, perturbation)
                break
    if best is None:
        raise RuntimeError(f"No candidate from the bank could be flipped to {target_digit}")
    return best[0], best[1], best[2]


def build_target_sequence_composite(model: nn.Module,
                                    bank_b64_list: list[str],
                                    target_sequence: list[int],
                                    device: torch.device) -> tuple[str, list[int], list[int]]:
    if len(bank_b64_list) < len(target_sequence):
        raise ValueError("Bank must contain at least as many images as target digits")
    bank_tensors = [tensor_from_b64_png(b64, device=device) for b64 in bank_b64_list]
    used_indices: set[int] = set()
    adv_tiles: list[np.ndarray] = []
    selected_indices: list[int] = []

    for target in target_sequence:
        index, adv_tensor, eps = find_best_targeted_flip(
            model,
            bank_tensors,
            target_digit=target,
            used_indices=used_indices,
            device=device,
        )
        used_indices.add(index)
        selected_indices.append(index)
        adv_tiles.append(np.clip((adv_tensor.squeeze(0).squeeze(0).cpu().numpy() * 0.3081) + 0.1307, 0.0, 1.0))
        print(f"Selected bank image {index} -> target {target} with eps={eps:.2f}")

    composite_b64 = compose_horizontal_png(adv_tiles)
    predicted_sequence = predict_sequence_from_composite(model, composite_b64, device=device, tile_count=len(target_sequence))
    return composite_b64, predicted_sequence, selected_indices


def classify_128x28_example(model: nn.Module, example_b64: str, device: torch.device) -> list[int]:
    example_b64 = "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"
    tiles = split_128x28_png_to_tiles(example_b64)
    print("Predicted digits:", ",".join(str(predict_digit(model, tensor_from_np_arr(tile, device=device))) for tile in tiles))
    return [predict_digit(model, tensor_from_np_arr(tile, device=device)) for tile in tiles]


def print_sequence_output(sequence: list[int]) -> None:
    print("Sequence prediction:", ",".join(str(x) for x in sequence))

def linf(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L_inf distance between two arrays.

    Args:
        a, b: Arrays to compare

    Returns:
        float: Maximum absolute difference
    """
    return float(np.max(np.abs(a - b)))

def _style_axes(ax: plt.Axes) -> None:
    """Apply Hack The Box dark theme to an axes instance.

    Args:
        ax: Matplotlib axes to style
    """
    ax.set_facecolor(NODE_BLACK)
    ax.tick_params(colors=HACKER_GREY)
    for spine in ax.spines.values():
        spine.set_color(HACKER_GREY)
    ax.grid(True, color=HACKER_GREY, linestyle="--", alpha=0.25)

# Configure reproducibility
set_reproducibility(1337)

# Configure computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for cached model and data, and download if not present
model_path = 'output/simplecnn_model.pth'
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
    train_loader, test_loader = get_mnist_loaders(batch_size=128, normalize=True)
    model = SimpleCNN().to(device)

    trained_model = train_model(
        model, train_loader, test_loader,
        epochs=1, device=device
    )
    
    # Evaluate and cache
    accuracy = evaluate_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")

    save_model({
        'model': model,
        'architecture': 'SimpleCNN',
        'accuracy': accuracy,
        'training_config': {
            'epochs': 5,
            'batch_size': 128,
            'device': str(device)
        }
    }, model_path)


# Analyze confidence distribution
_, test_loader = get_mnist_loaders(batch_size=100, normalize=True)
stats = analyze_model_confidence(model, test_loader, device=device, num_samples=1000)

# Computing Loss without Side Effects
def _forward_and_loss(model: nn.Module, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Forward pass and cross-entropy loss without side effects.

    Args:
        model: Neural network classifier
        x: Input images tensor
        y: Target labels tensor

    Returns:
        tuple[Tensor, Tensor]: Model logits and scalar loss value
    """
    if getattr(model, "training", False):
        raise RuntimeError("Expected model.eval() for attack computations to avoid BN/Dropout state updates")
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    return logits, loss

# Gradient Computation
def _input_gradient(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """Return gradient of loss with respect to input tensor x.

    Args:
        model: Neural network in evaluation mode
        x: Input images to compute gradients for
        y: True labels for loss computation

    Returns:
        Tensor: Gradient tensor with same shape as x
    """
    x_req = x.clone().detach().requires_grad_(True)
    _, loss = _forward_and_loss(model, x_req, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return x_req.grad.detach()


# FGSM Attack Implementation - Where editing takes place for attack logic
print("\n" + "="*60)
print("FGSM Attack Executing...") 
def fgsm_attack(model: nn.Module,
                images: Tensor,
                labels: Tensor,
                epsilon: float,
                targeted: bool = False) -> Tensor:

    # Valid normalized range for MNIST
    MNIST_NORM_MIN = (0.0 - 0.1307) / 0.3081
    MNIST_NORM_MAX = (1.0 - 0.1307) / 0.3081

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not images.is_floating_point():
        raise ValueError("images must be floating point tensors")

    grad = _input_gradient(model, images, labels)
    step_dir = -1.0 if targeted else 1.0
    x_adv = images + step_dir * epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, MNIST_NORM_MIN, MNIST_NORM_MAX)
    return x_adv.detach()

# Testing the Attack
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
# Epsilon in normalized space (≈0.25 in pixel space)
epsilon = 0.8
with torch.no_grad():
    clean_pred = model(images).argmax(dim=1)

x_adv = fgsm_attack(model, images, labels, epsilon)
with torch.no_grad():
    adv_pred = model(x_adv).argmax(dim=1)

originally_correct = (clean_pred == labels)
flipped = (adv_pred != labels) & originally_correct
success = flipped.sum().item() / max(int(originally_correct.sum().item()), 1)
print("\n" + "="*60)
print(f"FGSM flips (first batch): {success:.2%}")

def evaluate_attack(model: nn.Module,
                   clean_images: Tensor,
                   adversarial_images: Tensor,
                   true_labels: Tensor) -> Dict[str, float]:
    """Compute accuracy, success rate, confidence shift, and norms.

    Args:
        model: Evaluated classifier in evaluation mode
        clean_images: Clean inputs in the model's expected domain (e.g., normalized MNIST)
        adversarial_images: Adversarial counterparts in the same domain as `clean_images`
        true_labels: Ground-truth labels

    Returns:
        Dict[str, float]: Aggregated metrics summarizing attack impact
    """
    model.eval()
    with torch.no_grad():
        clean_logits = model(clean_images)
        adv_logits = model(adversarial_images)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_probs = F.softmax(adv_logits, dim=1)

        clean_pred = clean_logits.argmax(dim=1)
        adv_pred = adv_logits.argmax(dim=1)
        clean_correct = (clean_pred == true_labels)
        adv_correct = (adv_pred == true_labels)

        originally_correct = clean_correct
        flipped = (~adv_correct) & originally_correct
        conf_clean = clean_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        conf_adv = adv_probs.gather(1, true_labels.view(-1, 1)).squeeze(1)
        l2 = (adversarial_images - clean_images).view(clean_images.size(0), -1).norm(p=2, dim=1)
        linf = (adversarial_images - clean_images).abs().amax()

        return {
            "clean_accuracy": clean_correct.float().mean().item(),
            "adversarial_accuracy": adv_correct.float().mean().item(),
            # Success rate among originally correct samples only
            "attack_success_rate": (
                flipped.float().sum() / originally_correct.float().sum().clamp_min(1.0)
            ).item(),
            "avg_clean_confidence": conf_clean.mean().item(),
            "avg_adv_confidence": conf_adv.mean().item(),
            "avg_confidence_drop": (conf_clean - conf_adv).mean().item(),
            "avg_l2_perturbation": l2.mean().item(),
            "max_linf_perturbation": linf.item(),
        }
        
# Assume images, labels, x_adv from the Core Implementation section
metrics = evaluate_attack(model, images, x_adv, labels)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print("\n" + "="*60)
print("Building target sequence composite from base64 bank...")
target_sequence = [1, 3, 3, 7, 1, 7]
try:
    composite_b64, predicted_sequence, selected_indices = build_target_sequence_composite(
        model, base64_images, target_sequence, device=device
    )
    print_sequence_output(predicted_sequence)
    with open('final_sequence.png', 'wb') as f:
        f.write(base64.b64decode(composite_b64))
    print(f"Final composite saved to final_sequence.png")
    print(f"Selected bank tile indices: {selected_indices}")
except Exception as e:
    print(f"Failed to build target composite: {e}")

# Example helper usage for a 128x28 test image
# example_b64 = '...'  # base64 of 128x28 PNG
# example_sequence = classify_128x28_example(model, example_b64, device=device)
# print_sequence_output(example_sequence)
    
#
# Visualization
#
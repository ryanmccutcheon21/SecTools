import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities from HTB Evasion Library
from htb_ai_library.utils import (
    set_reproducibility,
    save_model,
    load_model,
    HTB_GREEN,
    NODE_BLACK,
    HACKER_GREY,
    WHITE,
    AZURE,
    NUGGET_YELLOW,
    MALWARE_RED,
    VIVID_PURPLE,
    AQUAMARINE,
)
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.training import train_model, evaluate_accuracy
from htb_ai_library.visualization import use_htb_style

# Apply HTB theme globally to all plots
use_htb_style()

# Set reproducibility
set_reproducibility(1337)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Get data loaders using library function
train_loader, test_loader = get_mnist_loaders(batch_size=128)
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

# Create output directory for saving models and results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Define model checkpoint path in output directory
model_path = output_dir / "mnist_target.pth"

# Initialize model using MNISTClassifierWithDropout from library
model = MNISTClassifierWithDropout(num_classes=10).to(device)

# Check if trained model exists, otherwise train from scratch
if model_path.exists():
    print(f"\nLoading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print(f"\nNo existing model found. Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, device=device)
    print(f"Saving trained model to {model_path}")
    save_model(model, model_path)

# Evaluate the trained model
accuracy = evaluate_accuracy(model, test_loader, device)
print(f"\nTest accuracy: {accuracy:.2f}%")

def compute_distances(adv_images, original_images, beta):
    """
    Compute L1, L2, and elastic-net distances.

    Returns all three distance metrics used in optimization
    and decision-making.

    Parameters:
        adv_images (torch.Tensor): Adversarial images (batch_size, C, H, W)
        original_images (torch.Tensor): Original images (batch_size, C, H, W)
        beta (float): Weight for L1 in elastic-net distance

    Returns:
        tuple: (l1_dist, l2_dist, elastic_dist) each shape (batch_size,)
    """
    l1_dist = torch.sum(torch.abs(adv_images - original_images), dim=(1, 2, 3))
    l2_dist = torch.sum((adv_images - original_images) ** 2, dim=(1, 2, 3))
    elastic_dist = l2_dist + beta * l1_dist

    return l1_dist, l2_dist, elastic_dist

def compute_adversarial_loss(logits, labels_onehot, confidence, targeted=False):
    """
    Compute margin-based adversarial loss.

    Uses C&W formulation: encourage misclassification with
    confidence margin. Loss becomes zero once margin achieved.

    Parameters:
        logits (torch.Tensor): Model outputs before softmax (batch_size, num_classes)
        labels_onehot (torch.Tensor): One-hot encoded labels (batch_size, num_classes)
        confidence (float): Confidence margin kappa
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Adversarial loss per example (batch_size,)
    """
    # Extract scores
    real = torch.sum(labels_onehot * logits, dim=1)
    other = torch.max((1 - labels_onehot) * logits - labels_onehot * 10000, dim=1)[0]

    # Compute margin loss
    if targeted:
        # For targeted attacks: want target to exceed other classes by the margin
        loss = torch.clamp(other - real + confidence, min=0)
    else:
        # For untargeted attacks: want real class to be exceeded
        loss = torch.clamp(real - other + confidence, min=0)

    return loss

def compute_total_loss(
    adv_images,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    confidence,
    targeted=False,
):
    """
    Combine smooth components: c * adversarial + squared L2 (L1 via proximal operator).

    The constant c balances misclassification vs distortion.
    Beta controls L1 vs L2 trade-off (sparsity vs smoothness).

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants c per example
        model (nn.Module): Target model
        beta (float): L1 weight in elastic-net distance
        confidence (float): Margin for misclassification
        targeted (bool): Whether this is a targeted attack

    Returns:
        tuple: (total_loss, adversarial_loss, distances)
    """
    # Get model predictions
    logits = model(adv_images)

    # Compute adversarial loss
    adversarial_loss = compute_adversarial_loss(
        logits, labels_onehot, confidence, targeted
    )

    # Compute distances
    l1_dist, l2_dist, elastic_dist = compute_distances(
        adv_images, original_images, beta
    )

    # Combine: c * adversarial_loss + L2_distance
    # Note: L1 is handled by FISTA's proximal operator, not in this gradient
    total_loss = const * adversarial_loss + l2_dist

    return total_loss, adversarial_loss, (l1_dist, l2_dist, elastic_dist)

def compute_fista_momentum(iteration):
    """
    Calculate FISTA momentum parameter for iteration k.

    Uses Nesterov acceleration: k/(k+3)
    Early iterations have small momentum, later ones accelerate.

    Parameters:
        iteration (int): Current FISTA iteration number

    Returns:
        float: Momentum coefficient in [0, 1)
    """
    return iteration / (iteration + 3.0)

# Test momentum progression
print("FISTA Momentum Progression:")
print("=" * 40)
test_iterations = [1, 5, 10, 50, 100, 500, 1000]

for k in test_iterations:
    momentum = compute_fista_momentum(k)
    print(f"Iteration {k:4d}: momentum = {momentum:.4f}")
    
def apply_shrinkage_thresholding(y, original_images, threshold, clip_min=0.0, clip_max=1.0):
    """
    Apply soft thresholding to create sparse perturbations.

    Values with magnitude at most threshold are zeroed (sparsity).
    Values exceeding the threshold are reduced by the threshold but remain non-zero.

    Parameters:
        y (torch.Tensor): Candidate adversarial images after gradient step
        original_images (torch.Tensor): Original unperturbed images
        threshold (float): Soft thresholding parameter (higher = sparser)
        clip_min (float): Minimum valid pixel value (default 0.0)
        clip_max (float): Maximum valid pixel value (default 1.0)

    Returns:
        torch.Tensor: Images after soft thresholding
    """
    # Compute the difference from original
    diff = y - original_images

    # Apply soft thresholding
    # Values within threshold are zeroed (sparsity!)
    # Values above threshold get reduced by threshold
    shrink_positive = torch.clamp(y - threshold, min=clip_min, max=clip_max)
    shrink_negative = torch.clamp(y + threshold, min=clip_min, max=clip_max)

    # Three-way decision: positive, zero, or negative
    cond_positive = (diff > threshold).float()
    cond_zero = (torch.abs(diff) <= threshold).float()
    cond_negative = (diff < -threshold).float()

    # Combine using conditions
    result = (
        cond_positive * shrink_positive
        + cond_zero * original_images
        + cond_negative * shrink_negative
    )

    return result

# Create synthetic perturbation for visualization
print("\nTesting shrinkage-thresholding operation...")

# Generate a 5x5 synthetic perturbation pattern
perturbation_pattern = torch.tensor([
    [-0.15, -0.08, -0.02,  0.03,  0.12],
    [-0.10, -0.05,  0.00,  0.06,  0.18],
    [-0.05,  0.00,  0.05,  0.10,  0.20],
    [ 0.00,  0.05,  0.08,  0.15,  0.25],
    [ 0.05,  0.08,  0.12,  0.20,  0.30]
], device=device)

# Assume original pixels are 0.5 (mid-gray)
original_values = torch.ones_like(perturbation_pattern) * 0.5
perturbed_values = original_values + perturbation_pattern

# Apply shrinkage with beta=0.1
beta_test = 0.1
thresholded = apply_shrinkage_thresholding(
    perturbed_values,
    original_values,
    beta_test,
    clip_min=0.0,
    clip_max=1.0
)

# Compute resulting perturbations
resulting_perturbation = thresholded - original_values

print(f"\nSoft Thresholding Results (beta={beta_test}):")
print("=" * 50)
print("\nOriginal perturbation pattern:")
print(perturbation_pattern.cpu().numpy())
print("\nAfter soft thresholding:")
print(resulting_perturbation.cpu().numpy())

# Count sparsity
original_nonzero = (perturbation_pattern.abs() > 1e-6).sum().item()
thresholded_nonzero = (resulting_perturbation.abs() > 1e-6).sum().item()
sparsity_gained = original_nonzero - thresholded_nonzero

print(f"\nSparsity Analysis:")
print(f"  Original non-zero elements: {original_nonzero}/25")
print(f"  After thresholding: {thresholded_nonzero}/25")
print(f"  Elements zeroed out: {sparsity_gained}")
print(f"  Sparsity achieved: {(25-thresholded_nonzero)/25*100:.1f}%")

def fista_step(
    adv_images,
    y_momentum,
    original_images,
    labels_onehot,
    const,
    model,
    beta,
    learning_rate,
    confidence,
    iteration,
    targeted=False,
    clip_min=0.0,
    clip_max=1.0,
):
    """
    Perform one complete FISTA iteration.

    Combines gradient computation, shrinkage-thresholding,
    and momentum update into a single optimization step.

    Parameters:
        adv_images (torch.Tensor): Current adversarial images
        y_momentum (torch.Tensor): Momentum point for gradient evaluation
        original_images (torch.Tensor): Original clean images
        labels_onehot (torch.Tensor): One-hot encoded labels
        const (torch.Tensor): Trade-off constants per example
        model (nn.Module): Target model
        beta (float): L1 weight parameter
        learning_rate (float): FISTA step size
        confidence (float): Margin for misclassification
        iteration (int): Current FISTA iteration number for momentum calculation
        targeted (bool): Whether this is a targeted attack
        clip_min (float): Minimum valid pixel value
        clip_max (float): Maximum valid pixel value

    Returns:
        tuple: (new_adv_images, new_y_momentum, loss_value, distances)
    """
    # Ensure y_momentum requires gradients for backprop
    y_momentum = y_momentum.detach().requires_grad_(True)

    # Compute loss at momentum point
    total_loss, adversarial_loss, distances = compute_total_loss(
        y_momentum,
        original_images,
        labels_onehot,
        const,
        model,
        beta,
        confidence,
        targeted,
    )

    # Compute gradient of total loss w.r.t. momentum point
    total_loss_summed = total_loss.sum()
    total_loss_summed.backward()
    grad = y_momentum.grad

    # Gradient step: move in negative gradient direction
    y_new = y_momentum - learning_rate * grad

    # Apply shrinkage-thresholding (proximal operator for L1)
    adv_new = apply_shrinkage_thresholding(
        y_new, original_images, learning_rate * beta, clip_min, clip_max
    )

    # Compute momentum coefficient
    momentum_coef = compute_fista_momentum(iteration)

    # Update momentum point for next iteration
    y_new_momentum = adv_new + momentum_coef * (adv_new - adv_images)

    return adv_new, y_new_momentum, total_loss_summed.item(), distances

def check_attack_success(adv_images, labels, model, targeted=False):
    """
    Verify whether adversarial examples achieve misclassification.

    Compares model predictions on adversarial images to target labels.

    Parameters:
        adv_images (torch.Tensor): Adversarial images to evaluate
        labels (torch.Tensor): True labels (or target labels if targeted)
        model (nn.Module): Target model
        targeted (bool): Whether this is a targeted attack

    Returns:
        torch.Tensor: Boolean mask indicating successful attacks (batch_size,)
    """
    with torch.no_grad():
        outputs = model(adv_images)
        predictions = outputs.argmax(dim=1)

        if targeted:
            # Success = prediction matches target
            success = predictions.eq(labels)
        else:
            # Success = prediction differs from true label
            success = predictions.ne(labels)

    return success

def update_binary_search_bounds(lower_bound, upper_bound, const, success_mask):
    """
    Update binary search bounds based on attack success.

    Successful attacks lower upper bound (c was sufficient).
    Failed attacks raise lower bound (c was insufficient).

    Parameters:
        lower_bound (torch.Tensor): Lower bounds on c per example
        upper_bound (torch.Tensor): Upper bounds on c per example
        const (torch.Tensor): Current c values per example
        success_mask (torch.Tensor): Boolean mask of successful attacks

    Returns:
        tuple: (new_lower_bound, new_upper_bound, new_const)
    """
    # Process each example individually
    for i in range(len(success_mask)):
        if success_mask[i]:
            # Success: try smaller c
            upper_bound[i] = min(upper_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
        else:
            # Failure: need larger c
            lower_bound[i] = max(lower_bound[i], const[i])
            if upper_bound[i] < 1e10:
                const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                const[i] *= 10  # Exponential increase for persistent failures

    return lower_bound, upper_bound, const

# ---------- Attack configuration ----------
# Attack hyperparameters
config = {
    "beta": 0.01,  # L1 vs L2 trade-off (higher = sparser)
    "confidence": 0,  # Margin for misclassification
    "learning_rate": 0.01,  # FISTA step size
    "max_iterations": 1000,  # FISTA iterations per binary search
    "binary_search_steps": 5,  # Number of binary search iterations
    "initial_const": 0.001,  # Starting trade-off constant
    "clip_min": 0.0,  # Minimum pixel value
    "clip_max": 1.0,  # Maximum pixel value
}

print("\nAttack Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
    
# ---------- Sample Selection and Initialization ----------
print("\nSelecting correctly classified samples for attack...")
model.eval()
num_samples = 20

for data, targets in test_loader:
    data, targets = data.to(device), targets.to(device)
    outputs = model(data)
    predictions = outputs.argmax(dim=1)

    # Select correctly classified samples
    correct_mask = predictions.eq(targets)
    attack_data = data[correct_mask][:num_samples]
    attack_targets = targets[correct_mask][:num_samples]

    if len(attack_data) >= num_samples:
        print(f"Selected {len(attack_data)} samples")
        break
    
print("\nPerforming ElasticNet attack...")
print("This may take several minutes due to iterative optimization...\n")

batch_size = len(attack_data)
original_images = attack_data.clone()

# Convert labels to one-hot encoding
labels_onehot = torch.zeros(batch_size, 10).to(device)
labels_onehot.scatter_(1, attack_targets.unsqueeze(1), 1)

# Initialize binary search bounds
lower_bound = torch.zeros(batch_size).to(device)
upper_bound = torch.ones(batch_size).to(device) * 1e10
const = torch.ones(batch_size).to(device) * config["initial_const"]

# Track best adversarial examples
best_adv = original_images.clone()
best_l2 = torch.ones(batch_size).to(device) * 1e10

# Binary search over trade-off constant c
for binary_step in range(config["binary_search_steps"]):
    print(f"Binary search step {binary_step + 1}/{config['binary_search_steps']}")

    # Initialize FISTA from original images
    adv_images = original_images.clone().detach()
    y_momentum = adv_images.clone()

    # FISTA optimization loop
    for iteration in range(config["max_iterations"]):
        # Perform one FISTA step
        adv_images, y_momentum, loss, distances = fista_step(
            adv_images,
            y_momentum,
            original_images,
            labels_onehot,
            const,
            model,
            config["beta"],
            config["learning_rate"],
            config["confidence"],
            iteration,
            targeted=False,
            clip_min=config["clip_min"],
            clip_max=config["clip_max"],
        )

    # Check which examples successfully fooled the model
    success_mask = check_attack_success(
        adv_images, attack_targets, model, targeted=False
    )

    # Update best adversarial examples
    l1_dist, l2_dist, elastic_dist = compute_distances(
        adv_images, original_images, config["beta"]
    )

    for i in range(batch_size):
        if success_mask[i] and l2_dist[i] < best_l2[i]:
            best_adv[i] = adv_images[i]
            best_l2[i] = l2_dist[i]

    # Update binary search bounds
    lower_bound, upper_bound, const = update_binary_search_bounds(
        lower_bound, upper_bound, const, success_mask
    )

    # Progress reporting
    num_success = success_mask.sum().item()
    print(f"  Successfully generated {num_success}/{batch_size} adversarial examples\n")
    
# ---------- Computing & Analyzing Results ----------
# Evaluate final adversarial examples
with torch.no_grad():
    adv_outputs = model(best_adv)
    adv_predictions = adv_outputs.argmax(dim=1)

    # Calculate success rate
    final_success = adv_predictions.ne(attack_targets)
    success_rate = final_success.float().mean().item() * 100

    # Calculate final distortions
    l1_dist, l2_dist, elastic_dist = compute_distances(
        best_adv, original_images, config["beta"]
    )
    linf_dist = torch.max(
        torch.abs(best_adv - original_images).view(batch_size, -1), dim=1
    )[0]

# Display results
print("=" * 60)
print("ElasticNet Attack Results:")
print("=" * 60)
print(f"Success Rate: {success_rate:.2f}% ({final_success.sum()}/{batch_size})")
print(f"Average L1 Distortion: {l1_dist.mean().item():.4f}")
print(f"Average Squared L2 Distortion: {l2_dist.mean().item():.4f}")
print(f"Average L∞ Distortion: {linf_dist.mean().item():.4f}")
print(f"Average Elastic Distortion: {elastic_dist.mean().item():.4f}")
print("=" * 60)

# ---------- Attack Process Visualization ----------
print("\nGenerating visualizations...")

# Visualization: Attack process (original → perturbation → adversarial)
print("Creating attack process visualization...")

fig, axes = plt.subplots(3, 10, figsize=(20, 6))

# Show first 10 examples
num_display = min(10, batch_size)

for i in range(num_display):
    # Original image
    orig_img = original_images[i].detach().cpu().squeeze()
    axes[0, i].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"True: {attack_targets[i].item()}", fontsize=10)

    # Perturbation (amplified for visibility)
    pert = (best_adv[i] - original_images[i]).detach().cpu().squeeze()
    pert_display = pert * 10  # Amplify by 10x for visibility
    axes[1, i].imshow(pert_display, cmap="seismic", vmin=-1, vmax=1)
    axes[1, i].axis("off")
    axes[1, i].set_title("Perturbation", fontsize=10)

    # Adversarial image
    adv_img = best_adv[i].detach().cpu().squeeze()
    axes[2, i].imshow(adv_img, cmap="gray", vmin=0, vmax=1)
    axes[2, i].axis("off")
    axes[2, i].set_title(f"Pred: {adv_predictions[i].item()}", fontsize=10, color=MALWARE_RED)

# Row labels
fig.text(0.02, 0.80, "Original", rotation=90, fontsize=14, weight="bold", ha="center", va="center")
fig.text(0.02, 0.50, "Perturbation\n(10× amplified)", rotation=90, fontsize=14, weight="bold", ha="center", va="center")
fig.text(0.02, 0.20, "Adversarial", rotation=90, fontsize=14, weight="bold", ha="center", va="center")

plt.tight_layout(rect=[0.03, 0, 1, 1])
plt.savefig(output_dir / "ead_attack_process.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_attack_process.png")

# Visualization 2: Distortion distributions for L1, L2, L∞, Elastic
print("Creating distortion analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Calculate distortion metrics
l1_values = l1_dist.detach().cpu().numpy()
l2_values = l2_dist.detach().cpu().numpy()
linf_values = linf_dist.detach().cpu().numpy()
elastic_values = elastic_dist.detach().cpu().numpy()

# L1 distribution
axes[0, 0].set_title(r"$L_1$ Distortion Distribution", fontsize=14)
axes[0, 0].set_xlabel(r"$L_1$ Distance")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].hist(l1_values, bins=15, color=AZURE, alpha=0.7, edgecolor=NODE_BLACK)
axes[0, 0].axvline(l1_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {l1_values.mean():.2f}")
axes[0, 0].legend(frameon=False, fontsize=10)

# L2 distribution
axes[0, 1].set_title(r"Squared $L_2$ Distortion Distribution", fontsize=14)
axes[0, 1].set_xlabel(r"Squared $L_2$ Distance")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].hist(l2_values, bins=15, color=VIVID_PURPLE, alpha=0.7, edgecolor=NODE_BLACK)
axes[0, 1].axvline(l2_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {l2_values.mean():.2f}")
axes[0, 1].legend(frameon=False, fontsize=10)

# L∞ distribution
axes[1, 0].set_title(r"$L_\infty$ Distortion Distribution", fontsize=14)
axes[1, 0].set_xlabel(r"$L_\infty$ Distance")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].hist(linf_values, bins=15, color=NUGGET_YELLOW, alpha=0.7, edgecolor=NODE_BLACK)
axes[1, 0].axvline(linf_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {linf_values.mean():.2f}")
axes[1, 0].legend(frameon=False, fontsize=10)

# Elastic-net distribution
axes[1, 1].set_title("Elastic-Net Distortion Distribution", fontsize=14)
axes[1, 1].set_xlabel("Elastic Distance")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].hist(elastic_values, bins=15, color=AQUAMARINE, alpha=0.7, edgecolor=NODE_BLACK)
axes[1, 1].axvline(elastic_values.mean(), color=MALWARE_RED, linestyle="--",
                   linewidth=2, label=f"Mean: {elastic_values.mean():.2f}")
axes[1, 1].legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "ead_distortion_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_distortion_analysis.png")

# Visualization 3: Mixed-norm relationship and sparsity-distortion tradeoff
print("Creating mixed-norm analysis...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Prepare data
l1_values = l1_dist.detach().cpu().numpy()
l2_values = l2_dist.detach().cpu().numpy()
perturbations = best_adv - original_images
nonzero_mask = torch.abs(perturbations) > 1e-6
sparsity_per_example = (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / 784) * 100
sparsity_values = sparsity_per_example.detach().cpu().numpy()

# Color by success (green=success, red=failed)
colors = [HTB_GREEN if s else MALWARE_RED for s in final_success.cpu().numpy()]

# Plot 1: L1 vs L2 relationship
axes[0].set_title(r"$L_1$ vs Squared $L_2$ Distortion Relationship", fontsize=14)
axes[0].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
axes[0].set_ylabel(r"$L_1$ Distance", fontsize=12)
axes[0].scatter(l2_values, l1_values, c=colors, s=100, alpha=0.7,
                edgecolors=NODE_BLACK, linewidth=1.5)

# Add reference line showing L1 = sqrt(L2) relationship
l2_range = np.linspace(l2_values.min(), l2_values.max(), 100)
axes[0].plot(l2_range, np.sqrt(l2_range) * 8, color=AZURE, linestyle="--",
             linewidth=2, alpha=0.5, label=r"Reference: $L_1 \propto \sqrt{L_2}$")
axes[0].legend(frameon=False, fontsize=10)
axes[0].grid(True, alpha=0.3, color=HACKER_GREY)

# Add success legend manually
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=HTB_GREEN, edgecolor=NODE_BLACK, label=f'Success ({final_success.sum()}/{batch_size})'),
    Patch(facecolor=MALWARE_RED, edgecolor=NODE_BLACK, label=f'Failed ({(~final_success).sum()}/{batch_size})')
]
axes[0].legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=10)

# Plot 2: Sparsity vs L2 distortion
axes[1].set_title("Sparsity vs Distortion Tradeoff", fontsize=14)
axes[1].set_xlabel(r"Squared $L_2$ Distance", fontsize=12)
axes[1].set_ylabel("Sparsity (%)", fontsize=12)
axes[1].scatter(l2_values, sparsity_values, c=colors, s=100, alpha=0.7,
                edgecolors=NODE_BLACK, linewidth=1.5)

# Add mean lines
axes[1].axvline(l2_values.mean(), color=VIVID_PURPLE, linestyle="--",
                linewidth=2, alpha=0.7, label=f"Mean $L_2$: {l2_values.mean():.2f}")
axes[1].axhline(sparsity_values.mean(), color=AQUAMARINE, linestyle="--",
                linewidth=2, alpha=0.7, label=f"Mean Sparsity: {sparsity_values.mean():.1f}%")
axes[1].legend(frameon=False, fontsize=10)
axes[1].grid(True, alpha=0.3, color=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "ead_success_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_success_analysis.png")

# Visualization 4: Sparsity analysis
print("Creating sparsity analysis...")

fig = plt.figure(figsize=(16, 8))

# Create grid: 2 rows × 5 columns for examples, 1 row for statistics
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 0.6], hspace=0.3, wspace=0.3)

# Show perturbation heatmaps for first 10 examples
num_display_sparse = min(10, batch_size)

for i in range(num_display_sparse):
    row = i // 5
    col = i % 5
    ax = fig.add_subplot(gs[row, col])

    pert = (best_adv[i] - original_images[i]).detach().cpu().squeeze()

    # Show perturbation with colorbar
    im = ax.imshow(torch.abs(pert), cmap="hot", vmin=0, vmax=pert.abs().max())
    ax.axis("off")
    ax.set_title(f"Example {i+1}", fontsize=10)

# Compute sparsity statistics
perturbations = best_adv - original_images
nonzero_mask = torch.abs(perturbations) > 1e-6
sparsity_per_example = (1 - nonzero_mask.float().sum(dim=(1, 2, 3)) / 784) * 100
sparsity_values = sparsity_per_example.detach().cpu().numpy()

# Statistics subplot
ax_stats = fig.add_subplot(gs[2, :])
ax_stats.set_title("Sparsity Distribution Across Examples", fontsize=14)
ax_stats.set_xlabel("Example Index")
ax_stats.set_ylabel("Sparsity (%)")
ax_stats.bar(range(len(sparsity_values)), sparsity_values, color=AQUAMARINE,
             alpha=0.7, edgecolor=NODE_BLACK)
ax_stats.axhline(sparsity_values.mean(), color=MALWARE_RED, linestyle="--",
                 linewidth=2, label=f"Mean: {sparsity_values.mean():.2f}%")
ax_stats.legend(frameon=False, fontsize=10)
ax_stats.set_ylim([0, 100])

plt.tight_layout()
plt.savefig(output_dir / "ead_sparsity_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Saved to {output_dir}/ead_sparsity_analysis.png")

print("\nAll visualizations generated and saved successfully!")
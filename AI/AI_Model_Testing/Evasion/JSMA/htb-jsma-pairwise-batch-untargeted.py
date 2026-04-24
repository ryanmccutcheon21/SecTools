import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from htb_ai_library.core import set_reproducibility
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import SimpleLeNet
from htb_ai_library.training import train_model
from htb_ai_library.utils import save_model, load_model
from htb_ai_library.visualization import use_htb_style

use_htb_style()
set_reproducibility(1337)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

train_loader, test_loader = get_mnist_loaders(batch_size=128)
model_path = output_dir / 'mnist_target.pth'

model = SimpleLeNet().to(device)

if model_path.exists():
    print(f"Loading existing model from {model_path}")
    model = load_model(model, model_path, device)
else:
    print(f"Training new model...")
    model = train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001, device=device)
    save_model(model, model_path)

model.eval()
print(f"Model ready for attacks")

def compute_class_gradient(x, model, class_idx, wrt='logits'):
    x_grad = x.detach().requires_grad_(True)
    logits = model(x_grad)

    if wrt == 'logits':
        scalar = logits[0, class_idx]
    else:
        probs = F.softmax(logits, dim=1)
        scalar = probs[0, class_idx]

    scalar.backward()
    grad = x_grad.grad.detach().cpu().numpy().flatten().copy()

    return grad

# Simple 2-class model for testing
model_test = nn.Sequential(
    nn.Flatten(),
    nn.Linear(4, 2)
)
model_test.eval()

# Test input: 1×1×2×2 image
x_test = torch.tensor([[[[0.5, 0.3], [0.2, 0.8]]]], requires_grad=True)

# Compute gradient for class 0
grad_class0 = compute_class_gradient(x_test, model_test, 0, wrt='logits')
print(f"Gradient shape: {grad_class0.shape}")
print(f"Gradient for class 0: {grad_class0}")

def compute_jacobian_matrix(x, model, num_classes=10, wrt='logits'):
    if x.shape[0] != 1:
        raise ValueError("compute_jacobian_matrix expects batch size 1")

    jacobian = []
    for class_idx in range(num_classes):
        grad = compute_class_gradient(x, model, class_idx, wrt)
        jacobian.append(grad)

    return np.asarray(jacobian)

# MNIST example: 1×1×28×28 input, 10 classes
# Jacobian should be (10, 784)
print(f"For MNIST: {10} classes × {28*28} pixels = {10 * 28*28} values")
print(f"Expected Jacobian shape: (10, 784)")

def extract_target_gradient(jacobian, target_class):
    return jacobian[target_class].copy()


def extract_other_gradients(jacobian, target_class):
    target_grad = jacobian[target_class]
    total_grad = jacobian.sum(axis=0)
    other_grad = total_grad - target_grad
    return other_grad

# Toy Jacobian: 3 classes, 4 features
J_toy = np.array([
    [ 0.2,  0.5, -0.1,  0.3],  # Class 0
    [-0.1,  0.2,  0.4, -0.2],  # Class 1
    [ 0.6, -0.3,  0.1,  0.5]   # Class 2
])

target = 2
alpha = extract_target_gradient(J_toy, target)
beta = extract_other_gradients(J_toy, target)

print(f"Target gradient (class {target}): {alpha}")
print(f"Other gradients sum:              {beta}")

def apply_search_mask(gradient, search_space):
    return gradient * search_space

# Gradient with 5 features
grad = np.array([0.5, -0.2, 0.8, 0.1, -0.4])

# Mask: features 1 and 3 already used
mask = np.array([True, False, True, False, True])

masked_grad = apply_search_mask(grad, mask)
print(f"Original gradient: {grad}")
print(f"Search space mask: {mask}")
print(f"Masked gradient:   {masked_grad}")

def score_increase_saliency(target_grad, other_grad):
    increase_mask = (target_grad > 0) & (other_grad < 0)
    scores = target_grad * np.abs(other_grad) * increase_mask
    return scores

def score_decrease_saliency(target_grad, other_grad):
    decrease_mask = (target_grad < 0) & (other_grad > 0)
    scores = np.abs(target_grad) * other_grad * decrease_mask
    return scores

# Gradients for 6 features
alpha = np.array([ 0.6, -0.3,  0.4,  0.1, -0.5,  0.2])
beta  = np.array([-0.2,  0.4, -0.5,  0.3,  0.6, -0.1])

inc_scores = score_increase_saliency(alpha, beta)
dec_scores = score_decrease_saliency(alpha, beta)

print("Feature | α     β    | Inc Score | Dec Score")
print("--------|-----------|-----------|----------")
for i in range(len(alpha)):
    print(f"   {i}    | {alpha[i]:5.1f} {beta[i]:5.1f} |  {inc_scores[i]:7.3f}  |  {dec_scores[i]:7.3f}")
    
def select_best_direction(inc_scores, dec_scores):
    max_inc_idx = int(np.argmax(inc_scores))
    max_dec_idx = int(np.argmax(dec_scores))

    max_inc_score = float(inc_scores[max_inc_idx])
    max_dec_score = float(dec_scores[max_dec_idx])

    if max_inc_score > max_dec_score:
        return max_inc_idx, max_inc_score, True
    else:
        return max_dec_idx, max_dec_score, False
    
pixel_idx, score, increase = select_best_direction(inc_scores, dec_scores)
direction = "increase" if increase else "decrease"
print(f"Selected: pixel {pixel_idx}, score {score:.3f}, {direction}")

def initialize_search_space(shape):
    num_features = int(np.prod(shape[1:]))
    return np.ones(num_features, dtype=bool)

# MNIST image shape
shape = (1, 1, 28, 28)
search_space = initialize_search_space(shape)
print(f"Search space shape: {search_space.shape}")
print(f"Initial modifiable pixels: {search_space.sum()}")

def remove_saturated_pixels(search_space, x, clip_min=0.0, clip_max=1.0, epsilon=1e-6):
    x_flat = x.detach().cpu().numpy().flatten()

    saturated_min = (x_flat <= clip_min + epsilon)
    saturated_max = (x_flat >= clip_max - epsilon)
    saturated = saturated_min | saturated_max

    updated_mask = search_space & ~saturated
    return updated_mask

# Toy image with some saturated pixels
x_toy = torch.tensor([[[[0.0, 0.3], [0.95, 1.0]]]])  # 4 pixels
mask = np.array([True, True, True, True])

updated_mask = remove_saturated_pixels(mask, x_toy, clip_min=0.0, clip_max=1.0)
print(f"Original mask:  {mask}")
print(f"Pixel values:   {x_toy.flatten().numpy()}")
print(f"Updated mask:   {updated_mask}")
print(f"Remaining pixels: {updated_mask.sum()}/4")


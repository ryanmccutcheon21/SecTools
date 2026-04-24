#!/usr/bin/env python3
"""
FGSM attack using Adversarial Robustness Toolbox (ART)
For HTB Academy MNIST challenge submission
"""

import numpy as np
import torch
import torch.nn as nn
import os, io, base64, requests
from PIL import Image

# Import from HTB Evasion Library
from htb_ai_library import (
    set_reproducibility,
    SimpleCNN,
    get_mnist_loaders,
    mnist_denormalize,
    train_model,
    evaluate_accuracy,
    save_model,
    load_model,
)

# Try to import ART - install if needed
try:
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod
    ART_AVAILABLE = True
except ImportError:
    print("Installing adversarial-robustness-toolbox...")
    os.system("pip install adversarial-robustness-toolbox -q")
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod
    ART_AVAILABLE = True

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")  
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# Configure reproducibility
set_reproducibility(1337)

# Configure computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader, test_loader = get_mnist_loaders(batch_size=64, normalize=True)
    model = SimpleCNN().to(device)

    model = train_model(
        model, train_loader, test_loader,
        epochs=5, device=device
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
            'batch_size': 64,
            'device': str(device)
        }
    }, model_path)

# Evaluate baseline accuracy
baseline_acc = evaluate_accuracy(model, test_loader, device)
print(f"Baseline test accuracy: {baseline_acc:.2f}%")

# Create ART classifier wrapper
# ART expects inputs in [0, 1] range and handles normalization internally
art_classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters()),
    input_shape=(1, 28, 28),
    nb_classes=10,
    preprocessing=(MNIST_MEAN, MNIST_STD),  # normalization params
    device_type='cuda' if torch.cuda.is_available() else 'cpu'
)

# Create FGSM attack using ART
# eps is the perturbation budget in [0,1] space
fgsm_attack = FastGradientMethod(estimator=art_classifier, eps=0.3)

def x01_from_b64_png(b64: str) -> np.ndarray:
    """Convert base64 PNG to [0,1] numpy array."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

def b64_png_from_x01(x2d: np.ndarray) -> str:
    """Convert [0,1] array to base64 PNG."""
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# Fetch challenge from server
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
lab = int(ch["label"]) 
epsilon = float(ch["epsilon"] * 0.95)  # Use slightly less than max to stay under limit

print(f"Challenge: label={lab}, epsilon={epsilon}")

# Get image and prepare for ART
x2d = x01_from_b64_png(ch["image_b64"])  # (28, 28)
x4d = x2d[None, None, ...]  # (1, 1, 28, 28) - ART expects (N, C, H, W)

# Generate adversarial example using ART
# ART's FGSM generates perturbations in the normalized space, then applies to input
# We need to set the attack epsilon to work in the [0,1] space
fgsm_attack.set_params(eps=epsilon)

# ART expects float32 in [0, 1] range
x_adv = fgsm_attack.generate(x=x4d.astype(np.float32), y=np.array([lab]))

# Clip to valid range
x_adv = np.clip(x_adv, 0.0, 1.0)

# Verify perturbation is within epsilon
perturbation = np.abs(x_adv - x4d).max()
print(f"Max perturbation: {perturbation:.4f} (epsilon: {epsilon})")

# Check baseline prediction
res = requests.post(f"{BASE_URL}/predict", json={"image_b64": b64_png_from_x01(x2d)}, timeout=10).json()
print(f"Baseline: true_label={lab}, server_pred={res['pred']}")

# Submit adversarial example
x2d_adv = x_adv[0, 0]  # Convert back to (28, 28)
b64 = b64_png_from_x01(x2d_adv)
bad = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64}, timeout=10)
print(f"Submit status: {bad.status_code}, response: {bad.text}")
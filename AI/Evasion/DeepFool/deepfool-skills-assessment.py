#!/usr/bin/env python3
"""
DeepFool attack on CIFAR-10 using Adversarial Robustness Toolbox.
Untargeted: any misclassification wins.
Constraint: L2(adv - orig) in NORMALIZED space <= 3.5
"""
import os, io, base64, requests
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import DeepFool

from cifarmodel import load_model, NORMALIZATION_MEAN, NORMALIZATION_STD, CIFAR10_CLASSES

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")

# ----------------- helpers -----------------
def tensor_from_b64_png(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return transforms.ToTensor()(img)  # (3,32,32) in [0,1]

def b64_png_from_uint8(arr_u8):
    """arr_u8: (3,H,W) uint8 -> base64 PNG. No float re-quantization."""
    img = Image.fromarray(arr_u8.transpose(1, 2, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def l2_normalized(a_f, b_f, mean_np, std_np):
    """L2 distance in normalized space for (3,H,W) arrays in [0,1]."""
    s = std_np.reshape(3, 1, 1)
    return float(np.linalg.norm((a_f - b_f) / s))

# ----------------- model -----------------
weights_path = "cifar10_model_best.pth"
if not os.path.exists(weights_path):
    print("Downloading model weights...")
    with open(weights_path, "wb") as f:
        f.write(requests.get(f"{BASE_URL}/model/weights").content)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(weights_path, device=device)
model.eval()
print(f"Model loaded on {device}")

# ----------------- challenge -----------------
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x          = tensor_from_b64_png(ch["image"])              # (3,32,32) in [0,1]
orig_class = int(ch["original_class"])
l2_budget  = float(ch["l2_threshold"])
overshoot  = float(ch.get("overshoot_hint", 0.02))
max_iter   = int(ch.get("max_iterations_hint", 50))
mean_np    = np.array(ch["normalization"]["mean"], dtype=np.float32)
std_np     = np.array(ch["normalization"]["std"],  dtype=np.float32)

mean_t = torch.tensor(mean_np).view(3, 1, 1)
std_t  = torch.tensor(std_np).view(3, 1, 1)

with torch.no_grad():
    clean = model(((x - mean_t)/std_t).unsqueeze(0).to(device)).argmax(1).item()
print(f"orig={orig_class} ({CIFAR10_CLASSES[orig_class]})  "
      f"clean_pred={clean}  L2_budget={l2_budget}  overshoot={overshoot}")

# ----------------- ART classifier -----------------
# preprocessing=(mean,std) => ART applies (x-mean)/std internally,
# so the attack operates in [0,1] space and clip_values bound it there.
art_clf = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(0.0, 1.0),
    preprocessing=(mean_np.reshape(3, 1, 1), std_np.reshape(3, 1, 1)),
    device_type="gpu" if device == "cuda" else "cpu",
)

# ----------------- DeepFool -----------------
attack = DeepFool(
    classifier=art_clf,
    max_iter=max_iter,
    epsilon=overshoot,   # ART's `epsilon` is the overshoot multiplier
    nb_grads=10,
    batch_size=1,
)

x4d   = x.unsqueeze(0).numpy().astype(np.float32)
x_adv = attack.generate(x=x4d)                             # untargeted

# pre-quantization sanity
l2_pre = l2_normalized(x_adv[0], x4d[0], mean_np, std_np)
with torch.no_grad():
    pred_pre = model(
        ((torch.from_numpy(x_adv[0]) - mean_t)/std_t).unsqueeze(0).to(device)
    ).argmax(1).item()
print(f"Pre-quant : L2(norm)={l2_pre:.4f}  pred={pred_pre} ({CIFAR10_CLASSES[pred_pre]})")

# ----------------- quantize to uint8 (PNG round-trip) -----------------
orig_u8 = np.round(x4d[0]   * 255).clip(0, 255).astype(np.uint8)
adv_u8  = np.round(x_adv[0] * 255).clip(0, 255).astype(np.uint8)

orig_f = orig_u8.astype(np.float32) / 255.0
adv_f  = adv_u8.astype(np.float32)  / 255.0
l2_post = l2_normalized(adv_f, orig_f, mean_np, std_np)
print(f"Post-quant: L2(norm)={l2_post:.4f}")

# ----------------- if over budget, scale perturbation down -----------------
if l2_post > l2_budget:
    delta = x_adv[0] - x4d[0]
    curr  = float(np.linalg.norm(delta / std_np.reshape(3, 1, 1)))
    safety = 0.95                           # 5% margin for quantization
    scale  = (l2_budget * safety) / max(curr, 1e-12)
    scaled = np.clip(x4d[0] + scale * delta, 0.0, 1.0)
    adv_u8 = np.round(scaled * 255).clip(0, 255).astype(np.uint8)
    adv_f  = adv_u8.astype(np.float32) / 255.0
    l2_post = l2_normalized(adv_f, orig_f, mean_np, std_np)
    print(f"After scale: L2(norm)={l2_post:.4f}")

# ----------------- final verification -----------------
with torch.no_grad():
    final_pred = model(
        ((torch.from_numpy(adv_f) - mean_t)/std_t).unsqueeze(0).to(device)
    ).argmax(1).item()
status = "OK" if final_pred != orig_class else "KILLED BY QUANTIZATION"
print(f"Final adv pred: {final_pred} ({CIFAR10_CLASSES[final_pred]}) [{status}]")

# ----------------- submit -----------------
b64 = b64_png_from_uint8(adv_u8)
r = requests.post(f"{BASE_URL}/submit", json={"image": b64}, timeout=15)
print(r.status_code, r.text)
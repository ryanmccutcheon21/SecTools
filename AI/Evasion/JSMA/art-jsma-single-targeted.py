#!/usr/bin/env python3
"""
JSMA attack on MNIST using ART's SaliencyMapMethod.

Constraints:
  ||x_adv - x||_0 ≤ l0_budget  (pixels with |Δ| > 1e-6)
  ||x_adv - x||_2 ≤ max_l2
  argmax f(x_adv) == target_class
  x_adv ∈ [0, 1]
"""
import os, io, base64, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import SaliencyMapMethod

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
TOTAL_PIXELS = 28 * 28

# ---------- helpers ----------
def x01_from_b64(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    return (np.asarray(img, dtype=np.float32) / 255.0).clip(0, 1)

def b64_from_u8(u8_2d):
    buf = io.BytesIO()
    Image.fromarray(u8_2d, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def count_modified(a, b, threshold=1e-6):
    return int(np.sum(np.abs(a - b) > threshold))

def l2_dist(a, b):
    return float(np.sqrt(((a - b) ** 2).sum()))

# ---------- model: match server but expose raw logits for ART ----------
class MNISTClassifier(nn.Module):
    """Server's LeNet-5, returning raw logits (no log-softmax).
       State-dict is identical to the served weights."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool  = nn.AvgPool2d(2, 2)
        self.fc1   = nn.Linear(16 * 4 * 4, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.act   = nn.Tanh()
    def forward(self, x):
        x = self.act(self.conv1(x)); x = self.pool(x)
        x = self.act(self.conv2(x)); x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)                      # logits

# ---------- fetch challenge + weights ----------
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x          = x01_from_b64(ch["image_b64"])
orig_label = int(ch["original_label"])
target     = int(ch["target_class"])
budget     = int(ch["l0_budget"])
max_l2     = float(ch["max_l2"])
print(f"[*] label={orig_label} target={target} "
      f"L0_budget={budget} max_L2={max_l2}")

wpath = "jsma_weights.pth"
if not os.path.exists(wpath):
    open(wpath, "wb").write(requests.get(f"{BASE_URL}/weights", timeout=10).content)

model = MNISTClassifier().eval()
model.load_state_dict(torch.load(wpath, map_location="cpu"))

# ---------- ART classifier ----------
art_clf = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),                 # logits + CE
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0.0, 1.0),
    preprocessing=(np.array([MNIST_MEAN], dtype=np.float32),
                   np.array([MNIST_STD],  dtype=np.float32)),
    device_type="cpu",
)

x4d  = x[None, None, :, :].astype(np.float32)
x_u8 = np.round(x * 255).clip(0, 255).astype(np.uint8)
x_q  = x_u8.astype(np.float32) / 255.0
clean = int(art_clf.predict(x4d).argmax(1)[0])
print(f"[*] clean local pred: {clean}")

# ---------- ladder: ascending gamma, two directions ----------
# gamma = max fraction of features ART is allowed to perturb.
# Try under-budget first to keep L0 minimal; walk up only if needed.
gammas = sorted({max(b, 2) / TOTAL_PIXELS for b in
                 [budget // 2, int(budget*0.75), budget]})

best = None
for theta_val in (1.0, -1.0):                    # +: add ink, -: erase ink
    for gamma in gammas:
        attack = SaliencyMapMethod(
            classifier=art_clf,
            theta=theta_val,                     # ±1 → saturates to 0 or 1
            gamma=gamma,
            batch_size=1,
            verbose=False,
        )
        adv4d = attack.generate(x=x4d, y=np.array([target]))
        adv01 = adv4d[0, 0]

        # PNG uint8 round-trip (theta=±1 lands on 0/255 → lossless)
        adv_u8 = np.round(adv01 * 255).clip(0, 255).astype(np.uint8)
        adv_q  = adv_u8.astype(np.float32) / 255.0

        l0   = count_modified(adv_q, x_q)
        l2   = l2_dist(adv_q, x_q)
        pred = int(art_clf.predict(adv_q[None, None]).argmax(1)[0])
        fit  = (pred == target and l0 <= budget and l2 <= max_l2)
        flag = "✓" if fit else "✗"
        print(f"  {flag} theta={theta_val:+.0f} gamma={gamma:.3f}: "
              f"pred={pred} L0={l0}/{budget} L2={l2:.3f}/{max_l2:.1f}")

        if fit:
            best = (adv_u8, l0, l2, pred)
            break                                # smallest gamma wins
    if best is not None:
        break                                    # first working direction wins

if best is None:
    raise RuntimeError(
        "JSMA could not satisfy all constraints within L0 budget. "
        "Try raising the upper end of `gammas` or attacking with EAD instead.")

adv_u8, l0, l2, pred = best
print(f"[+] final: pred={pred} (target {target}) "
      f"L0={l0}/{budget} L2={l2:.3f}/{max_l2}")

# ---------- submit ----------
b64 = b64_from_u8(adv_u8)
r = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64}, timeout=15)
print(f"[+] {r.status_code} {r.text}")
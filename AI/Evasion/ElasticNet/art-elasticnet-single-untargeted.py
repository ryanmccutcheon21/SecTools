#!/usr/bin/env python3
"""
ElasticNet (EAD) attack on MNIST using Adversarial Robustness Toolbox.

Constraints (all must hold):
  ||x_adv - x||_2 + β||x_adv - x||_1 ≤ elastic_max
  ||x_adv - x||_2                     ≤ l2_max
  ||x_adv - x||_1                     ≤ l1_max
  argmax f(x_adv) ≠ label
  x_adv ∈ [0, 1]
"""
import os, io, base64, requests
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ElasticNet

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

# ---------- helpers ----------
def x01_from_b64_png(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    return (np.asarray(img, dtype=np.float32) / 255.0).clip(0, 1)

def b64_png_from_u8(u8_2d):
    buf = io.BytesIO()
    Image.fromarray(u8_2d, mode="L").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def distances(a, b, beta):
    d = (a - b).ravel()
    l1 = float(np.abs(d).sum())
    l2 = float(np.sqrt((d**2).sum()))
    return {"l1": l1, "l2": l2, "linf": float(np.abs(d).max()),
            "elastic": l2 + beta * l1}

def constraints_ok(dist, elastic_max, l2_max, l1_max):
    return (dist["elastic"] <= elastic_max
            and dist["l2"]  <= l2_max
            and dist["l1"]  <= l1_max)

# ---------- model ----------
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# ---------- fetch challenge + weights ----------
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x           = x01_from_b64_png(ch["image_b64"])                 # (28, 28)
label       = int(ch["label"])
beta        = float(ch["beta"])
elastic_max = float(ch["elastic_max"])
l2_max      = float(ch["l2_max"])
l1_max      = float(ch["l1_max"])
print(f"[*] label={label} beta={beta} elastic_max={elastic_max} "
      f"l2_max={l2_max} l1_max={l1_max}")

wpath = "elasticnet_weights.pth"
if not os.path.exists(wpath):
    open(wpath, "wb").write(requests.get(f"{BASE_URL}/weights", timeout=10).content)

model = SimpleClassifier().eval()
model.load_state_dict(torch.load(wpath, map_location="cpu"))

# ---------- ART classifier ----------
# preprocessing=(mean,std) => model sees normalised inputs, attack operates in [0,1]
art_clf = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    input_shape=(1, 28, 28),
    nb_classes=10,
    clip_values=(0.0, 1.0),
    preprocessing=(np.array([MNIST_MEAN], dtype=np.float32),
                   np.array([MNIST_STD],  dtype=np.float32)),
    device_type="cpu",
)

# sanity
x4d = x[None, None, :, :].astype(np.float32)
clean_pred = int(art_clf.predict(x4d).argmax(1)[0])
print(f"[*] clean local pred: {clean_pred} (server label: {label})")

# ---------- ElasticNet attack ----------
# decision_rule='EN' = elastic-net objective (matches server's constraint)
# beta mirrors the challenge's β so FISTA optimises the same penalty
attack = ElasticNet(
    classifier=art_clf,
    confidence=0.0,             # no extra margin → minimal distortion
    targeted=False,
    learning_rate=0.01,
    binary_search_steps=9,      # finds a good const trading distortion ↔ success
    max_iter=100,
    beta=beta,
    initial_const=1e-3,
    batch_size=1,
    decision_rule="EN",
    verbose=False,
)

x_adv = attack.generate(x=x4d)  # untargeted → no y needed
adv01 = x_adv[0, 0]             # (28, 28) float in [0,1]

pre = distances(adv01, x, beta)
pre_pred = int(art_clf.predict(x_adv).argmax(1)[0])
print(f"[*] pre-quant : pred={pre_pred}  "
      f"L1={pre['l1']:.4f}  L2={pre['l2']:.4f}  elastic={pre['elastic']:.4f}")

# ---------- uint8 PNG round-trip ----------
# round (not truncate) keeps quantisation error ≤ 0.5/255 per pixel
adv_u8 = np.round(adv01 * 255).clip(0, 255).astype(np.uint8)
adv_q  = adv_u8.astype(np.float32) / 255.0
x_u8   = np.round(x    * 255).clip(0, 255).astype(np.uint8)
x_q    = x_u8.astype(np.float32) / 255.0

q = distances(adv_q, x_q, beta)
q_pred = int(art_clf.predict(adv_q[None, None]).argmax(1)[0])
print(f"[*] post-quant: pred={q_pred}  "
      f"L1={q['l1']:.4f}  L2={q['l2']:.4f}  elastic={q['elastic']:.4f}")

# ---------- shrink toward x if any constraint is violated ----------
# All three distances scale linearly with α when adv = x + α·δ, so the tightest
# ratio gives the largest α we can keep while satisfying every constraint.
if not constraints_ok(q, elastic_max, l2_max, l1_max) or q_pred == label:
    delta = adv01 - x
    ratios = []
    d0 = distances(adv01, x, beta)
    if d0["elastic"] > 0: ratios.append(elastic_max / d0["elastic"])
    if d0["l2"]      > 0: ratios.append(l2_max      / d0["l2"])
    if d0["l1"]      > 0: ratios.append(l1_max      / d0["l1"])
    alpha = min(ratios) * 0.95 if ratios else 1.0      # 5% margin for quant
    scaled = np.clip(x + alpha * delta, 0.0, 1.0)
    adv_u8 = np.round(scaled * 255).clip(0, 255).astype(np.uint8)
    adv_q  = adv_u8.astype(np.float32) / 255.0
    q      = distances(adv_q, x_q, beta)
    q_pred = int(art_clf.predict(adv_q[None, None]).argmax(1)[0])
    print(f"[*] shrunk by α={alpha:.3f}: pred={q_pred}  "
          f"L1={q['l1']:.4f}  L2={q['l2']:.4f}  elastic={q['elastic']:.4f}")

# ---------- verify locally ----------
ok = constraints_ok(q, elastic_max, l2_max, l1_max) and q_pred != label
print(f"[{'✓' if ok else '✗'}] local check: "
      f"misclassified={q_pred != label}  constraints={constraints_ok(q, elastic_max, l2_max, l1_max)}")
if not ok:
    print("[!] Submitting anyway — server may round differently than local quant")

# ---------- submit ----------
b64 = b64_png_from_u8(adv_u8)
r = requests.post(f"{BASE_URL}/submit", json={"image_b64": b64}, timeout=15)
print(f"[+] {r.status_code} {r.text}")
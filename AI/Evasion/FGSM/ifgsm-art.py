import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os, io, base64, requests
from PIL import Image

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import BasicIterativeMethod  # iterative FGSM

from htb_ai_library import set_reproducibility
from cifarmodel import load_model

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
set_reproducibility(1337)

# --- load model ---
weights_path = "cifar10_model_best.pth"
if not os.path.exists(weights_path):
    with open(weights_path, "wb") as f:
        f.write(requests.get(f"{BASE_URL}/model/weights").content)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(weights_path, device=device)
model.eval()

# --- helpers ---
def tensor_from_b64_png(b64):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return transforms.ToTensor()(img)  # (3, 32, 32) in [0,1]

def b64_png_from_tensor(t):
    arr = (t.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

# --- fetch challenge ---
ch = requests.get(f"{BASE_URL}/challenge", timeout=10).json()
x = tensor_from_b64_png(ch["image"])            # (3, 32, 32)
orig_class   = int(ch["original_class"])
target_class = int(ch["target_class"])
epsilon      = float(ch["epsilon"])
mean_np      = np.array(ch["normalization"]["mean"], dtype=np.float32)  # (3,)
std_np       = np.array(ch["normalization"]["std"],  dtype=np.float32)  # (3,)

mean_t = torch.tensor(mean_np).view(3, 1, 1)
std_t  = torch.tensor(std_np).view(3, 1, 1)

with torch.no_grad():
    clean_pred = model(((x - mean_t) / std_t).unsqueeze(0).to(device)).argmax(1).item()
print(f"orig={orig_class} target={target_class} clean_pred={clean_pred} eps={epsilon:.6f}")

# --- ART classifier: preprocessing is (mean, std) per-channel, model gets input in [0,1] ---
art_clf = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    input_shape=(3, 32, 32),
    nb_classes=10,
    clip_values=(0.0, 1.0),
    preprocessing=(mean_np.reshape(3, 1, 1), std_np.reshape(3, 1, 1)),
    device_type="cuda" if torch.cuda.is_available() else "cpu",
)

# --- iterative targeted FGSM (a.k.a. BIM) ---
attack = BasicIterativeMethod(
    estimator=art_clf,
    eps=epsilon * 0.95,
    eps_step=epsilon / 10,
    max_iter=40,
    targeted=True,
)

x4d = x.unsqueeze(0).numpy().astype(np.float32)  # (1, 3, 32, 32) in [0,1]
y   = np.array([target_class])

x_adv = attack.generate(x=x4d, y=y)  # ART enforces clip + eps internally

print(f"linf perturbation: {np.abs(x_adv - x4d).max():.6f} (budget {epsilon:.6f})")

# sanity-check locally before submitting
with torch.no_grad():
    adv_t = torch.from_numpy(x_adv[0])
    local_pred = model(((adv_t - mean_t) / std_t).unsqueeze(0).to(device)).argmax(1).item()
print(f"local adv pred: {local_pred} (want {target_class})")

# --- submit ---
b64 = b64_png_from_tensor(torch.from_numpy(x_adv[0]))  # (3, 32, 32)
r = requests.post(f"{BASE_URL}/submit", json={"image": b64}, timeout=10)
print("== Submission Result ==")
print(r.status_code, r.text)
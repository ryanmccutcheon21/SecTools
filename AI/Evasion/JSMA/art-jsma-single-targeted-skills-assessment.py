#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import ElasticNet, SaliencyMapMethod
except ImportError as exc:
    raise SystemExit(
        "Missing ART. Install it with:\n"
        "python3 -m pip install adversarial-robustness-toolbox"
    ) from exc


SEED = 1337

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_host(host: str) -> str:
    host = host.strip().rstrip("/")

    if not host:
        raise ValueError("Empty host. Use --host http://IP:PORT")

    return host


def make_url(host: str, path: str) -> str:
    return f"{clean_host(host)}/{path.lstrip('/')}"


def http_get_json(host: str, path: str, timeout: int = 30) -> Any:
    url = make_url(host, path)

    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}:\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def http_post_json(host: str, path: str, body: Dict[str, Any], timeout: int = 90) -> Any:
    url = make_url(host, path)
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}:\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc


def download_binary(host: str, path: str, out_path: str, timeout: int = 60) -> None:
    url = path if path.startswith("http://") or path.startswith("https://") else make_url(host, path)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download {url} failed with HTTP {exc.code}:\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Download {url} failed: {exc}") from exc

    with open(out_path, "wb") as f:
        f.write(data)


def one_hot(label: int, nb_classes: int = 10) -> np.ndarray:
    y = np.zeros((1, nb_classes), dtype=np.float32)
    y[0, int(label)] = 1.0
    return y


def local_pred(classifier: PyTorchClassifier, x: np.ndarray) -> int:
    out = classifier.predict(x.astype(np.float32), batch_size=1)
    return int(np.argmax(out[0]))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a.astype(np.float32) - b.astype(np.float32)).reshape(-1)))


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


# ---------------------------------------------------------------------
# CIFAR image helpers
# ---------------------------------------------------------------------

def cifar_x01_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    if img.size != (32, 32):
        raise ValueError(f"Expected 32x32 RGB PNG, got {img.size}")

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def cifar_b64_from_x01(x4d: np.ndarray) -> str:
    x = np.transpose(x4d[0], (1, 2, 0))
    x255 = np.clip((x * 255.0).round(), 0, 255).astype(np.uint8)

    img = Image.fromarray(x255, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def cifar_roundtrip(x4d: np.ndarray) -> np.ndarray:
    return cifar_x01_from_b64(cifar_b64_from_x01(x4d))


# ---------------------------------------------------------------------
# MNIST image helpers
# ---------------------------------------------------------------------

def mnist_x01_from_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")

    if img.size != (28, 28):
        raise ValueError(f"Expected 28x28 grayscale PNG, got {img.size}")

    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def mnist_b64_from_x01(x2d: np.ndarray) -> str:
    x255 = np.clip((x2d * 255.0).round(), 0, 255).astype(np.uint8)

    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)

    return base64.b64encode(buf.getvalue()).decode("ascii")


def mnist_roundtrip_4d(x4d: np.ndarray) -> np.ndarray:
    x2d = x4d[0, 0]
    q2d = mnist_x01_from_b64(mnist_b64_from_x01(x2d))
    return q2d[None, None, ...].astype(np.float32)


def count_mnist_l0(a4d: np.ndarray, b4d: np.ndarray, threshold: float = 1e-6) -> int:
    return int(np.sum(np.abs(a4d[0, 0] - b4d[0, 0]) > threshold))


def mnist_project_topk(x0: np.ndarray, cand: np.ndarray, budget: int) -> np.ndarray:
    diff = cand - x0
    flat_abs = np.abs(diff.reshape(-1))

    changed = np.where(flat_abs > 1e-6)[0]
    if len(changed) <= budget:
        return cand

    keep = changed[np.argsort(flat_abs[changed])[::-1][:budget]]

    out = x0.copy().reshape(-1)
    cand_flat = cand.reshape(-1)
    out[keep] = cand_flat[keep]

    return out.reshape(cand.shape).astype(np.float32)


# ---------------------------------------------------------------------
# CIFAR model
# ---------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2), num_classes: int = 10) -> None:
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class CIFARNormalizedWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean: Iterable[float], std: Iterable[float]) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(tuple(mean), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(tuple(std), dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def load_cifar_model(host: str, device: torch.device, weights_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    meta = http_get_json(host, "/model")

    if not os.path.exists(weights_path):
        weights_url = meta.get("weights_url", "/model/weights")
        print(f"[+] Downloading CIFAR weights from {weights_url} -> {weights_path}")
        download_binary(host, weights_url, weights_path)

    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict):
        sd = ckpt.get("state_dict_ema") or ckpt.get("state_dict") or ckpt
    else:
        sd = ckpt

    sd = strip_module_prefix(sd)

    model = ResNetCIFAR().to(device)
    model.load_state_dict(sd)
    model.eval()

    return model, meta


# ---------------------------------------------------------------------
# MNIST model
# ---------------------------------------------------------------------

class MNISTClassifierLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class MNISTNormalizedWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - MNIST_MEAN) / MNIST_STD)


def load_mnist_model(host: str, device: torch.device, weights_path: str) -> nn.Module:
    if not os.path.exists(weights_path):
        print(f"[+] Downloading MNIST weights -> {weights_path}")
        download_binary(host, "/weights", weights_path)

    state = torch.load(weights_path, map_location=device)
    state = strip_module_prefix(state)

    model = MNISTClassifierLogits().to(device)
    model.load_state_dict(state)
    model.eval()

    return model


# ---------------------------------------------------------------------
# ART classifier builders
# ---------------------------------------------------------------------

def make_art_classifier(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> PyTorchClassifier:
    kwargs = dict(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=10,
        clip_values=(0.0, 1.0),
    )

    try:
        return PyTorchClassifier(
            **kwargs,
            device_type="gpu" if device.type == "cuda" else "cpu",
        )
    except TypeError:
        return PyTorchClassifier(**kwargs)


# ---------------------------------------------------------------------
# CIFAR ART attacks
# ---------------------------------------------------------------------

def normalize_method(method: str) -> str:
    m = str(method or "either").strip().lower()

    if m in ("jsma", "jacobian", "saliency", "saliencymap"):
        return "jacobian"

    if m in ("ead", "elastic", "elasticnet", "elastic-net", "elastic_net"):
        return "ead"

    if m in ("either", "any", "both", ""):
        return "either"

    return m


def ensure_cifar_min_l2(
    classifier: PyTorchClassifier,
    x0: np.ndarray,
    cand: np.ndarray,
    target: int,
    min_l2: float,
) -> Optional[np.ndarray]:
    cand = np.clip(cifar_roundtrip(cand), 0.0, 1.0).astype(np.float32)

    pred = local_pred(classifier, cand)
    dist = l2_distance(x0, cand)

    if pred == target and dist >= min_l2:
        return cand

    if pred != target:
        return None

    diff = cand - x0
    base = l2_distance(x0, cand)

    if base < 1e-9:
        return None

    start = max(1.05, (min_l2 / base) * 1.05)

    for scale in np.linspace(start, 4.0, 20):
        trial = np.clip(x0 + diff * scale, 0.0, 1.0).astype(np.float32)
        trial = cifar_roundtrip(trial)

        if local_pred(classifier, trial) == target and l2_distance(x0, trial) >= min_l2:
            return trial

    return None


def try_cifar_ead(
    classifier: PyTorchClassifier,
    x: np.ndarray,
    target: int,
    min_l2: float,
) -> Optional[np.ndarray]:
    y = one_hot(target)

    configs = [
        dict(confidence=0.0, learning_rate=0.01, binary_search_steps=6, max_iter=300, beta=0.01, initial_const=0.01, decision_rule="EN"),
        dict(confidence=0.0, learning_rate=0.005, binary_search_steps=6, max_iter=500, beta=0.005, initial_const=0.02, decision_rule="EN"),
        dict(confidence=0.0, learning_rate=0.01, binary_search_steps=8, max_iter=500, beta=0.001, initial_const=0.05, decision_rule="EN"),
        dict(confidence=0.5, learning_rate=0.005, binary_search_steps=8, max_iter=700, beta=0.001, initial_const=0.1, decision_rule="EN"),
    ]

    for i, cfg in enumerate(configs, 1):
        print(f"    [EAD {i}/{len(configs)}] {cfg}")

        try:
            attack = ElasticNet(
                classifier=classifier,
                targeted=True,
                batch_size=1,
                verbose=False,
                **cfg,
            )
            adv = attack.generate(x=x.astype(np.float32), y=y)
            adv = np.clip(adv, 0.0, 1.0).astype(np.float32)

            checked = ensure_cifar_min_l2(classifier, x, adv, target, min_l2)
            if checked is not None:
                print(
                    f"    [+] EAD local success: pred={local_pred(classifier, checked)}, "
                    f"L2={l2_distance(x, checked):.4f}"
                )
                return checked

            q = cifar_roundtrip(adv)
            print(
                f"    [-] EAD miss: pred={local_pred(classifier, q)}, "
                f"L2={l2_distance(x, q):.4f}"
            )

        except Exception as exc:
            print(f"    [!] EAD error: {exc}")

    return None


def try_cifar_jsma(
    classifier: PyTorchClassifier,
    x: np.ndarray,
    target: int,
    min_l2: float,
) -> Optional[np.ndarray]:
    y = one_hot(target)

    configs = [
        dict(theta=0.10, gamma=0.15),
        dict(theta=0.15, gamma=0.20),
        dict(theta=0.20, gamma=0.25),
        dict(theta=-0.10, gamma=0.15),
        dict(theta=-0.15, gamma=0.20),
        dict(theta=-0.20, gamma=0.25),
    ]

    for i, cfg in enumerate(configs, 1):
        print(f"    [JSMA {i}/{len(configs)}] {cfg}")

        try:
            attack = SaliencyMapMethod(
                classifier=classifier,
                batch_size=1,
                verbose=False,
                **cfg,
            )
            adv = attack.generate(x=x.astype(np.float32), y=y)
            adv = np.clip(adv, 0.0, 1.0).astype(np.float32)

            checked = ensure_cifar_min_l2(classifier, x, adv, target, min_l2)
            if checked is not None:
                print(
                    f"    [+] JSMA local success: pred={local_pred(classifier, checked)}, "
                    f"L2={l2_distance(x, checked):.4f}"
                )
                return checked

            q = cifar_roundtrip(adv)
            print(
                f"    [-] JSMA miss: pred={local_pred(classifier, q)}, "
                f"L2={l2_distance(x, q):.4f}"
            )

        except Exception as exc:
            print(f"    [!] JSMA error: {exc}")

    return None


def server_predict_cifar(host: str, x: np.ndarray) -> Optional[int]:
    try:
        resp = http_post_json(host, "/predict", {"image_b64": cifar_b64_from_x01(x)}, timeout=30)
    except Exception as exc:
        print(f"    [!] Server /predict unavailable or failed: {exc}")
        return None

    for key in ("predicted_class", "prediction", "pred", "label"):
        if key in resp:
            return int(resp[key])

    return None


def solve_cifar(host: str, ch: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    print("[+] Detected CIFAR-10 ResNet challenge")

    model, meta = load_cifar_model(host, device, args.cifar_weights)

    norm = meta.get("normalize", {})
    mean = norm.get("mean", CIFAR10_MEAN)
    std = norm.get("std", CIFAR10_STD)

    wrapped = CIFARNormalizedWrapper(model, mean=mean, std=std).to(device).eval()
    classifier = make_art_classifier(wrapped, input_shape=(3, 32, 32), device=device)

    submit_items = []

    for raw_item in ch["items"]:
        sample_id = int(raw_item["sample_id"])
        target = int(raw_item["target"])
        required = normalize_method(raw_item.get("required_method", "either"))
        x = cifar_x01_from_b64(raw_item["image_b64"])

        clean_pred = local_pred(classifier, x)
        print(
            f"[Sample {sample_id}] clean_pred={clean_pred}, "
            f"label={raw_item.get('label')}, target={target}, required={required}"
        )

        order = [required] if required != "either" else ["ead", "jacobian"]

        adv = None
        method_used = None

        for method in order:
            if method == "ead":
                adv = try_cifar_ead(classifier, x, target, args.cifar_min_l2)
                method_used = "ead" if adv is not None else None

            elif method == "jacobian":
                adv = try_cifar_jsma(classifier, x, target, args.cifar_min_l2)
                method_used = "jacobian" if adv is not None else None

            else:
                raise RuntimeError(f"Unknown required_method from challenge: {required}")

            if adv is not None:
                server_pred = server_predict_cifar(host, adv)
                if server_pred is None or server_pred == target:
                    if server_pred is not None:
                        print(f"    [+] Server /predict success: {server_pred}")
                    break

                print(f"    [-] Local success but server predicted {server_pred}; trying next method")
                adv = None
                method_used = None

        if adv is None or method_used is None:
            raise RuntimeError(f"Could not craft successful adversarial image for sample {sample_id}")

        submit_items.append(
            {
                "sample_id": sample_id,
                "method": method_used,
                "image_b64": cifar_b64_from_x01(adv),
            }
        )

    print("[+] Submitting CIFAR adversarial bundle")
    resp = http_post_json(host, "/submit_images", {"items": submit_items}, timeout=120)
    print(json.dumps(resp, indent=2))

    if not resp.get("ok", False) and not resp.get("success", False):
        raise SystemExit(2)

    if resp.get("flag"):
        print("Flag:", resp["flag"])


# ---------------------------------------------------------------------
# MNIST ART JSMA
# ---------------------------------------------------------------------

def validate_mnist_candidate(
    classifier: PyTorchClassifier,
    x0: np.ndarray,
    cand: np.ndarray,
    target: int,
    budget: int,
    max_l2: float,
) -> Tuple[bool, np.ndarray, int, float, int]:
    cand = np.clip(cand, 0.0, 1.0).astype(np.float32)
    cand = mnist_roundtrip_4d(cand)

    pred = local_pred(classifier, cand)
    l0 = count_mnist_l0(x0, cand)
    dist = l2_distance(x0, cand)

    ok = pred == target and l0 <= budget and dist <= max_l2
    return ok, cand, pred, l0, dist


def try_mnist_jsma(
    classifier: PyTorchClassifier,
    x: np.ndarray,
    target: int,
    budget: int,
    max_l2: float,
) -> Optional[np.ndarray]:
    y = one_hot(target)

    base_gamma = max(1.0 / 784.0, min(1.0, budget / 784.0))

    configs = []
    for gamma_mul in (0.75, 0.9, 1.0):
        gamma = max(1.0 / 784.0, min(base_gamma, base_gamma * gamma_mul))
        for theta in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, -1.0, -0.7, -0.5, -0.3, -0.2, -0.1):
            configs.append(dict(theta=theta, gamma=gamma))

    seen = set()
    uniq_configs = []
    for cfg in configs:
        key = (round(cfg["theta"], 4), round(cfg["gamma"], 6))
        if key not in seen:
            seen.add(key)
            uniq_configs.append(cfg)

    for i, cfg in enumerate(uniq_configs, 1):
        print(f"    [MNIST JSMA {i}/{len(uniq_configs)}] {cfg}")

        try:
            attack = SaliencyMapMethod(
                classifier=classifier,
                batch_size=1,
                verbose=False,
                **cfg,
            )

            adv = attack.generate(x=x.astype(np.float32), y=y)
            adv = np.clip(adv, 0.0, 1.0).astype(np.float32)

            # Strictly keep at most budget pixels, in case PNG round-trip or ART exceeds it.
            adv = mnist_project_topk(x, adv, budget)

            ok, q, pred, l0, dist = validate_mnist_candidate(
                classifier, x, adv, target, budget, max_l2
            )

            print(f"        pred={pred}, L0={l0}/{budget}, L2={dist:.4f}/{max_l2:.4f}")

            if ok:
                return q

        except Exception as exc:
            print(f"    [!] MNIST JSMA error: {exc}")

    return None


def server_predict_mnist(host: str, x: np.ndarray) -> Optional[int]:
    try:
        resp = http_post_json(
            host,
            "/predict",
            {"image_b64": mnist_b64_from_x01(x[0, 0])},
            timeout=30,
        )
    except Exception as exc:
        print(f"    [!] Server /predict unavailable or failed: {exc}")
        return None

    for key in ("predicted_class", "prediction", "pred", "label"):
        if key in resp:
            return int(resp[key])

    return None


def solve_mnist(host: str, ch: Dict[str, Any], args: argparse.Namespace, device: torch.device) -> None:
    print("[+] Detected MNIST JSMA challenge")

    x2d = mnist_x01_from_b64(ch["image_b64"])
    x = x2d[None, None, ...].astype(np.float32)

    target = int(ch["target_class"])
    original_label = int(ch.get("original_label", -1))
    budget = int(ch["l0_budget"])
    max_l2 = float(ch["max_l2"])

    model = load_mnist_model(host, device, args.mnist_weights)
    wrapped = MNISTNormalizedWrapper(model).to(device).eval()
    classifier = make_art_classifier(wrapped, input_shape=(1, 28, 28), device=device)

    clean_pred = local_pred(classifier, x)
    print(
        f"[MNIST] clean_pred={clean_pred}, original_label={original_label}, "
        f"target={target}, L0_budget={budget}, max_l2={max_l2}"
    )

    adv = try_mnist_jsma(classifier, x, target, budget, max_l2)

    if adv is None:
        raise RuntimeError("Could not craft a valid MNIST JSMA adversarial example")

    server_pred = server_predict_mnist(host, adv)
    if server_pred is not None:
        print(f"[+] Server /predict: {server_pred}")

    print("[+] Submitting MNIST adversarial image")
    resp = http_post_json(
        host,
        "/submit",
        {"image_b64": mnist_b64_from_x01(adv[0, 0])},
        timeout=60,
    )

    print(json.dumps(resp, indent=2))

    if not resp.get("success", False) and not resp.get("ok", False):
        raise SystemExit(2)

    if resp.get("flag"):
        print("Flag:", resp["flag"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--cifar-weights", default="cifar10_model.pth")
    parser.add_argument("--mnist-weights", default="jsma_weights.pth")
    parser.add_argument("--cifar-min-l2", type=float, default=1.55)
    args = parser.parse_args()

    set_seed(SEED)

    host = clean_host(args.host)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("[!] CUDA not available; ART attacks may be slower on CPU.")

    try:
        health = http_get_json(host, "/health", timeout=15)
        print("[+] Health:", json.dumps(health))
    except Exception as exc:
        print(f"[!] /health failed, continuing anyway: {exc}")

    ch = http_get_json(host, "/challenge", timeout=30)

    if isinstance(ch, dict) and "items" in ch:
        solve_cifar(host, ch, args, device)
        return

    if isinstance(ch, dict) and "target_class" in ch and "l0_budget" in ch:
        solve_mnist(host, ch, args, device)
        return

    raise RuntimeError(
        "Unknown challenge shape. Response was:\n"
        + json.dumps(ch, indent=2)[:3000]
    )


if __name__ == "__main__":
    main()
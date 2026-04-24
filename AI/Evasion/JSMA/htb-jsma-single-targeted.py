#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple


import numpy as np
import requests
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class MNISTClassifier(nn.Module):
    """LeNet-5 style network mirroring the FastAPI server model."""

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
        """Return log-softmax scores for the supplied MNIST batch."""
        x = self.act(self.conv1(x))  # (B,6,24,24)
        x = self.pool(x)  # (B,6,12,12)
        x = self.act(self.conv2(x))  # (B,16,8,8)
        x = self.pool(x)  # (B,16,4,4)
        x = torch.flatten(x, 1)  # (B,256)
        x = self.act(self.fc1(x))  # (B,120)
        x = self.act(self.fc2(x))  # (B,84)
        x = self.fc3(x)  # (B,10)
        return F.log_softmax(x, dim=1)


def mnist_normalize(x01: torch.Tensor) -> torch.Tensor:
    """Normalize a [0,1] tensor to MNIST stats for the classifier."""
    return (x01 - MNIST_MEAN) / MNIST_STD

def png_from_x01(x01: np.ndarray) -> str:
    """Encode a [0,1] grayscale image (28,28) to base64 PNG string."""
    x255 = np.clip((x01 * 255.0).round(), 0, 255).astype(np.uint8)
    img = Image.fromarray(x255, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def x01_from_b64_png(b64: str) -> np.ndarray:
    """Decode base64 PNG to [0,1] numpy array of shape (28,28)."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    if img.size != (28, 28):
        raise ValueError("Expected 28x28 PNG")
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)

@dataclass
class Challenge:
    target_class: int
    l0_budget: int
    original_label: int
    sample_index: int
    x01: np.ndarray  # (1,1,28,28)


def fetch_challenge(host: str, retries: int = 30, delay: float = 1.0) -> Challenge:
    """Fetch challenge with simple retry/backoff to tolerate startup races."""
    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = requests.get(f"{host}/challenge", timeout=5)
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            last_err = e
            time.sleep(delay)
    else:
        raise RuntimeError(
            f"Failed to connect to {host}/challenge: {last_err}"
        ) from last_err
    x2d = x01_from_b64_png(payload["image_b64"])  # (28,28)
    x4d = x2d[None, None, ...]
    return Challenge(
        target_class=int(payload["target_class"]),
        l0_budget=int(payload["l0_budget"]),
        original_label=int(payload["original_label"]),
        sample_index=int(payload["sample_index"]),
        x01=x4d.astype(np.float32),
    )
    
def load_model(host: str, weights_path: Optional[str] = None) -> MNISTClassifier:
    """Load the JSMA classifier using either a local file or the lab endpoint."""
    model = MNISTClassifier()
    if weights_path:
        state = torch.load(weights_path, map_location=torch.device("cpu"))
    else:
        resp = requests.get(f"{host}/weights", timeout=30)
        try:
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download weights from {host}/weights"
            ) from exc
        buffer = io.BytesIO(resp.content)
        state = torch.load(buffer, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model

def compute_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian matrix for all classes with respect to input.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    x : torch.Tensor
        Input tensor of shape (1, 1, 28, 28) with requires_grad=True.

    Returns
    -------
    torch.Tensor
        Jacobian matrix of shape (10, 784) where entry [k, i] is dF_k/dx_i.
    """
    x_flat = x.view(-1)  # Flatten to (784,)
    num_classes = 10
    num_features = x_flat.shape[0]

    jacobian = torch.zeros(num_classes, num_features)

    # Compute gradient for each class
    for k in range(num_classes):
        if x.grad is not None:
            x.grad.zero_()

        # Forward pass to get logits
        logits = model(mnist_normalize(x))

        # Select k-th class logit
        selected = logits[0, k]

        # Backward pass
        selected.backward(retain_graph=True)

        # Store gradient
        jacobian[k] = x.grad.view(-1).clone()

    return jacobian

def compute_saliency_map(
    jacobian: torch.Tensor, target: int, search_space: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute JSMA saliency scores for increasing and decreasing features.

    Parameters
    ----------
    jacobian : torch.Tensor
        Jacobian matrix of shape (num_classes, num_features).
    target : int
        Target class index.
    search_space : torch.Tensor
        Boolean mask of valid features to consider.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Saliency scores for increasing features and decreasing features.
    """
    num_features = jacobian.shape[1]

    # Extract target gradients
    target_grad = jacobian[target]

    # Sum of other class gradients
    other_grad_sum = jacobian.sum(dim=0) - target_grad

    # Saliency for increasing features
    saliency_inc = torch.zeros(num_features)
    mask_inc = (target_grad > 0) & (other_grad_sum < 0) & search_space
    saliency_inc[mask_inc] = target_grad[mask_inc] * torch.abs(other_grad_sum[mask_inc])

    # Saliency for decreasing features
    saliency_dec = torch.zeros(num_features)
    mask_dec = (target_grad < 0) & (other_grad_sum > 0) & search_space
    saliency_dec[mask_dec] = torch.abs(target_grad[mask_dec]) * other_grad_sum[mask_dec]

    return saliency_inc, saliency_dec

def compute_pairwise_saliency(
    jacobian: torch.Tensor,
    target: int,
    search_space: torch.Tensor,
    direction: str,
    top_k: Optional[int] = 128,
) -> Tuple[int, int, float]:
    """Select the highest scoring pixel pair for the requested JSMA direction.

    Parameters
    ----------
    jacobian : torch.Tensor
        Jacobian matrix of shape (num_classes, num_features).
    target : int
        Target class index.
    search_space : torch.Tensor
        Boolean mask of valid features to consider.
    direction : str
        Either ``"increase"`` or ``"decrease"`` to signal the perturbation sign.
    top_k : Optional[int]
        Optional cap on the number of candidate features considered per pair.

    Returns
    -------
    Tuple[int, int, float]
        Pair of flattened pixel indices and their saliency score. ``(-1, -1, 0.0)``
        is returned when no valid pair exists.
    """
    valid = torch.nonzero(search_space, as_tuple=False).squeeze(1)
    if valid.numel() < 2:
        return -1, -1, 0.0

    target_grad = jacobian[target]
    other_grad = jacobian.sum(dim=0) - target_grad

    if top_k is not None and valid.numel() > top_k:
        alpha = target_grad[valid]
        beta = other_grad[valid]
        prelim = torch.abs(alpha) * torch.abs(beta)
        _, top_idx = torch.topk(prelim, top_k)
        valid = valid[top_idx]

    best_score = 0.0
    best_pair = (-1, -1)

    for i in range(valid.numel()):
        p = int(valid[i].item())
        alpha_p = target_grad[p]
        beta_p = other_grad[p]
        for j in range(i + 1, valid.numel()):
            q = int(valid[j].item())
            alpha_sum = alpha_p + target_grad[q]
            beta_sum = beta_p + other_grad[q]

            if direction == "increase":
                if alpha_sum <= 0 or beta_sum >= 0:
                    continue
                score = alpha_sum * torch.abs(beta_sum)
            else:
                if alpha_sum >= 0 or beta_sum <= 0:
                    continue
                score = torch.abs(alpha_sum) * beta_sum

            score_val = float(score.item())
            if score_val > best_score:
                best_score = score_val
                best_pair = (p, q)

    if best_pair[0] == -1:
        return -1, -1, 0.0
    return best_pair[0], best_pair[1], best_score


# ------------- JSMA Targeted Attack Implementation --------------
def jsma_targeted(
    model: nn.Module,
    x01: np.ndarray,
    target_class: int,
    l0_budget: int,
    theta: float = 1.0,
    max_iters: int = 2000,
) -> np.ndarray:
    """Perform JSMA targeted attack.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    x01 : np.ndarray
        Baseline [0,1] image with shape (1,1,28,28).
    target_class : int
        Target class for misclassification.
    l0_budget : int
        Maximum number of pixels to modify.
    theta : float
        Modification magnitude per selected pixel.
    max_iters : int
        Maximum iterations to prevent infinite loops.

    Returns
    -------
    np.ndarray
        Adversarial image in [0,1] with shape (1,1,28,28).
    """
    # Convert to tensor and enable gradients
    x_orig = torch.from_numpy(x01.copy()).float()
    x_adv = x_orig.clone()

    # Track modified pixels
    num_features = 28 * 28
    search_space = torch.ones(num_features, dtype=torch.bool)

    for iteration in range(max_iters):
        # Check if we've achieved target misclassification
        with torch.no_grad():
            logits = model(mnist_normalize(x_adv))
            pred = logits.argmax(dim=1).item()
            probs = torch.exp(logits)
            target_prob = probs[0, target_class].item()

        if pred == target_class:
            print(f"Success at iteration {iteration}: predicted class = {target_class}")
            break

        # Count actually modified pixels (those different from original)
        x_diff = torch.abs(x_adv - x_orig).view(-1)
        pixels_modified = (x_diff > 1e-6).sum().item()

        if iteration % 100 == 0:
            print(
                f"Iter {iteration}: pred={pred}, target_prob={target_prob:.3f}, pixels_modified={pixels_modified}"
            )

        if pixels_modified >= l0_budget:
            print(
                f"L0 budget exhausted at iteration {iteration}, pixels modified: {pixels_modified}"
            )
            break

        # Compute Jacobian and saliency information
        x_adv_grad = x_adv.clone().requires_grad_(True)
        jacobian = compute_jacobian(model, x_adv_grad)
        saliency_inc, saliency_dec = compute_saliency_map(
            jacobian, target_class, search_space
        )
        max_sal_inc = saliency_inc.max()
        max_sal_dec = saliency_dec.max()

        # Evaluate pairwise saliency in both directions
        pair_inc_p, pair_inc_q, pair_inc_score = compute_pairwise_saliency(
            jacobian, target_class, search_space, "increase"
        )
        pair_dec_p, pair_dec_q, pair_dec_score = compute_pairwise_saliency(
            jacobian, target_class, search_space, "decrease"
        )

        best_pair_indices = (pair_inc_p, pair_inc_q)
        best_pair_score = pair_inc_score
        best_pair_increase = True
        if pair_dec_score > best_pair_score:
            best_pair_indices = (pair_dec_p, pair_dec_q)
            best_pair_score = pair_dec_score
            best_pair_increase = False

        current_mask = x_diff > 1e-6
        budget_remaining = l0_budget - pixels_modified
        required_budget = sum(
            1
            for idx in best_pair_indices
            if idx >= 0 and not bool(current_mask[idx].item())
        )
        use_pair = (
            best_pair_score > 0.0
            and best_pair_indices[0] >= 0
            and required_budget <= budget_remaining
        )

        if use_pair:
            x_adv_flat = x_adv.view(-1)
            step = theta if best_pair_increase else -theta
            for idx in best_pair_indices:
                if idx < 0:
                    continue
                x_adv_flat[idx] = torch.clamp(x_adv_flat[idx] + step, 0.0, 1.0)
                # Prune saturated pixels from the search space immediately so
                # subsequent saliency computations ignore them.
                if x_adv_flat[idx] <= 0.0 or x_adv_flat[idx] >= 1.0:
                    search_space[idx] = False
            x_adv = x_adv_flat.view(1, 1, 28, 28)
            continue

        if max_sal_inc == 0 and max_sal_dec == 0:
            print(f"No valid features at iteration {iteration}")
            break

        x_adv_flat = x_adv.view(-1)
        if max_sal_inc >= max_sal_dec:
            idx = saliency_inc.argmax().item()
            x_adv_flat[idx] = torch.clamp(x_adv_flat[idx] + theta, 0.0, 1.0)
        else:
            idx = saliency_dec.argmax().item()
            x_adv_flat[idx] = torch.clamp(x_adv_flat[idx] - theta, 0.0, 1.0)
        # Prune saturated pixel after single-pixel update
        if x_adv_flat[idx] <= 0.0 or x_adv_flat[idx] >= 1.0:
            search_space[idx] = False
        x_adv = x_adv_flat.view(1, 1, 28, 28)

    return x_adv.detach().cpu().numpy()

# ---------------- Sumbit and Evaluate -----------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default="http://127.0.0.1:8000", help="Server base URL"
    )
    parser.add_argument(
        "--weights",
        default="",
        help="Optional local path to weights; downloads from the server when omitted",
    )
    args = parser.parse_args()

    chall = fetch_challenge(args.host)
    weights_path = args.weights if args.weights else None
    model = load_model(args.host, weights_path)

    # Local sanity prediction on clean image
    x = torch.from_numpy(chall.x01)
    clean_pred = int(torch.argmax(model(mnist_normalize(x)), dim=1).item())
    print(f"Original prediction: {clean_pred}, True label: {chall.original_label}")
    print(f"Target class: {chall.target_class}, L0 budget: {chall.l0_budget}")

    # Perform JSMA attack
    x_adv = jsma_targeted(
        model,
        chall.x01,
        chall.target_class,
        chall.l0_budget,
        theta=1.0,
    )

    # Verify result locally
    adv_pred = int(
        torch.argmax(model(mnist_normalize(torch.from_numpy(x_adv))), dim=1).item()
    )
    l0_used = int(np.sum(np.abs(x_adv - chall.x01) > 1e-6))

    print(
        json.dumps(
            {
                "l0_used": l0_used,
                "adv_pred": adv_pred,
                "target": chall.target_class,
                "clean_pred": clean_pred,
            },
            indent=2,
        )
    )

    # Submit
    x2d = x_adv[0, 0]
    b64 = png_from_x01(x2d)
    r = requests.post(f"{args.host}/submit", json={"image_b64": b64}, timeout=10)
    try:
        r.raise_for_status()
    except Exception:
        print("Server response:", r.text)
        raise
    print("Flag:", r.json().get("flag"))


if __name__ == "__main__":
    main()

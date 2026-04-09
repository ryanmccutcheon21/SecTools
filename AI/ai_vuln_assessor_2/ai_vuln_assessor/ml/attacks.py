from __future__ import annotations
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

def compute_accuracy(model, loader, device):
    correct = total = 0
    confs = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)
            preds = probs.argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
            confs.extend(probs.max(1).values.cpu().numpy().tolist())
    return correct / total if total else 0.0, np.array(confs)

def membership_inference_attack(model, train_loader, test_loader, device, n_samples=2000):
    def _confs(loader, max_n):
        confs = []
        with torch.no_grad():
            for x, _ in loader:
                if len(confs) >= max_n:
                    break
                probs = F.softmax(model(x.to(device)), dim=1)
                confs.extend(probs.max(1).values.cpu().numpy().tolist())
        return np.array(confs[:max_n])
    mem = _confs(train_loader, n_samples)
    non = _confs(test_loader, n_samples)
    n = min(len(mem), len(non))
    all_conf = np.concatenate([mem[:n], non[:n]])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    best_acc, best_t = 0.5, 0.5
    for t in np.percentile(all_conf, np.linspace(0, 100, 300)):
        for direction in [True, False]:
            preds = (all_conf >= t) if direction else (all_conf < t)
            acc = np.mean(preds == labels)
            if acc > best_acc:
                best_acc, best_t = acc, t
    return {"advantage": float(best_acc - 0.5), "best_accuracy": float(best_acc), "threshold": float(best_t), "member_conf": mem[:n].tolist(), "nonmember_conf": non[:n].tolist()}

def fgsm_robustness(model, loader, device, epsilons=None):
    if epsilons is None:
        epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]
    criterion = nn.CrossEntropyLoss()
    robustness = {}
    for eps in epsilons:
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if eps == 0.0:
                adv_x = x
            else:
                x.requires_grad_(True)
                loss = criterion(model(x), y)
                model.zero_grad()
                loss.backward()
                adv_x = (x + eps * x.grad.sign()).detach().clamp(0.0, 1.0)
            with torch.no_grad():
                preds = model(adv_x).argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
        robustness[str(eps)] = correct / total if total else 0.0
    return robustness

def expected_calibration_error(model, loader, device, bins=10):
    confs, corrects = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = F.softmax(model(x), dim=1)
            conf, pred = probs.max(1)
            confs.extend(conf.cpu().numpy().tolist())
            corrects.extend(pred.eq(y).cpu().numpy().astype(float).tolist())
    confs = np.array(confs)
    corrects = np.array(corrects)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        mask = (confs > lo) & (confs <= hi)
        if not mask.any():
            continue
        ece += abs(corrects[mask].mean() - confs[mask].mean()) * mask.mean()
    return float(ece)

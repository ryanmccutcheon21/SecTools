from __future__ import annotations
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None; nn = None; F = None

def compute_accuracy(model, loader, device):
    correct = total = 0; confs=[]; labels=[]; preds_all=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x); probs = F.softmax(out, dim=1); preds = probs.argmax(1)
            correct += preds.eq(y).sum().item(); total += y.size(0)
            confs.extend(probs.max(1).values.cpu().numpy().tolist()); labels.extend(y.cpu().numpy().tolist()); preds_all.extend(preds.cpu().numpy().tolist())
    return (correct/total if total else 0.0), np.array(confs), np.array(labels), np.array(preds_all)

def membership_inference_attack(model, train_loader, test_loader, device, n_samples=2000):
    def _confs(loader, max_n):
        confs=[]
        with torch.no_grad():
            for x,_ in loader:
                if len(confs) >= max_n: break
                x = x.to(device); probs = F.softmax(model(x), dim=1); confs.extend(probs.max(1).values.cpu().numpy().tolist())
        return np.array(confs[:max_n])
    mem = _confs(train_loader, n_samples); non = _confs(test_loader, n_samples); n = min(len(mem), len(non))
    all_conf = np.concatenate([mem[:n], non[:n]]); labels = np.concatenate([np.ones(n), np.zeros(n)])
    thresholds = np.percentile(all_conf, np.linspace(0, 100, 300)); best_acc = 0.5; best_t = 0.5
    for t in thresholds:
        for direction in [True, False]:
            preds = (all_conf >= t) if direction else (all_conf < t); acc = np.mean(preds == labels)
            if acc > best_acc: best_acc = acc; best_t = t
    return {"advantage": float(best_acc-0.5), "best_accuracy": float(best_acc), "threshold": float(best_t), "member_conf_mean": float(mem[:n].mean()), "nonmember_conf_mean": float(non[:n].mean()), "member_conf": mem[:n].tolist(), "nonmember_conf": non[:n].tolist()}

def fgsm_robustness(model, loader, device, epsilons=None):
    if epsilons is None: epsilons = [0.0,0.01,0.05,0.1,0.2,0.3]
    criterion = nn.CrossEntropyLoss(); robustness={}
    for eps in epsilons:
        correct = total = 0
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            if eps == 0.0:
                adv_x = x
            else:
                x.requires_grad_(True); out = model(x); loss = criterion(out,y); model.zero_grad(); loss.backward(); adv_x = (x + eps*x.grad.sign()).detach().clamp(-3.0,3.0)
            with torch.no_grad():
                preds = model(adv_x).argmax(1)
            correct += preds.eq(y).sum().item(); total += y.size(0)
        robustness[str(eps)] = correct/total if total else 0.0
    return robustness

def feature_sensitivity_gini(model, loader, device, n_batches=5):
    vals=[]
    for i,(x,y) in enumerate(loader):
        if i >= n_batches: break
        x,y = x.to(device), y.to(device); x.requires_grad_(True); out = model(x); loss = nn.CrossEntropyLoss()(out,y); model.zero_grad(); loss.backward()
        sal = x.grad.abs().view(x.size(0), -1); sal_sorted = sal.sort(dim=1).values.cpu().detach().numpy(); n = sal_sorted.shape[1]; idx = np.arange(1, n+1)
        gini = ((2*idx - n - 1) * sal_sorted).sum(1) / (n * sal_sorted.sum(1) + 1e-9); vals.extend(gini.tolist())
    return float(np.mean(vals)) if vals else 0.5

def expected_calibration_error(model, loader, device, bins=10):
    confs=[]; corrects=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device); probs = F.softmax(model(x), dim=1); conf, pred = probs.max(1)
            confs.extend(conf.cpu().numpy().tolist()); corrects.extend(pred.eq(y).cpu().numpy().astype(float).tolist())
    confs=np.array(confs); corrects=np.array(corrects); ece=0.0
    for b in range(bins):
        lo=b/bins; hi=(b+1)/bins; mask=(confs>lo)&(confs<=hi)
        if not mask.any(): continue
        acc=corrects[mask].mean(); avg_conf=confs[mask].mean(); ece += abs(acc-avg_conf)*mask.mean()
    return float(ece)

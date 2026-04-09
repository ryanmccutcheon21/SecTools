from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)

def generate_charts(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {}
    mia = result.metrics.get("mia")
    if mia and mia.get("member_conf") and mia.get("nonmember_conf"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(np.array(mia["member_conf"]), bins=40, alpha=0.7, label="Members")
        ax.hist(np.array(mia["nonmember_conf"]), bins=40, alpha=0.7, label="Non-members")
        ax.set_title("MIA Confidence Distribution")
        ax.legend()
        charts["MIA Confidence Distribution"] = _save(fig, out_dir / "mia_confidence_distribution.png")
    fgsm = result.metrics.get("fgsm_robustness")
    if fgsm:
        fig, ax = plt.subplots(figsize=(7, 4))
        eps = [float(k) for k in fgsm.keys()]
        vals = [v * 100 for v in fgsm.values()]
        ax.plot(eps, vals, marker="o")
        ax.set_title("FGSM Robustness")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy (%)")
        charts["FGSM Robustness"] = _save(fig, out_dir / "fgsm_robustness.png")
    art = result.metrics.get("art_attacks", {})
    if art.get("art_available"):
        attacks = art.get("attacks", {})
        names = [k for k, v in attacks.items() if "success_rate" in v]
        if names:
            fig, ax = plt.subplots(figsize=(7, 4))
            vals = [attacks[n]["success_rate"] * 100 for n in names]
            ax.bar(names, vals)
            ax.set_title("ART Evasion Attack Success Rate")
            ax.set_ylabel("Success Rate (%)")
            charts["ART Evasion Attack Success Rate"] = _save(fig, out_dir / "art_evasion_success.png")
    result.charts = charts
    return charts

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG="#0d1117"; CARD="#161b22"; BORDER="#30363d"; GREEN="#39d353"; CYAN="#58a6ff"; AMBER="#f0a500"; RED="#ff4444"; GREY="#8b949e"; WHITE="#e6edf3"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":CARD,"axes.edgecolor":BORDER,"axes.labelcolor":WHITE,"xtick.color":GREY,"ytick.color":GREY,"text.color":WHITE,"grid.color":BORDER,"grid.alpha":0.4})

def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig); return str(path)

def generate_charts(result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True); charts={}
    mia = result.metrics.get("mia")
    if mia and mia.get("member_conf") and mia.get("nonmember_conf"):
        member=np.array(mia["member_conf"]); non=np.array(mia["nonmember_conf"]); fig,ax=plt.subplots(figsize=(8,4)); bins=np.linspace(0,1,40)
        ax.hist(member, bins=bins, alpha=0.7, label="Members", color=CYAN); ax.hist(non, bins=bins, alpha=0.7, label="Non-members", color=AMBER); ax.set_title("MIA Confidence Distribution"); ax.legend()
        charts["MIA Confidence Distribution"] = _save(fig, out_dir/"mia_confidence_distribution.png")
    fgsm = result.metrics.get("fgsm_robustness")
    if fgsm:
        eps=[float(k) for k in fgsm.keys()]; vals=[v*100 for v in fgsm.values()]; fig,ax=plt.subplots(figsize=(7,4)); ax.plot(eps, vals, marker="o", color=CYAN); ax.set_title("FGSM Robustness"); ax.set_xlabel("Epsilon"); ax.set_ylabel("Accuracy (%)"); charts["FGSM Robustness"]=_save(fig, out_dir/"fgsm_robustness.png")
    probe_results=result.probe_results
    if probe_results:
        categories=sorted(set(p.category for p in probe_results)); scores=[]
        for cat in categories:
            subset=[p for p in probe_results if p.category==cat]; frac=sum(1 for p in subset if p.triggered)/max(len(subset),1); scores.append(frac*100)
        fig,ax=plt.subplots(figsize=(8,max(3,len(categories)*0.5))); ax.barh(categories, scores, color=[RED if x>=60 else AMBER if x>=30 else GREEN for x in scores]); ax.set_xlim(0,100); ax.set_title("LLM Probe Trigger Rate by Category"); ax.set_xlabel("Triggered (%)"); charts["LLM Probe Trigger Rate"]=_save(fig, out_dir/"llm_probe_trigger_rate.png")
    coverage=result.coverage
    if coverage:
        labels=list(coverage.keys()); vals=[]
        for item in coverage.values(): vals.append(100 if item.status=="fail" else 50 if item.status=="warn" else 10 if item.status=="pass" else 0)
        fig,ax=plt.subplots(figsize=(10,max(4,len(labels)*0.35))); ax.barh(labels, vals, color=[RED if v==100 else AMBER if v==50 else GREEN if v==10 else GREY for v in vals]); ax.set_title("Coverage Heatmap (Fail/Warn/Pass)"); ax.set_xlabel("Risk Signal"); charts["Coverage Heatmap"]=_save(fig, out_dir/"coverage_heatmap.png")
    result.charts = charts; return charts

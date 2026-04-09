from __future__ import annotations
import hashlib, os
from ..core.models import AssessmentResult, Finding
from ..core.coverage import CoverageRegistry
from ..core.scoring import finding_risk_score
from ..governance.poc import proof_of_concepts_for_refs
from .attacks import compute_accuracy, membership_inference_attack, fgsm_robustness, feature_sensitivity_gini, expected_calibration_error

class MLAssessor:
    def __init__(self, model, train_loader, test_loader, device, result: AssessmentResult, coverage: CoverageRegistry):
        self.model = model.to(device); self.train_loader = train_loader; self.test_loader = test_loader; self.device = device; self.result = result; self.coverage = coverage; self.model.eval()
    def add_finding(self, finding: Finding):
        finding.score = finding_risk_score(finding); self.result.findings.append(finding)
    def assess_overfitting(self):
        train_acc, _, _, _ = compute_accuracy(self.model, self.train_loader, self.device); test_acc, _, _, _ = compute_accuracy(self.model, self.test_loader, self.device); gap = train_acc - test_acc
        self.result.metrics["train_accuracy"] = train_acc; self.result.metrics["test_accuracy"] = test_acc; self.result.metrics["overfitting_gap"] = gap
        sev = "HIGH" if gap > 0.10 else "MEDIUM" if gap > 0.05 else "LOW"
        self.coverage.mark("ML02_POISONING_SIGNAL", "fail" if gap > 0.10 else "warn" if gap > 0.05 else "pass", f"gap={gap:.4f}")
        self.add_finding(Finding("Overfitting / Memorization Gap", sev, ["ML02","ML04","SAIF:Protect data"], f"Train accuracy {train_acc*100:.1f}% vs test accuracy {test_acc*100:.1f}% produced a gap of {gap*100:.1f}%.", evidence=f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, gap={gap:.4f}", remediation="Add stronger regularization, early stopping, dataset review, and retraining validation.", category="Traditional ML", proof_of_concept=proof_of_concepts_for_refs(["ML02","ML04"]), defense_in_depth=["Version datasets and compare train/test distributions each release.","Add approval gates for significant generalization gap increases."], tags=["ml","overfitting","privacy"]))
    def assess_mia(self):
        info = membership_inference_attack(self.model, self.train_loader, self.test_loader, self.device); self.result.metrics["mia"] = info; adv = info["advantage"]
        sev = "HIGH" if adv > 0.15 else "MEDIUM" if adv > 0.05 else "LOW"
        self.coverage.mark("ML04_PRIVACY", "fail" if adv > 0.05 else "pass", f"advantage={adv:.4f}")
        self.add_finding(Finding("Membership Inference Attack", sev, ["ML04","SAIF:Protect data"], f"Threshold-based MIA achieved {info['best_accuracy']*100:.1f}% attacker accuracy.", evidence=f"advantage={adv:.4f}, threshold={info['threshold']:.4f}", remediation="Use stronger regularization, privacy-preserving training for sensitive data, and reduce confidence leakage.", category="Privacy", proof_of_concept=proof_of_concepts_for_refs(["ML04"]), defense_in_depth=["Do not expose raw logits or rich confidence scores to untrusted clients."], tags=["ml","privacy","mia"]))
    def assess_fgsm(self):
        robustness = fgsm_robustness(self.model, self.test_loader, self.device); self.result.metrics["fgsm_robustness"] = robustness; clean = robustness.get("0.0",0.0); at01 = robustness.get("0.1", list(robustness.values())[-1]); drop = clean - at01
        sev = "CRITICAL" if drop > 0.40 else "HIGH" if drop > 0.20 else "MEDIUM" if drop > 0.10 else "LOW"
        self.coverage.mark("ML01_EVASION", "fail" if drop > 0.10 else "pass", f"drop@0.1={drop:.4f}")
        self.add_finding(Finding("FGSM Evasion Robustness", sev, ["ML01","SAIF:Adapt controls to AI"], f"Accuracy dropped from {clean*100:.1f}% to {at01*100:.1f}% at ε=0.1.", evidence=" | ".join(f"eps={k}: {v:.4f}" for k,v in robustness.items()), remediation="Introduce adversarial training, defensive preprocessing, and operational anomaly detection.", category="Traditional ML", proof_of_concept=proof_of_concepts_for_refs(["ML01"]), defense_in_depth=["Add OOD and adversarial monitoring at the serving boundary."], tags=["ml","robustness","evasion"]))
    def assess_feature_sensitivity(self):
        gini = feature_sensitivity_gini(self.model, self.test_loader, self.device); self.result.metrics["feature_sensitivity_gini"] = gini
        sev = "HIGH" if gini > 0.75 else "MEDIUM" if gini > 0.55 else "LOW"
        self.coverage.mark("ML01_EVASION", "warn" if gini > 0.55 else "pass", f"gini={gini:.4f}")
        self.add_finding(Finding("Feature Saliency Concentration", sev, ["ML01","SAIF:Contextualize model behavior"], f"Mean saliency Gini coefficient was {gini:.3f}.", evidence=f"gini={gini:.4f}", remediation="Increase input diversity and reduce reliance on small feature subsets.", category="Traditional ML", defense_in_depth=["Review feature attribution drift in each model release."], tags=["ml","saliency"]))
    def assess_calibration(self):
        ece = expected_calibration_error(self.model, self.test_loader, self.device); self.result.metrics["ece"] = ece
        sev = "MEDIUM" if ece > 0.10 else "LOW"
        self.coverage.mark("ML08_CALIBRATION", "warn" if ece > 0.10 else "pass", f"ece={ece:.4f}")
        self.add_finding(Finding("Calibration / Confidence Reliability", sev, ["ML08","SAIF:Contextualize model behavior"], f"Expected Calibration Error was {ece:.4f}.", evidence=f"ece={ece:.4f}", remediation="Apply temperature scaling or calibration-aware retraining and use abstention thresholds where needed.", category="Traditional ML", defense_in_depth=["Display confidence conservatively in user-facing workflows."], tags=["ml","calibration"]))
    def assess_supply_chain(self, model_path: str | None = None):
        param_count = sum(p.numel() for p in self.model.parameters()); has_dropout = any(m.__class__.__name__.lower().startswith("dropout") for m in self.model.modules()); has_batchnorm = any("batchnorm" in m.__class__.__name__.lower() for m in self.model.modules()); sha256 = None; pickle_flag = False
        if model_path and os.path.exists(model_path):
            h = hashlib.sha256()
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
            sha256 = h.hexdigest(); pickle_flag = model_path.endswith((".pkl",".pickle"))
        self.result.metrics["supply_chain"] = {"param_count":param_count,"has_dropout":has_dropout,"has_batchnorm":has_batchnorm,"sha256":sha256,"pickle_flag":pickle_flag}
        self.coverage.mark("ML06_SUPPLY_CHAIN", "fail" if pickle_flag else "pass", f"pickle_flag={pickle_flag}")
        sev = "CRITICAL" if pickle_flag else "MEDIUM" if (not has_dropout and not has_batchnorm) else "LOW"
        self.add_finding(Finding("Supply-Chain / Serialization Audit", sev, ["ML06","ML10","SAIF:Harden infrastructure"], f"Model has {param_count:,} params. Dropout={has_dropout}, BatchNorm={has_batchnorm}, Pickle={pickle_flag}.", evidence=f"sha256={sha256}, pickle_flag={pickle_flag}", remediation="Prefer safe model formats, verify hashes and provenance, and isolate model loading.", category="Supply Chain", proof_of_concept=proof_of_concepts_for_refs(["ML06"]), defense_in_depth=["Require signed artifacts in the model promotion pipeline."], tags=["ml","supply-chain"]))
    def run_all(self, model_path: str | None = None):
        self.assess_overfitting(); self.assess_mia(); self.assess_fgsm(); self.assess_feature_sensitivity(); self.assess_calibration(); self.assess_supply_chain(model_path)

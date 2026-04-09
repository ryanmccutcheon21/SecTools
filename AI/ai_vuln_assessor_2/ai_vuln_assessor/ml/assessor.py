from __future__ import annotations
from ..core.models import Finding
from ..core.scoring import finding_risk_score
from ..governance.poc import proof_of_concepts_for_refs
from .attacks import compute_accuracy, membership_inference_attack, fgsm_robustness, expected_calibration_error
from .attack_runner import run_art_attacks
from .privacy_runner import evaluate_dp_sgd_capability, evaluate_pate_capability
from .htb_integration import try_load_htb_library

class MLAssessor:
    def __init__(self, model, train_loader, test_loader, device, result, coverage, dataset_name: str = "mnist"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.result = result
        self.coverage = coverage
        self.dataset_name = dataset_name
        self.model.eval()

    def _add_finding(self, finding: Finding):
        finding.score = finding_risk_score(finding)
        self.result.findings.append(finding)

    def assess_baseline(self):
        train_acc, _ = compute_accuracy(self.model, self.train_loader, self.device)
        test_acc, _ = compute_accuracy(self.model, self.test_loader, self.device)
        gap = train_acc - test_acc
        self.result.metrics["train_accuracy"] = train_acc
        self.result.metrics["test_accuracy"] = test_acc
        self.result.metrics["overfitting_gap"] = gap
        self.coverage.mark("ML02", "fail" if gap > 0.10 else "warn" if gap > 0.05 else "pass", f"gap={gap:.4f}")

    def assess_mia(self):
        info = membership_inference_attack(self.model, self.train_loader, self.test_loader, self.device)
        self.result.metrics["mia"] = info
        adv = info["advantage"]
        self.coverage.mark("ML04", "fail" if adv > 0.05 else "pass", f"advantage={adv:.4f}")
        sev = "HIGH" if adv > 0.15 else "MEDIUM" if adv > 0.05 else "LOW"
        self._add_finding(Finding(title="Membership Inference Attack", severity=sev, framework_refs=["ML04"], description=f"Threshold-based MIA achieved {info['best_accuracy']*100:.1f}% attacker accuracy.", evidence=f"advantage={adv:.4f}, threshold={info['threshold']:.4f}", remediation="Reduce confidence leakage and consider privacy-preserving training for sensitive data.", category="Privacy", proof_of_concept=proof_of_concepts_for_refs(["ML04"]), defense_in_depth=["Do not expose rich confidence scores to untrusted users."], tags=["ml", "privacy", "mia"]))

    def assess_fgsm(self):
        robustness = fgsm_robustness(self.model, self.test_loader, self.device)
        self.result.metrics["fgsm_robustness"] = robustness
        clean = robustness.get("0.0", 0.0)
        at01 = robustness.get("0.1", list(robustness.values())[-1])
        drop = clean - at01
        self.coverage.mark("ML01", "fail" if drop > 0.10 else "pass", f"drop@0.1={drop:.4f}")
        sev = "CRITICAL" if drop > 0.40 else "HIGH" if drop > 0.20 else "MEDIUM" if drop > 0.10 else "LOW"
        self._add_finding(Finding(title="FGSM Evasion Robustness", severity=sev, framework_refs=["ML01"], description=f"Accuracy dropped from {clean*100:.1f}% to {at01*100:.1f}% at ε=0.1.", evidence=" | ".join(f"eps={k}: {v:.4f}" for k, v in robustness.items()), remediation="Introduce adversarial training and serving-time anomaly detection.", category="Traditional ML", proof_of_concept=proof_of_concepts_for_refs(["ML01"]), defense_in_depth=["Evaluate robust accuracy on every release."], tags=["ml", "robustness", "fgsm"]))

    def assess_art_evasion(self):
        art_result = run_art_attacks(self.model, self.test_loader, self.dataset_name)
        self.result.metrics["art_attacks"] = art_result
        if not art_result.get("art_available"):
            return
        attacks = art_result.get("attacks", {})
        for name, info in attacks.items():
            if "success_rate" not in info:
                continue
            rate = info["success_rate"]
            sev = "HIGH" if rate > 0.60 else "MEDIUM" if rate > 0.30 else "LOW"
            self.coverage.mark("ML01", "fail" if rate > 0.30 else "warn", f"{name}_success={rate:.4f}")
            self._add_finding(Finding(title=f"{name.title()} Evasion Attack", severity=sev, framework_refs=["ML01"], description=f"{name.title()} produced adversarial examples with success rate {rate*100:.1f}%.", evidence=f"success_rate={rate:.4f}", remediation="Harden robustness with adversarial training, preprocessing, and detection controls.", category="Traditional ML", proof_of_concept=proof_of_concepts_for_refs(["ML01"]), defense_in_depth=["Compare clean vs robust accuracy across multiple attack families."], tags=["ml", "art", name]))

    def assess_calibration(self):
        ece = expected_calibration_error(self.model, self.test_loader, self.device)
        self.result.metrics["ece"] = ece
        self.coverage.mark("ML08", "warn" if ece > 0.10 else "pass", f"ece={ece:.4f}")
        self._add_finding(Finding(title="Calibration / Confidence Reliability", severity="MEDIUM" if ece > 0.10 else "LOW", framework_refs=["ML08"], description=f"Expected Calibration Error was {ece:.4f}.", evidence=f"ece={ece:.4f}", remediation="Apply temperature scaling and use abstention thresholds when confidence is misleading.", category="Traditional ML", tags=["ml", "calibration"]))

    def assess_privacy_defenses(self):
        htb_ok, htb_msg = try_load_htb_library()
        dpsgd = evaluate_dp_sgd_capability()
        pate = evaluate_pate_capability()
        self.result.metrics["htb_ai_library"] = {"available": htb_ok, "message": htb_msg}
        self.result.metrics["dp_sgd"] = dpsgd
        self.result.metrics["pate"] = pate
        self.coverage.mark("PRIVACY_DEFENSE", "pass" if dpsgd.get("available") else "warn", dpsgd.get("message", ""))
        self._add_finding(Finding(title="Privacy Defense Capability Assessment", severity="INFO" if dpsgd.get("available") else "LOW", framework_refs=["PRIVACY_DEFENSE"], description="This run evaluated whether the environment is ready for DP-SGD and PATE-style privacy experiments.", evidence=f"HTB AI Library: {htb_msg} | DP-SGD: {dpsgd.get('message')} | PATE: {pate.get('message')}", remediation="Integrate DP-SGD and PATE experiment workflows where sensitive data protection is required.", category="Privacy", proof_of_concept=proof_of_concepts_for_refs(["PRIVACY_DEFENSE"]), tags=["privacy", "dp-sgd", "pate", "htb-ai-library"]))

    def run_all(self):
        self.assess_baseline()
        self.assess_mia()
        self.assess_fgsm()
        self.assess_art_evasion()
        self.assess_calibration()
        self.assess_privacy_defenses()

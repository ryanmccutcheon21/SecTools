from __future__ import annotations
from ..core.models import Finding, Recommendation

RECOMMENDATION_LIBRARY = {
    "LLM01": Recommendation("LLM01", "Defend against prompt injection", "High", preventive=["Separate trusted instructions from user content server-side.", "Constrain tool and action permissions."], detective=["Log delimiter abuse, translation reframing, and repeated prompt leak attempts."], corrective=["Rotate secrets and patch prompt handling logic."]),
    "LLM02": Recommendation("LLM02", "Treat model output as untrusted input", "High", preventive=["HTML-encode and sanitize output.", "Never execute raw model-built code or SQL."], detective=["Scan outputs for script tags, shell metacharacters, and dangerous patterns."], corrective=["Disable execution sinks until validation exists."]),
    "LLM04": Recommendation("LLM04", "Reduce DoS and cost-amplification risk", "Medium", preventive=["Enforce token budgets, rate limits, and request complexity caps."], detective=["Monitor latency, token use, and burst behavior."], corrective=["Throttle abusive clients and reduce context windows temporarily."]),
    "LLM06": Recommendation("LLM06", "Prevent sensitive data leakage", "High", preventive=["Filter outputs for secrets, credentials, PII, and hidden prompt patterns."], detective=["Scan completions continuously for emails, keys, and prompt fragments."], corrective=["Rotate exposed secrets and harden prompt/data access controls."]),
    "LLM09": Recommendation("LLM09", "Reduce hallucination and unsafe overreliance", "Medium", preventive=["Ground answers with retrieval and encourage uncertainty when evidence is weak."], detective=["Track unsupported citations and fabricated entities."], corrective=["Route high-impact answers through human review."]),
    "ML01": Recommendation("ML01", "Improve adversarial robustness", "High", preventive=["Use adversarial training or stronger preprocessing defenses."], detective=["Monitor for confidence shifts and suspicious inputs."], corrective=["Fallback to conservative decisions when attacks are suspected."]),
    "ML02": Recommendation("ML02", "Protect training data integrity", "High", preventive=["Version and attest datasets.", "Audit label quality and ingestion sources."], detective=["Look for abnormal train/test gaps and drift."], corrective=["Quarantine suspect datasets and retrain from trusted baselines."]),
    "ML04": Recommendation("ML04", "Reduce privacy leakage from training data", "High", preventive=["Apply regularization and reduce confidence leakage."], detective=["Run recurring membership inference checks."], corrective=["Retrain with tighter privacy controls."]),
    "ML06": Recommendation("ML06", "Harden the ML supply chain", "High", preventive=["Prefer safe model formats and verify hashes/provenance."], detective=["Continuously scan artifacts and dependencies."], corrective=["Rebuild from trusted artifacts and isolate loading paths."]),
    "ML08": Recommendation("ML08", "Calibrate model confidence", "Medium", preventive=["Apply temperature scaling and use abstention thresholds."], detective=["Track ECE over time."], corrective=["Retrain or recalibrate misleading models."]),
    "PRIVACY_DEFENSE": Recommendation("PRIVACY_DEFENSE", "Adopt privacy-preserving training where appropriate", "Medium", preventive=["Compare baseline training with DP-SGD for sensitive datasets.", "Evaluate PATE for teacher/student privacy-sensitive workflows."], detective=["Track epsilon/delta and privacy/utility regressions across releases."], corrective=["Switch sensitive workloads to privacy-preserving training pipelines."]),
}

def collect_recommendations(findings: list[Finding]) -> list[Recommendation]:
    refs = set()
    for f in findings:
        for ref in f.framework_refs:
            key = ref.split(":")[0]
            if key in RECOMMENDATION_LIBRARY:
                refs.add(key)
    return [RECOMMENDATION_LIBRARY[r] for r in sorted(refs)]

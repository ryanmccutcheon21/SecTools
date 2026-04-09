from __future__ import annotations
from ..core.models import Finding, Recommendation

RECOMMENDATION_LIBRARY = {
    "LLM01": Recommendation("LLM01", "Defend against prompt injection", "High",
        preventive=["Separate trusted instructions from user content server-side.", "Constrain tool/action permissions."],
        detective=["Log override phrases, delimiter abuse, and repeated prompt leak attempts."],
        corrective=["Rotate secrets and patch prompt handling logic."]),
    "LLM02": Recommendation("LLM02", "Treat model output as untrusted input", "High",
        preventive=["HTML-encode and sanitize output.", "Never execute raw model-built code, SQL, or shell."],
        detective=["Scan outputs for script tags, shell metacharacters, and dangerous patterns."],
        corrective=["Disable downstream execution until validation layers exist."]),
    "LLM04": Recommendation("LLM04", "Reduce DoS and cost-amplification risk", "Medium",
        preventive=["Enforce token budgets, rate limits, and request complexity caps."],
        detective=["Monitor latency, token use, and burst behavior."],
        corrective=["Throttle abusive clients and reduce context windows temporarily."]),
    "LLM06": Recommendation("LLM06", "Prevent sensitive data leakage", "High",
        preventive=["Filter outputs for secrets, credentials, PII, and hidden prompt patterns."],
        detective=["Scan completions continuously for emails, keys, and prompt fragments."],
        corrective=["Rotate exposed secrets and harden prompt/data access controls."]),
    "LLM09": Recommendation("LLM09", "Reduce hallucination and unsafe overreliance", "Medium",
        preventive=["Ground answers with retrieval and encourage uncertainty when evidence is weak."],
        detective=["Track unsupported citations and fabricated entities."],
        corrective=["Route high-impact answers through human review."]),
    "ML01": Recommendation("ML01", "Improve adversarial robustness", "High",
        preventive=["Use adversarial training or stronger preprocessing defenses."],
        detective=["Monitor for confidence shifts and suspicious inputs."],
        corrective=["Fallback to conservative decisions when attacks are suspected."]),
    "ML02": Recommendation("ML02", "Protect training data integrity", "High",
        preventive=["Version and attest datasets.", "Audit label quality and ingestion sources."],
        detective=["Look for abnormal train/test gaps and drift."],
        corrective=["Quarantine suspect datasets and retrain from trusted baselines."]),
    "ML04": Recommendation("ML04", "Reduce privacy leakage from training data", "High",
        preventive=["Apply regularization and reduce confidence leakage."],
        detective=["Run recurring membership inference checks."],
        corrective=["Retrain with tighter privacy controls."]),
    "ML06": Recommendation("ML06", "Harden the ML supply chain", "High",
        preventive=["Prefer safe model formats and verify hashes/provenance."],
        detective=["Continuously scan artifacts and dependencies."],
        corrective=["Rebuild from trusted artifacts and isolate loading paths."]),
    "ML08": Recommendation("ML08", "Calibrate model confidence", "Medium",
        preventive=["Apply temperature scaling and use abstention thresholds."],
        detective=["Track ECE over time."],
        corrective=["Retrain or recalibrate misleading models."]),
    "SAIF": Recommendation("SAIF", "Adopt defense-in-depth for AI systems", "High",
        preventive=["Harden model, data, runtime, and interface layers together.", "Map controls to SAIF pillars in CI/CD."],
        detective=["Extend centralized logging and anomaly detection to AI workflows."],
        corrective=["Create AI-specific incident response and rollback procedures."]),
}

def collect_recommendations(findings: list[Finding]) -> list[Recommendation]:
    refs = set()
    for f in findings:
        for ref in f.framework_refs:
            refs.add("SAIF" if ref.startswith("SAIF") else ref.split(":")[0])
    return [RECOMMENDATION_LIBRARY[r] for r in sorted(refs) if r in RECOMMENDATION_LIBRARY]

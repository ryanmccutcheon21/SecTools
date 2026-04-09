from __future__ import annotations
from .models import Finding

SEVERITY_WEIGHT = {"CRITICAL":1.0,"HIGH":0.75,"MEDIUM":0.5,"LOW":0.25,"INFO":0.1,"PASS":0.0}

def finding_risk_score(f: Finding) -> float:
    base = SEVERITY_WEIGHT.get(f.severity.upper(), 0.25)
    return round(min(100.0, base * f.confidence * f.exploitability * f.repeatability * f.business_impact * 100.0), 2)

def overall_risk_score(findings: list[Finding]) -> float:
    if not findings:
        return 0.0
    return round(sum(finding_risk_score(f) for f in findings) / len(findings), 2)

def severity_counts(findings: list[Finding]) -> dict:
    keys = ["CRITICAL","HIGH","MEDIUM","LOW","INFO","PASS"]
    return {k: sum(1 for f in findings if f.severity == k) for k in keys}

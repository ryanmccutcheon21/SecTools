from __future__ import annotations
from .models import CoverageItem

DEFAULT_COVERAGE = {
    "LLM01": CoverageItem("LLM01", "Prompt injection and jailbreak resistance", "Generative AI", ["LLM01", "SAIF:Contextualize model behavior"]),
    "LLM02": CoverageItem("LLM02", "Insecure output handling", "Generative AI", ["LLM02", "SAIF:Secure system interfaces"]),
    "LLM04": CoverageItem("LLM04", "Model denial of service", "Generative AI", ["LLM04", "SAIF:Automate defenses"]),
    "LLM06": CoverageItem("LLM06", "Sensitive info disclosure and prompt leakage", "Generative AI", ["LLM06", "LLM10", "SAIF:Protect data"]),
    "LLM09": CoverageItem("LLM09", "Hallucination and overreliance", "Generative AI", ["LLM09", "SAIF:Contextualize model behavior"]),
    "ML01": CoverageItem("ML01", "Input manipulation and adversarial evasion", "Traditional ML", ["ML01", "SAIF:Adapt controls to AI"]),
    "ML02": CoverageItem("ML02", "Poisoning and training integrity signals", "Traditional ML", ["ML02", "SAIF:Protect data"]),
    "ML04": CoverageItem("ML04", "Membership inference and privacy leakage", "Privacy", ["ML04", "SAIF:Protect data"]),
    "ML06": CoverageItem("ML06", "Supply-chain and unsafe serialization", "Supply Chain", ["ML06", "ML10", "SAIF:Harden infrastructure"]),
    "ML08": CoverageItem("ML08", "Reliability, confidence, and calibration", "Traditional ML", ["ML08", "SAIF:Contextualize model behavior"]),
    "PRIVACY_DEFENSE": CoverageItem("PRIVACY_DEFENSE", "Differential privacy and privacy defenses", "Privacy", ["DP-SGD", "PATE", "SAIF:Protect data"]),
}

class CoverageRegistry:
    def __init__(self):
        self.items = {k: CoverageItem(**vars(v)) for k, v in DEFAULT_COVERAGE.items()}

    def mark(self, item_id: str, status: str, evidence: str = ""):
        if item_id in self.items:
            self.items[item_id].executed = True
            self.items[item_id].status = status
            if evidence:
                self.items[item_id].evidence = evidence

    def to_dict(self):
        return self.items

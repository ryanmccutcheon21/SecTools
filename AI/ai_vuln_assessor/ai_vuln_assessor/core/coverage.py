from __future__ import annotations
from .models import CoverageItem

DEFAULT_COVERAGE = {
    "LLM01_PROMPT_INJECTION": CoverageItem("LLM01_PROMPT_INJECTION", "Prompt injection and jailbreak resistance", "Generative AI", ["LLM01", "SAIF:Contextualize model behavior"], mapped_components=["prompt handling", "policy enforcement"]),
    "LLM02_OUTPUT_HANDLING": CoverageItem("LLM02_OUTPUT_HANDLING", "Insecure output handling", "Generative AI", ["LLM02", "SAIF:Secure system interfaces"], mapped_components=["rendering", "execution sinks"]),
    "LLM03_TRAINING_DATA": CoverageItem("LLM03_TRAINING_DATA", "Training data poisoning and data trust", "Generative AI", ["LLM03", "SAIF:Extend detection and response"], mapped_components=["datasets", "RAG corpus"]),
    "LLM04_DOS": CoverageItem("LLM04_DOS", "Model denial of service", "Generative AI", ["LLM04", "SAIF:Automate defenses"], mapped_components=["rate limits", "context windows"]),
    "LLM05_SUPPLY_CHAIN": CoverageItem("LLM05_SUPPLY_CHAIN", "Supply chain vulnerabilities", "Generative AI", ["LLM05", "SAIF:Harden infrastructure"], mapped_components=["models", "dependencies"]),
    "LLM06_DATA_LEAKAGE": CoverageItem("LLM06_DATA_LEAKAGE", "Sensitive information disclosure and prompt leakage", "Generative AI", ["LLM06", "LLM07", "LLM10", "SAIF:Protect data"], mapped_components=["memory", "hidden prompts", "training outputs"]),
    "LLM09_HALLUCINATION": CoverageItem("LLM09_HALLUCINATION", "Hallucination and overreliance", "Generative AI", ["LLM09", "SAIF:Contextualize model behavior"], mapped_components=["grounding", "user trust"]),
    "ML01_EVASION": CoverageItem("ML01_EVASION", "Input manipulation and adversarial evasion", "Traditional ML", ["ML01", "SAIF:Adapt controls to AI"], mapped_components=["robustness", "validation"]),
    "ML02_POISONING_SIGNAL": CoverageItem("ML02_POISONING_SIGNAL", "Poisoning and training integrity signals", "Traditional ML", ["ML02", "SAIF:Protect data"], mapped_components=["training data", "labels"]),
    "ML03_INVERSION": CoverageItem("ML03_INVERSION", "Model inversion and reconstruction risk", "Traditional ML", ["ML03", "SAIF:Protect data"], mapped_components=["privacy", "sensitive features"]),
    "ML04_PRIVACY": CoverageItem("ML04_PRIVACY", "Membership inference and privacy leakage", "Privacy", ["ML04", "SAIF:Protect data"], mapped_components=["training privacy", "confidence leakage"]),
    "ML06_SUPPLY_CHAIN": CoverageItem("ML06_SUPPLY_CHAIN", "Supply-chain and unsafe serialization", "Supply Chain", ["ML06", "ML10", "SAIF:Harden infrastructure"], mapped_components=["serialization", "dependencies"]),
    "ML08_CALIBRATION": CoverageItem("ML08_CALIBRATION", "Reliability, confidence, and calibration", "Traditional ML", ["ML08", "SAIF:Contextualize model behavior"], mapped_components=["confidence", "safety gating"]),
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

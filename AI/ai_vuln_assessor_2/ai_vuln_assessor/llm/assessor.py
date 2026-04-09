from __future__ import annotations
import time
from ..core.models import Finding, ProbeResult
from ..core.scoring import finding_risk_score
from ..governance.poc import proof_of_concepts_for_refs
from .profiles import PROFILES
from .probes import PROBE_LIBRARY
from .detectors import KeywordDetector, InstructionLeakDetector, UnsafeCodeDetector, RegexSecretDetector, HallucinationDetector

def build_detector(name: str, args: dict):
    if name == "keyword": return KeywordDetector(**args)
    if name == "instruction_leak": return InstructionLeakDetector()
    if name == "unsafe_code": return UnsafeCodeDetector()
    if name == "regex_secret": return RegexSecretDetector(**args)
    if name == "hallucination": return HallucinationDetector()
    raise ValueError(name)

class LLMAssessor:
    def __init__(self, adapter, result, coverage, system_prompt: str = ""):
        self.adapter = adapter
        self.result = result
        self.coverage = coverage
        self.system_prompt = system_prompt

    def run_profile(self, profile_name: str = "general_llm", delay: float = 0.5):
        for pid in PROFILES[profile_name]:
            probe = PROBE_LIBRARY[pid]
            detector = build_detector(probe.detector_name, probe.detector_args)
            t0 = time.time()
            raw = self.adapter.send(probe.prompt or "", system_prompt=self.system_prompt, history=None)
            text = raw.get("text", "")
            detection = detector.evaluate(text, {"probe_id": pid})
            for ref in probe.framework_refs:
                key = ref.split(":")[0]
                if key in self.coverage.items:
                    self.coverage.mark(key, "fail" if detection.triggered else "pass", detection.reason)
            self.result.probe_results.append(ProbeResult(probe.id, probe.category, probe.framework_refs, True, detection.triggered, probe.severity, detection.confidence, text[:400], detection.reason, round(time.time()-t0, 2), raw))
            if detection.triggered:
                finding = Finding(title=f"[LLM] {probe.category}: {probe.id}", severity=probe.severity, framework_refs=probe.framework_refs, description=f"Probe '{probe.id}' triggered.", evidence=f"{detection.reason} | Response: {text[:400]}", remediation=probe.remediation, category=probe.category, proof_of_concept=proof_of_concepts_for_refs(probe.framework_refs), defense_in_depth=["Centralize logging for repeated attack attempts.", "Add application-layer validation."], tags=["llm", probe.category.lower().replace(" ", "-")])
                finding.score = finding_risk_score(finding)
                self.result.findings.append(finding)
            time.sleep(delay)

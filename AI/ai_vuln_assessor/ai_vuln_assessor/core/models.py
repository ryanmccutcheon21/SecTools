from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import datetime

@dataclass
class Finding:
    title: str
    severity: str
    framework_refs: List[str]
    description: str
    evidence: str = ""
    remediation: str = ""
    score: float = 0.0
    confidence: float = 1.0
    exploitability: float = 1.0
    repeatability: float = 1.0
    business_impact: float = 1.0
    category: str = "General"
    affected_asset: str = ""
    proof_of_concept: List[str] = field(default_factory=list)
    defense_in_depth: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    status: str = "open"

@dataclass
class ProbeResult:
    probe_id: str
    category: str
    framework_refs: List[str]
    executed: bool
    triggered: bool
    severity: str
    confidence: float
    response_excerpt: str
    reason: str
    latency_s: float
    raw: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoverageItem:
    id: str
    title: str
    category: str
    framework_refs: List[str]
    executed: bool = False
    status: str = "not_tested"
    evidence: str = ""
    mapped_components: List[str] = field(default_factory=list)

@dataclass
class Recommendation:
    framework_ref: str
    title: str
    priority: str
    preventive: List[str] = field(default_factory=list)
    detective: List[str] = field(default_factory=list)
    corrective: List[str] = field(default_factory=list)

@dataclass
class AssessmentResult:
    target_name: str
    mode: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    findings: List[Finding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    charts: Dict[str, str] = field(default_factory=dict)
    probe_results: List[ProbeResult] = field(default_factory=list)
    coverage: Dict[str, CoverageItem] = field(default_factory=dict)
    target_meta: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Recommendation] = field(default_factory=list)
    executive_summary: Dict[str, Any] = field(default_factory=dict)

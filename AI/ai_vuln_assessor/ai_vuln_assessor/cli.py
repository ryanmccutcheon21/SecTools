from __future__ import annotations
import argparse
from pathlib import Path
from .core.models import AssessmentResult
from .core.coverage import CoverageRegistry
from .core.reporting import write_html_report, write_json_report, write_markdown_report, write_csv_findings
from .adapters.factory import build_adapter
from .llm.assessor import LLMAssessor
from .ml.assessor import MLAssessor
from .ml.model_loader import load_model
from .ml.data import load_fashion_mnist
from .charts import generate_charts
from .governance.mapping import collect_recommendations
from .core.scoring import overall_risk_score, severity_counts

def build_parser():
    p = argparse.ArgumentParser(description="Enterprise AI / ML / LLM Vulnerability Assessment Tool")
    p.add_argument("--mode", choices=["llm","ml","both"], required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-name", default="AI Target")
    p.add_argument("--adapter", choices=["openai_chat","anthropic_messages","openai_compatible","simple_http_json","simple_http_form"], default="openai_compatible")
    p.add_argument("--profile", choices=["prompt_injection","system_prompt_leak","privacy","general_llm","full"], default="general_llm")
    p.add_argument("--model-name", default="unknown-model")
    p.add_argument("--api-key", default="")
    p.add_argument("--base-url", default="")
    p.add_argument("--system-prompt", default="")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--probe-delay", type=float, default=0.5)
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--http-method", choices=["GET","POST"], default="POST")
    p.add_argument("--json-prompt-key", default="prompt")
    p.add_argument("--form-prompt-key", default="query")
    p.add_argument("--response-json-path", default="response")
    p.add_argument("--model", default="")
    p.add_argument("--dataset", choices=["fashion_mnist"], default="fashion_mnist")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-test", type=int, default=2000)
    return p

def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    result = AssessmentResult(target_name=args.target_name, mode=args.mode)
    coverage = CoverageRegistry(); result.coverage = coverage.to_dict()
    result.target_meta = {"adapter":getattr(args,"adapter",None),"profile":getattr(args,"profile",None),"base_url":getattr(args,"base_url",None),"model_name":getattr(args,"model_name",None),"dataset":getattr(args,"dataset",None)}
    if args.mode in ("llm","both"):
        adapter = build_adapter(args)
        LLMAssessor(adapter=adapter, result=result, coverage=coverage, system_prompt=args.system_prompt, max_turns=args.max_turns).run_profile(args.profile, delay=args.probe_delay)
    if args.mode in ("ml","both"):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required for ML mode.") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(args.model or None)
        train_loader, test_loader = load_fashion_mnist(batch_size=args.batch_size, n_train=args.n_train, n_test=args.n_test)
        MLAssessor(model=model, train_loader=train_loader, test_loader=test_loader, device=device, result=result, coverage=coverage).run_all(args.model or None)
    result.coverage = coverage.to_dict()
    result.recommendations = collect_recommendations(result.findings)
    result.executive_summary = {"overall_risk_score": overall_risk_score(result.findings), "severity_counts": severity_counts(result.findings), "total_findings": len(result.findings), "coverage_executed": sum(1 for x in result.coverage.values() if x.executed), "coverage_total": len(result.coverage)}
    generate_charts(result, out_dir/"charts")
    write_json_report(result, out_dir/"report.json")
    write_html_report(result, out_dir/"report.html")
    write_markdown_report(result, out_dir/"report.md")
    write_csv_findings(result, out_dir/"findings.csv")
    print(f"[OK] Reports written to {out_dir}")

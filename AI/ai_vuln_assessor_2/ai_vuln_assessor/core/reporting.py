from __future__ import annotations
import base64, csv, json
from pathlib import Path
from .scoring import overall_risk_score, severity_counts

def write_json_report(result, out_path: Path):
    payload = {"target_name": result.target_name, "mode": result.mode, "timestamp": result.timestamp, "metrics": result.metrics, "target_meta": result.target_meta, "charts": result.charts, "executive_summary": result.executive_summary, "findings": [vars(f) for f in result.findings], "probe_results": [vars(p) for p in result.probe_results], "coverage": {k: vars(v) for k, v in result.coverage.items()}, "recommendations": [vars(r) for r in result.recommendations]}
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def write_csv_findings(result, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["severity", "framework_refs", "title", "category", "description", "evidence", "remediation", "score", "tags"])
        for x in result.findings:
            w.writerow([x.severity, ", ".join(x.framework_refs), x.title, x.category, x.description, x.evidence, x.remediation, x.score, ", ".join(x.tags)])

def write_markdown_report(result, out_path: Path):
    counts = severity_counts(result.findings)
    lines = [f"# AI Vulnerability Assessment — {result.target_name}", "", f"- Mode: `{result.mode}`", f"- Generated: `{result.timestamp}`", f"- Overall risk score: `{overall_risk_score(result.findings)}`", "", "## Executive Summary", ""]
    for k, v in counts.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Findings", ""]
    for f in result.findings:
        lines += [f"### {f.title}", f"- Severity: **{f.severity}**", f"- Framework: {', '.join(f.framework_refs)}", f"- Category: {f.category}", f"- Description: {f.description}"]
        if f.evidence: lines.append(f"- Evidence: {f.evidence}")
        if f.remediation: lines.append(f"- Remediation: {f.remediation}")
        if f.proof_of_concept:
            lines.append("- Proof of concept:")
            for p in f.proof_of_concept: lines.append(f"  - `{p}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")

def _embed_img(path: str) -> str:
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""

def write_html_report(result, out_path: Path):
    counts = severity_counts(result.findings)
    overall = overall_risk_score(result.findings)
    chart_html = []
    for name, path in result.charts.items():
        b64 = _embed_img(path)
        if b64: chart_html.append(f'<div class="card"><h3>{name}</h3><img src="data:image/png;base64,{b64}" alt="{name}"></div>')
    finding_rows = []
    for f in result.findings:
        poc_html = "<br>".join(f"<code>{p}</code>" for p in f.proof_of_concept[:4])
        did_html = "".join(f"<li>{d}</li>" for d in f.defense_in_depth[:4])
        finding_rows.append(f"<tr><td>{f.severity}</td><td>{', '.join(f.framework_refs)}</td><td>{f.title}</td><td>{f.category}</td><td>{f.description}</td><td>{f.evidence}</td><td>{f.remediation}</td><td>{poc_html}</td><td><ul>{did_html}</ul></td><td>{f.score:.1f}</td></tr>")
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>AI Vulnerability Assessment - {result.target_name}</title><style>body{{font-family:Arial,sans-serif;margin:0;padding:0}}header{{padding:24px;background:#eee}}main{{padding:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}.card{{border:1px solid #ccc;border-radius:8px;padding:16px}}table{{width:100%;border-collapse:collapse;margin-top:20px}}th,td{{border:1px solid #ccc;padding:8px;vertical-align:top}}img{{width:100%}}code{{background:#f4f4f4;padding:2px 4px}}</style></head><body><header><h1>AI Vulnerability Assessment</h1><div><strong>Target:</strong> {result.target_name}</div><div><strong>Mode:</strong> {result.mode}</div><div><strong>Generated:</strong> {result.timestamp}</div></header><main><section class="grid"><div class="card"><h3>Overall Risk Score</h3><div style="font-size:28px">{overall:.1f}</div></div><div class="card"><h3>Critical</h3><div style="font-size:28px">{counts['CRITICAL']}</div></div><div class="card"><h3>High</h3><div style="font-size:28px">{counts['HIGH']}</div></div><div class="card"><h3>Medium</h3><div style="font-size:28px">{counts['MEDIUM']}</div></div></section><section><h2>Charts</h2><div class="grid">{''.join(chart_html)}</div></section><section><h2>Findings</h2><table><thead><tr><th>Severity</th><th>Framework</th><th>Title</th><th>Category</th><th>Description</th><th>Evidence</th><th>Remediation</th><th>PoC</th><th>Defense in Depth</th><th>Score</th></tr></thead><tbody>{''.join(finding_rows)}</tbody></table></section></main></body></html>'''
    out_path.write_text(html, encoding="utf-8")

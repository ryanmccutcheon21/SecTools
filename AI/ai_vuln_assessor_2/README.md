# AI Vulnerability Assessor vNext

This is a vNext framework that combines:
- your reporting/orchestration framework
- HTB AI Library integration hooks
- ART-backed ML evasion attack runners
- Opacus-backed DP-SGD evaluation hooks
- OWASP ML / OWASP LLM / Google SAIF mapping
- enterprise-style report outputs

## Install

```bash
pip install -r requirements.txt
```

## Example: LLM target

```bash
python ai_vuln_assessor.py \
  --mode llm \
  --target-name "HTB Prompt Leak 1" \
  --adapter simple_http_form \
  --base-url http://127.0.0.1:5000/prompt_inject/prompt_leak_1 \
  --form-prompt-key query \
  --profile prompt_injection \
  --out-dir ./report_llm
```

## Example: ML target on CIFAR-10 with ART attacks

```bash
python ai_vuln_assessor.py \
  --mode ml \
  --target-name "CIFAR CNN" \
  --dataset cifar10 \
  --ml-profile full \
  --model ./model.pt \
  --out-dir ./report_ml
```

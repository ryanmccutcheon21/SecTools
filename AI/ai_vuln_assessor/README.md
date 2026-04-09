# AI Vulnerability Assessor Enterprise

Enterprise-grade AI / ML / LLM assessment framework with:
- OWASP LLM / OWASP ML / Google SAIF mapping
- HTML / JSON / Markdown / CSV reports
- Proof-of-concept payloads
- Remediation recommendations
- Adapters for OpenAI, Anthropic, OpenAI-compatible, JSON HTTP, and form HTTP
- LLM profiles and ML checks

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python ai_vuln_assessor.py --mode llm --target-name "Assistant" --adapter openai_compatible --base-url http://localhost:11434/v1/chat/completions --api-key ollama --model-name llama3.1 --system-prompt "You are a helpful assistant." --profile full --out-dir ./report_llm
```

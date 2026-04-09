from __future__ import annotations
import time, requests
from typing import Dict, List, Optional
from .base import TargetAdapter

class OpenAICompatibleAdapter(TargetAdapter):
    def __init__(self, base_url: str, model_name: str, api_key: str, max_tokens: int = 512):
        self.base_url = base_url; self.model_name = model_name; self.api_key = api_key; self.max_tokens = max_tokens
    def send(self, message: str, system_prompt: str = "", history: Optional[List[Dict[str, str]]] = None) -> Dict:
        msgs = []
        if system_prompt:
            msgs.append({"role":"system","content":system_prompt})
        if history:
            msgs.extend(history)
        msgs.append({"role":"user","content":message})
        t0 = time.time()
        resp = requests.post(self.base_url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type":"application/json"}, json={"model":self.model_name,"messages":msgs,"max_tokens":self.max_tokens,"temperature":0.0}, timeout=45)
        text, raw, err = "", {}, None
        try:
            raw = resp.json(); text = raw.get("choices",[{}])[0].get("message",{}).get("content","")
        except Exception as exc:
            err = str(exc); text = resp.text
        return {"text": text, "raw": raw if isinstance(raw, dict) else {}, "status_code": resp.status_code, "latency_s": time.time()-t0, "error": err}

from __future__ import annotations
import time
from typing import Dict, List, Optional
from .base import TargetAdapter

class OpenAIChatAdapter(TargetAdapter):
    def __init__(self, model_name: str, api_key: str, base_url: str | None = None, max_tokens: int = 512):
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model_name = model_name
        self.max_tokens = max_tokens
    def send(self, message: str, system_prompt: str = "", history: Optional[List[Dict[str, str]]] = None) -> Dict:
        msgs = []
        if system_prompt:
            msgs.append({"role":"system","content":system_prompt})
        if history:
            msgs.extend(history)
        msgs.append({"role":"user","content":message})
        t0 = time.time()
        resp = self.client.chat.completions.create(model=self.model_name, messages=msgs, max_tokens=self.max_tokens, temperature=0.0)
        return {"text": resp.choices[0].message.content or "", "raw": resp.model_dump() if hasattr(resp,"model_dump") else {}, "status_code": 200, "latency_s": time.time()-t0, "error": None}

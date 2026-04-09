from __future__ import annotations
import time
from typing import Dict, List, Optional
from .base import TargetAdapter

class AnthropicMessagesAdapter(TargetAdapter):
    def __init__(self, model_name: str, api_key: str, max_tokens: int = 512):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
    def send(self, message: str, system_prompt: str = "", history: Optional[List[Dict[str, str]]] = None) -> Dict:
        msgs = history[:] if history else []
        msgs.append({"role":"user","content":message})
        t0 = time.time()
        resp = self.client.messages.create(model=self.model_name, system=system_prompt, messages=msgs, max_tokens=self.max_tokens)
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return {"text": text, "raw": resp.model_dump() if hasattr(resp,"model_dump") else {}, "status_code": 200, "latency_s": time.time()-t0, "error": None}

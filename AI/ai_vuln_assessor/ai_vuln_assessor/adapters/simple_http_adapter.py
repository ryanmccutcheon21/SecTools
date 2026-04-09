from __future__ import annotations
import time, requests
from typing import Any, Dict, List, Optional
from .base import TargetAdapter

def _extract_json_path(data: Any, path: str) -> str:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            cur = cur[int(part)]
        else:
            return ""
    return cur if isinstance(cur, str) else str(cur)

class SimpleHTTPJSONAdapter(TargetAdapter):
    def __init__(self, base_url: str, prompt_key: str = "prompt", response_json_path: str = "response", method: str = "POST"):
        self.base_url = base_url; self.prompt_key = prompt_key; self.response_json_path = response_json_path; self.method = method.upper()
    def send(self, message: str, system_prompt: str = "", history: Optional[List[Dict[str, str]]] = None) -> Dict:
        payload = {self.prompt_key: message}
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if history:
            payload["history"] = history
        t0 = time.time()
        resp = requests.get(self.base_url, params=payload, timeout=45) if self.method == "GET" else requests.post(self.base_url, json=payload, timeout=45)
        text, raw, err = "", {}, None
        try:
            raw = resp.json(); text = _extract_json_path(raw, self.response_json_path)
        except Exception as exc:
            err = str(exc); text = resp.text
        return {"text": text, "raw": raw if isinstance(raw, dict) else {}, "status_code": resp.status_code, "latency_s": time.time()-t0, "error": err}

class SimpleHTTPFormAdapter(TargetAdapter):
    def __init__(self, base_url: str, prompt_key: str = "query", response_json_path: str | None = None, method: str = "POST"):
        self.base_url = base_url; self.prompt_key = prompt_key; self.response_json_path = response_json_path; self.method = method.upper()
    def send(self, message: str, system_prompt: str = "", history: Optional[List[Dict[str, str]]] = None) -> Dict:
        payload = {self.prompt_key: message}
        if system_prompt:
            payload["system_prompt"] = system_prompt
        t0 = time.time()
        resp = requests.get(self.base_url, params=payload, timeout=45) if self.method == "GET" else requests.post(self.base_url, data=payload, timeout=45)
        text, raw, err = resp.text, {}, None
        if self.response_json_path:
            try:
                raw = resp.json(); text = _extract_json_path(raw, self.response_json_path)
            except Exception as exc:
                err = str(exc)
        return {"text": text, "raw": raw if isinstance(raw, dict) else {}, "status_code": resp.status_code, "latency_s": time.time()-t0, "error": err}

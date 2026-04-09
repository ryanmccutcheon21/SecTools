from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class TargetAdapter(ABC):
    @abstractmethod
    def send(self, message: str, system_prompt: str = "", history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        raise NotImplementedError

from __future__ import annotations
from pathlib import Path
try:
    import torch
    import torch.nn as nn
except Exception:
    torch=None; nn=None

class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.25), nn.Flatten(), nn.Linear(32*14*14,128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128,n_classes),
        )
    def forward(self, x): return self.net(x)

def load_model(model_path: str | None):
    if model_path:
        p = Path(model_path)
        if p.suffix == ".pt":
            obj = torch.load(model_path, map_location="cpu")
            if isinstance(obj, nn.Module): return obj
            raise ValueError("Loaded .pt file is not a torch.nn.Module.")
    return SimpleCNN()

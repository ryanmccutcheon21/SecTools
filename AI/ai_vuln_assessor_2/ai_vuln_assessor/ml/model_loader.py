from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int = 1, image_size: int = 28, n_classes: int = 10):
        super().__init__()
        pooled = image_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(32 * pooled * pooled, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_model(model_path: str | None, in_channels: int = 1, image_size: int = 28, n_classes: int = 10):
    if model_path and torch is not None:
        obj = torch.load(model_path, map_location="cpu")
        if isinstance(obj, nn.Module):
            return obj
        raise ValueError("Loaded .pt file is not a torch.nn.Module.")
    return SimpleCNN(in_channels=in_channels, image_size=image_size, n_classes=n_classes)

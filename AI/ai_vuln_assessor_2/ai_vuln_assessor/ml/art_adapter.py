from __future__ import annotations

def build_art_classifier(model, input_shape, nb_classes, clip_values=(0.0, 1.0)):
    try:
        import torch.nn as nn
        import torch.optim as optim
        from art.estimators.classification import PyTorchClassifier
    except Exception as exc:
        raise RuntimeError(f"ART not available: {exc}") from exc
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return PyTorchClassifier(model=model, loss=loss, optimizer=optimizer, input_shape=input_shape, nb_classes=nb_classes, clip_values=clip_values)

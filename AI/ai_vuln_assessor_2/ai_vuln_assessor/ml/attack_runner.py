from __future__ import annotations
import numpy as np

def _loader_to_numpy(loader, max_batches=4):
    xs, ys = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        xs.append(x.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def run_art_attacks(model, test_loader, dataset_name: str):
    try:
        from art.attacks.evasion import DeepFool, ElasticNet, SaliencyMapMethod
    except Exception as exc:
        return {"art_available": False, "error": str(exc), "attacks": {}}
    x_test, y_test = _loader_to_numpy(test_loader)
    channels = x_test.shape[1]
    image_size = x_test.shape[2]
    from .art_adapter import build_art_classifier
    classifier = build_art_classifier(model, input_shape=(channels, image_size, image_size), nb_classes=10, clip_values=(0.0, 1.0))
    attacks = {}
    for name, ctor in [("deepfool", DeepFool), ("elasticnet", lambda classifier: ElasticNet(classifier=classifier, max_iter=5)), ("jsma", SaliencyMapMethod)]:
        try:
            attack = ctor(classifier=classifier) if name != "elasticnet" else ctor(classifier)
            x_adv = attack.generate(x=x_test)
            preds = classifier.predict(x_adv).argmax(axis=1)
            attacks[name] = {"success_rate": float(np.mean(preds != y_test))}
        except Exception as exc:
            attacks[name] = {"error": str(exc)}
    return {"art_available": True, "attacks": attacks}

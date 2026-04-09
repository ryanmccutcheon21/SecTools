from __future__ import annotations

def try_load_htb_library():
    try:
        import htb_ai_library  # noqa: F401
        return True, "htb-ai-library import succeeded"
    except Exception as exc:
        return False, f"htb-ai-library unavailable: {exc}"

def dataset_info(dataset_name: str) -> dict:
    info = {
        "mnist": {"channels": 1, "classes": 10, "image_size": 28},
        "cifar10": {"channels": 3, "classes": 10, "image_size": 32},
        "adult": {"channels": None, "classes": 2, "image_size": None},
    }
    return info.get(dataset_name, {})

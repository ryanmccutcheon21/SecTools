from __future__ import annotations

def evaluate_dp_sgd_capability():
    try:
        import opacus  # noqa: F401
        return {"available": True, "message": "Opacus import succeeded"}
    except Exception as exc:
        return {"available": False, "message": str(exc)}

def evaluate_pate_capability():
    return {"available": False, "message": "PATE is included here as a reporting/evaluation stub. Add your preferred teacher/student implementation."}

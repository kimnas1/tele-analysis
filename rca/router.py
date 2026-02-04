from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Tuple

from rca.vendor import import_module_from_path


def load_reasoned_predict(data_dir: str) -> Any:
    candidates = [
        os.path.join(data_dir, "reasoned_predict.py"),
        os.path.join("tools", "reasoned_predict.py"),
        "reasoned_predict.py",
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("Could not locate reasoned_predict.py (tried data_dir and ./tools).")
    return import_module_from_path("reasoned_predict", path)


def load_router_module() -> Any:
    path = os.path.join("tools", "router.py")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing tools/router.py")
    return import_module_from_path("exp_router", path)


def route_questions_phase2(phase2_questions: Dict[str, str], *, rp: Any) -> Dict[str, dict]:
    router_mod = load_router_module()
    out: Dict[str, dict] = {}
    for bid, q in phase2_questions.items():
        dec, metrics, raw = router_mod.route_question(q, rp=rp)  # type: ignore[attr-defined]
        out[bid] = {
            "base_id": bid,
            "route": dec.route,
            "task_type": dec.task_type,
            "metrics_status": dec.metrics_status,
            "reason": dec.reason,
        }
    return out

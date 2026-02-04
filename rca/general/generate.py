from __future__ import annotations

import os
import sys
from typing import Any, Dict

from rca.util import log
from rca.vendor import import_module_from_path


def run_vote3_general(
    *,
    questions_csv: str,
    submission_template_csv: str,
    submission_column: str,
    out_submission_csv: str,
    out_audit_csv: str,
    out_traces_jsonl: str,
    vote3_cfg: dict,
    backend_cfg: dict,
) -> None:
    tools_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    mod = import_module_from_path(
        "vote3_traces_general", os.path.join("tools", "generate_submission_rcaood_vote3_traces.py")
    )

    cfg: Dict[str, Any] = dict(mod.CONFIG)  # type: ignore[attr-defined]
    cfg["RUN_MODE"] = "GENERAL"
    cfg["question_paths_override"] = [questions_csv]
    cfg["submission_template_csv"] = submission_template_csv
    cfg["submission_column"] = submission_column
    cfg["out_submission_csv"] = out_submission_csv
    cfg["out_audit_csv"] = out_audit_csv
    cfg["out_traces_jsonl"] = out_traces_jsonl

    for k in ("strategies", "finalize_if_unboxed", "max_tokens", "temperature", "trace_prefix", "prefix"):
        if k in vote3_cfg:
            cfg[k] = vote3_cfg[k]

    # Force HF backend
    cfg["backend"] = "hf"

    env_model = os.environ.get("HF_MODEL")
    if env_model:
        cfg["hf_model"] = env_model
    elif backend_cfg.get("hf_model"):
        cfg["hf_model"] = backend_cfg["hf_model"]
    
    if backend_cfg.get("dtype"):
        cfg["dtype"] = backend_cfg["dtype"]

    mod.CONFIG.clear()  # type: ignore[attr-defined]
    mod.CONFIG.update(cfg)  # type: ignore[attr-defined]
    log(f"GENERAL vote3: running on {questions_csv}")
    mod.main()  # type: ignore[attr-defined]

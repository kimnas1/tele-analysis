from __future__ import annotations

import importlib.util
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RouteDecision:
    route: str  # "RULES" | "LLM"
    task_type: str  # "RCA" | "RCA_OOD" | "GENERAL"
    metrics_status: str  # "ok" | "missingCols" | "emptyTables" | "missingCols" | "skipped" | ...
    reason: str


def strip_question_wrapper(text: str) -> str:
    """If input is already a prompted blob, extract the inner <question>...</question>."""
    s = str(text)
    start = s.find("<question>")
    end = s.rfind("</question>")
    if start == -1 or end == -1 or end <= start:
        return s
    inner = s[start + len("<question>") : end]
    return inner.strip("\n")


def looks_like_rca_blob(text: str) -> bool:
    """
    Cheap pre-check to avoid running heavy parsing on clearly OOD questions.

    We treat it as RCA-like if it contains:
    - the two table section markers (or close variants)
    - at least one pipe '|' (table delimiter)
    - a throughput hint
    """
    s = str(text)
    if "|" not in s:
        return False
    low = s.lower()
    if "user plane drive test data" not in low:
        return False
    if "parameters data" not in low:
        return False
    if "throughput" not in low:
        return False
    return True


def looks_like_rca_task(text: str) -> bool:
    """
    Detect an RCA classification task even when the telelog tables aren't present.

    We treat the input as RCA_OOD if:
    - it clearly references the C1..C8 label set, OR
    - it looks like an RCA prompt with drive-test + engineering parameters but uses a
      different answer schema (e.g., options A..I).
    """
    s = str(text)
    up = s.upper()
    labels = set(re.findall(r"\bC([1-8])\b", up))
    if len(labels) >= 6:
        return True
    if "C1..C8" in up or "C1-C8" in up:
        return True
    if re.search(r"\bC1\s*(?:\.\.|-|TO)\s*C8\b", up):
        return True
    low = s.lower()
    if (
        "drive test" in low
        and "engineering" in low
        and "throughput" in low
        and "root cause" in low
        and re.search(r"(?m)^\s*A\s*:", s)
        and re.search(r"(?m)^\s*I\s*:", s)
    ):
        return True
    return False


def load_reasoned_predict_module() -> Any:
    """
    Load tools/reasoned_predict.py as a module (Kaggle-friendly).

    Supports:
    - /kaggle/input/rca-math/reasoned_predict.py (dataset)
    - ./tools/reasoned_predict.py (working dir)
    """
    candidates: list[str] = [
        "/kaggle/input/rca-math/reasoned_predict.py",
    ]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "reasoned_predict.py"))
    except NameError:
        pass
    cwd = os.getcwd()
    candidates.append(os.path.join(cwd, "tools", "reasoned_predict.py"))
    candidates.append(os.path.join(cwd, "reasoned_predict.py"))

    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise RuntimeError("Unable to locate reasoned_predict.py (tried kaggle path and cwd/tools)")

    spec = importlib.util.spec_from_file_location("reasoned_predict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def route_question(question_text: str, *, rp: Any) -> tuple[RouteDecision, Optional[Any], str]:
    """
    Decide whether to use deterministic rules or call an LLM.

    Returns:
      - RouteDecision
      - metrics (or None)
      - raw_question (str) used for parsing
    """
    raw = strip_question_wrapper(question_text)

    if not looks_like_rca_blob(raw):
        if looks_like_rca_task(raw):
            return (
                RouteDecision(
                    route="LLM",
                    task_type="RCA_OOD",
                    metrics_status="skipped",
                    reason="no_tables_but_mentions_C_labels",
                ),
                None,
                raw,
            )
        return (
            RouteDecision(
                route="LLM",
                task_type="GENERAL",
                metrics_status="skipped",
                reason="no_rca_markers",
            ),
            None,
            raw,
        )

    try:
        metrics, status = rp.compute_metrics(raw)
    except Exception:
        metrics, status = None, "exception"
    if status == "ok" and metrics is not None:
        return (
            RouteDecision(
                route="RULES",
                task_type="RCA",
                metrics_status=status,
                reason="metrics_ok",
            ),
            metrics,
            raw,
        )

    # RCA-like text, but our deterministic extractor couldn't compute the expected metrics.
    return (
        RouteDecision(
            route="LLM",
            task_type="RCA_OOD",
            metrics_status=status,
            reason=f"rca_like_but_metrics_{status}",
        ),
        metrics,
        raw,
    )

#!/usr/bin/env python3
"""
Generate a grounded RCA_RULE trace corpus (no label selection by the LLM).

Pipeline (per base ID):
  1) Deterministic rule engine (tools/reasoned_predict.py):
       - compute DerivedMetrics on low-throughput window
       - choose fixed_label in {C1..C8}
       - compute RuleChecks booleans + TriggeredRule
       => EvidencePack (cached as JSONL)
  2) LLM writes a short *justification trace* for the fixed label (grounded):
       - We verify the boxed answer equals fixed_label
       - We verify the Evidence line is copied exactly
       - We verify no new key=value fields are invented
       - We verify no new numeric values are introduced (except allowed constants)
     Retry up to MAX_ATTEMPTS. If still failing, emit a deterministic fallback trace.

Backends:
  - Local HF/Transformers:            BACKEND="HF_LOCAL"

Notes:
  - This script is meant to be Kaggle-friendly (edit CONFIG below; no argparse).
  - For smoke tests, set LIMIT_BASE_IDS to a small number.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict
from typing import Any, Optional


# =============================================================================
# ⚙️  EASY CONFIGURATION - CHANGE THESE VALUES
# =============================================================================

# Dataset: "TRAIN" is recommended for corpus generation (has ground truth).
RUN_MODE = "TRAIN"  # "TRAIN" | "PHASE1"

# How many base IDs to process (None = all). Used for smoke tests.
LIMIT_BASE_IDS: Optional[int] = None  # None = all 2400

# Determinism / reproducibility
SEED = 42
SHUFFLE_IDS = False

# LLM backend
# - Use "HF_LOCAL" to run a local HF model (Kaggle/H100)
# - Use "NONE" for smoke tests (always deterministic fallback; no tokens)
BACKEND = "HF_LOCAL"  # "HF_LOCAL" | "NONE"


# HF local model (Kaggle/H100). Imports torch/transformers only if used.
HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_DTYPE = os.environ.get("HF_DTYPE", "bfloat16")  # "bfloat16" | "float16"
HF_DEVICE_MAP = os.environ.get("HF_DEVICE_MAP", "auto")

# LLM generation
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 0.95
DO_SAMPLE = False

# Attempts: 4 LLM attempts, then deterministic fallback on the 5th "attempt".
MAX_ATTEMPTS = 4

# Output files
OUT_DIR = "tools/rule_trace_corpus"
SCHEMA_VERSION = "v3"
EVIDENCE_CACHE_JSONL = os.path.join(OUT_DIR, f"evidence_{RUN_MODE.lower()}_{SCHEMA_VERSION}.jsonl")
TRACES_JSONL = os.path.join(OUT_DIR, f"traces_{RUN_MODE.lower()}_{SCHEMA_VERSION}.jsonl")


# =============================================================================
# Paths (repo-local or Kaggle)
# =============================================================================


DATA_DIR_CANDIDATES = [
    "/kaggle/input/rca-math",
    "/kaggle/input/rca-5g",
    "data",
]


def pick_data_dir() -> str:
    for p in DATA_DIR_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return "data"


DATA_DIR = pick_data_dir()

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
PHASE1_CSV = os.path.join(DATA_DIR, "phase_1_test.csv")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_reproducible(seed: int) -> None:
    random.seed(seed)


# =============================================================================
# Rule engine loading
# =============================================================================


def load_reasoned_predict_module() -> Any:
    candidates: list[str] = [
        os.path.join(DATA_DIR, "reasoned_predict.py"),
        "/kaggle/input/rca-math/reasoned_predict.py",
        "/kaggle/input/rca-5g/reasoned_predict.py",
        os.path.join("tools", "reasoned_predict.py"),
        os.path.join("reasoned_predict.py"),
    ]
    path = next((p for p in candidates if p and os.path.exists(p)), None)
    if not path:
        raise RuntimeError("Unable to locate reasoned_predict.py (tried DATA_DIR, Kaggle, ./tools).")
    spec = importlib.util.spec_from_file_location("reasoned_predict", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def strip_question_wrapper(text: str) -> str:
    s = str(text)
    start = s.find("<question>")
    end = s.rfind("</question>")
    if start == -1 or end == -1 or end <= start:
        return s
    inner = s[start + len("<question>") : end]
    return inner.strip("\n")


# =============================================================================
# Evidence pack (deterministic, LLM justifies fixed label)
# =============================================================================


def _r(x: Any, ndigits: int) -> Any:
    try:
        if x is None:
            return None
        return round(float(x), ndigits)
    except Exception:
        return None


def build_derived_metrics_dict(metrics: Any, status: str) -> dict[str, Any]:
    if metrics is None:
        return {
            "metricsStatus": status,
            "nRows": None,
            "lowRows": None,
            "speedMax": None,
            "avgRB_low": None,
            "handovers": None,
            "maxDist_m": None,
            "mod30Frac": None,
            "sameGnbCloseFrac": None,
            "diffGnbCloseFrac": None,
            "beyondBeamRows": None,
            "beyondBeamDenom": None,
            "rsrpMedLow": None,
            "tiltMedLow": None,
            "tiltMaxLow": None,
            "c1StrongPcis": [],
        }
    return {
        "metricsStatus": status,
        "nRows": int(getattr(metrics, "n_rows", 0)),
        "lowRows": int(getattr(metrics, "n_low", 0)),
        "speedMax": _r(getattr(metrics, "speed_max", None), 2),
        "avgRB_low": _r(getattr(metrics, "avg_rb_low", None), 3),
        "handovers": int(getattr(metrics, "handovers", 0)),
        "maxDist_m": _r(getattr(metrics, "max_dist_m", 0.0), 3),
        "mod30Frac": _r(getattr(metrics, "mod30_frac", 0.0), 6),
        "sameGnbCloseFrac": _r(getattr(metrics, "same_gnb_close_frac", 0.0), 6),
        "diffGnbCloseFrac": _r(getattr(metrics, "diff_gnb_close_frac", 0.0), 6),
        "beyondBeamRows": int(getattr(metrics, "beyond_beam_rows", 0)),
        "beyondBeamDenom": int(getattr(metrics, "beyond_beam_denom", 0)),
        "rsrpMedLow": _r(getattr(metrics, "rsrp_med_low", None), 2),
        "tiltMedLow": _r(getattr(metrics, "tilt_med_low", None), 1),
        "tiltMaxLow": _r(getattr(metrics, "tilt_max_low", None), 1),
        "c1StrongPcis": list(getattr(metrics, "c1_strong_pcis", []) or []),
    }


def infer_triggered_rule(fixed_label: str, rule_reason: str) -> str:
    rr = rule_reason or ""
    if "decision=C7" in rr:
        return "R1_speedMax_gt_40"
    if "decision=C8" in rr:
        return "R2_avgRB_low_lt_160"
    if "decision=C2" in rr:
        return "R3_maxDist_m_gt_1000"
    if "decision=C6" in rr:
        return "R4_mod30Frac_ge_0.75"
    if "decision=C5" in rr:
        return "R5_handovers_ge_2"
    if "decision=C4" in rr:
        return "R6_diffGnbCloseFrac_eq_1"
    if "decision=C3" in rr:
        return "R8_sameGnbCloseFrac_eq_1"
    if "decision=C1(weak+beam-edge, override C4)" in rr:
        return "R6_override_C1_weak_beam_edge"
    if "decision=C1(confident downtilt" in rr:
        return "R7_c1StrongPcis_nonempty"
    if "decision=C1(all low rows beyond far-edge" in rr:
        return "R8_override_C1_allLowBeyond"
    if "decision=C1(extreme downtilt" in rr:
        return "R8_override_C1_extremeDowntilt"
    if "decision=C1(weak+beam-edge, block C3)" in rr:
        return "R8_override_C1_weak_beam_edge"
    if "decision=C1(UE beyond far-edge)" in rr:
        return "R9_beyondBeamRows_gt_0"
    if "decision=C1(fallback" in rr or "fallback" in rr:
        return "R10_fallback"
    return f"RULES_{fixed_label or 'C1'}_unknown"


def build_rule_checks_dict(metrics: Any, status: str, rp: Any) -> dict[str, Any]:
    if metrics is None:
        return {"metricsStatus": status}

    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))
    rsrp_thr = float(getattr(rp, "C3_BLOCK_IF_BEYOND_AND_RSRP_LE", -87.0))

    m = metrics

    def gt(val: Optional[float], thr: float) -> Optional[bool]:
        return None if val is None else bool(val > thr)

    def lt(val: Optional[float], thr: float) -> Optional[bool]:
        return None if val is None else bool(val < thr)

    def le(val: Optional[float], thr: float) -> Optional[bool]:
        return None if val is None else bool(val <= thr)

    def eq1(val: float) -> bool:
        return abs(float(val) - 1.0) < 1e-9

    r1 = gt(getattr(m, "speed_max", None), speed_thr)
    r2 = lt(getattr(m, "avg_rb_low", None), rb_thr)
    r3 = bool(getattr(m, "max_dist_m", 0.0) > dist_thr)
    r4 = bool(getattr(m, "mod30_frac", 0.0) >= mod30_thr)
    r5 = bool(getattr(m, "handovers", 0) >= ho_thr)
    r6 = bool(getattr(m, "diff_gnb_close_frac", 0.0) >= diff_thr - 1e-9)

    same_eq_1 = eq1(float(getattr(m, "same_gnb_close_frac", 0.0)))
    beyond_rows = int(getattr(m, "beyond_beam_rows", 0))
    beyond_denom = int(getattr(m, "beyond_beam_denom", 0))
    low_rows = int(getattr(m, "n_low", 0))

    r6_override: Optional[bool] = False
    if r6 and same_eq_1 and beyond_rows > 0:
        r6_override = le(getattr(m, "rsrp_med_low", None), rsrp_thr)

    c1_strong = list(getattr(m, "c1_strong_pcis", []) or [])
    r7 = bool(len(c1_strong) > 0)

    r8 = bool(same_eq_1)
    r8_all_beyond = bool(beyond_denom > 0 and beyond_rows == beyond_denom == low_rows)

    r8_extreme_tilt: Optional[bool] = False
    if beyond_rows > 0:
        tilt_med = getattr(m, "tilt_med_low", None)
        r8_extreme_tilt = None if tilt_med is None else bool(float(tilt_med) >= 30.0)

    r8_weak_edge: Optional[bool] = False
    if beyond_rows > 0:
        r8_weak_edge = le(getattr(m, "rsrp_med_low", None), rsrp_thr)

    r9 = bool(beyond_rows > 0)

    return {
        "metricsStatus": status,
        "R1_speedMax_gt_40": r1,
        "R2_avgRB_low_lt_160": r2,
        "R3_maxDist_m_gt_1000": r3,
        "R4_mod30Frac_ge_0.75": r4,
        "R5_handovers_ge_2": r5,
        "R6_diffGnbCloseFrac_eq_1": r6,
        "R6_override_C1_weak_beam_edge": r6_override,
        "R7_c1StrongPcis_nonempty": r7,
        "R8_sameGnbCloseFrac_eq_1": r8,
        "R8_override_C1_allLowBeyond": r8_all_beyond,
        "R8_override_C1_extremeDowntilt": r8_extreme_tilt,
        "R8_override_C1_weak_beam_edge": r8_weak_edge,
        "R9_beyondBeamRows_gt_0": r9,
    }


def _fmt_num(x: Any, ndigits: int) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def _fmt_thr(x: Any) -> str:
    try:
        xf = float(x)
        if abs(xf - int(xf)) < 1e-9:
            return str(int(xf))
        # keep up to 2 decimals for thresholds used in this rubric
        return f"{xf:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)


def build_triggered_rule_line(
    *,
    fixed_label: str,
    triggered_rule: str,
    derived: dict[str, Any],
    rp: Any,
) -> str:
    # Prefer the simplest human-facing condition with a clear threshold.
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    if fixed_label == "C7":
        v = derived.get("speedMax")
        return f"TriggeredRule: speedMax > {_fmt_thr(speed_thr)} (speedMax={_fmt_num(v,1)}, threshold={_fmt_thr(speed_thr)})"
    if fixed_label == "C8":
        v = derived.get("avgRB_low")
        return f"TriggeredRule: avgRB_low < {_fmt_thr(rb_thr)} (avgRB_low={_fmt_num(v,1)}, threshold={_fmt_thr(rb_thr)})"
    if fixed_label == "C2":
        v = derived.get("maxDist_m")
        return f"TriggeredRule: maxDist_m > {_fmt_thr(dist_thr)} (maxDist_m={_fmt_num(v,1)}, threshold={_fmt_thr(dist_thr)})"
    if fixed_label == "C6":
        v = derived.get("mod30Frac")
        return f"TriggeredRule: mod30Frac >= {_fmt_thr(mod30_thr)} (mod30Frac={_fmt_num(v,2)}, threshold={_fmt_thr(mod30_thr)})"
    if fixed_label == "C5":
        v = derived.get("handovers")
        return f"TriggeredRule: handovers >= {_fmt_thr(ho_thr)} (handovers={v}, threshold={_fmt_thr(ho_thr)})"
    if fixed_label == "C4":
        v = derived.get("diffGnbCloseFrac")
        return f"TriggeredRule: diffGnbCloseFrac == {_fmt_thr(diff_thr)} (diffGnbCloseFrac={_fmt_num(v,2)}, threshold={_fmt_thr(diff_thr)})"
    if fixed_label == "C3":
        v = derived.get("sameGnbCloseFrac")
        return f"TriggeredRule: sameGnbCloseFrac == 1.0 (sameGnbCloseFrac={_fmt_num(v,2)}, threshold=1.0)"
    if fixed_label == "C1":
        # Keep it simple (works for most C1 outcomes): beyondBeamRows > 0.
        v = derived.get("beyondBeamRows")
        return f"TriggeredRule: beyondBeamRows > 0 (beyondBeamRows={v}, threshold=0)"

    return f"TriggeredRule: {triggered_rule or 'missing'} (threshold=missing)"


def build_evidence_bullets(*, fixed_label: str, derived: dict[str, Any]) -> list[str]:
    # Target the 4-line "evidence-first" style while ensuring the triggering metric is present.
    base = ["speedMax", "avgRB_low", "maxDist_m", "handovers"]
    trigger_key = {
        "C7": "speedMax",
        "C8": "avgRB_low",
        "C2": "maxDist_m",
        "C6": "mod30Frac",
        "C5": "handovers",
        "C4": "diffGnbCloseFrac",
        "C3": "sameGnbCloseFrac",
        "C1": "beyondBeamRows",
    }.get(fixed_label, "avgRB_low")

    keys = list(base)
    if trigger_key not in keys:
        keys[-1] = trigger_key

    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        v = derived.get(k)
        if k in {"speedMax"}:
            out.append(f"- {k}={_fmt_num(v,1)}")
        elif k in {"avgRB_low"}:
            out.append(f"- {k}={_fmt_num(v,1)}")
        elif k in {"maxDist_m"}:
            out.append(f"- {k}={_fmt_num(v,1)}")
        elif k in {"handovers", "beyondBeamRows"}:
            out.append(f"- {k}={v if v is not None else 'NA'}")
        elif k in {"mod30Frac", "sameGnbCloseFrac", "diffGnbCloseFrac"}:
            out.append(f"- {k}={_fmt_num(v,2)}")
        else:
            out.append(f"- {k}={v if v is not None else 'NA'}")
    return out


def build_rejection_bullets(
    *,
    fixed_label: str,
    triggered_rule: str,
    derived: dict[str, Any],
    rule_checks: dict[str, Any],
    rp: Any,
) -> list[str]:
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    # Canonical conditions
    specs = [
        ("C7", "speedMax", _fmt_num(derived.get("speedMax"), 1), f"speedMax>{_fmt_thr(speed_thr)}", _fmt_thr(speed_thr), "R1_speedMax_gt_40"),
        ("C8", "avgRB_low", _fmt_num(derived.get("avgRB_low"), 1), f"avgRB_low<{_fmt_thr(rb_thr)}", _fmt_thr(rb_thr), "R2_avgRB_low_lt_160"),
        ("C2", "maxDist_m", _fmt_num(derived.get("maxDist_m"), 1), f"maxDist_m>{_fmt_thr(dist_thr)}", _fmt_thr(dist_thr), "R3_maxDist_m_gt_1000"),
        ("C6", "mod30Frac", _fmt_num(derived.get("mod30Frac"), 2), f"mod30Frac>={_fmt_thr(mod30_thr)}", _fmt_thr(mod30_thr), "R4_mod30Frac_ge_0.75"),
        ("C5", "handovers", str(derived.get("handovers") if derived.get("handovers") is not None else "NA"), f"handovers>={_fmt_thr(ho_thr)}", _fmt_thr(ho_thr), "R5_handovers_ge_2"),
        ("C4", "diffGnbCloseFrac", _fmt_num(derived.get("diffGnbCloseFrac"), 2), f"diffGnbCloseFrac=={_fmt_thr(diff_thr)}", _fmt_thr(diff_thr), "R6_diffGnbCloseFrac_eq_1"),
        ("C3", "sameGnbCloseFrac", _fmt_num(derived.get("sameGnbCloseFrac"), 2), "sameGnbCloseFrac==1.0", "1.0", "R8_sameGnbCloseFrac_eq_1"),
        ("C1", "beyondBeamRows", str(derived.get("beyondBeamRows") if derived.get("beyondBeamRows") is not None else "NA"), "beyondBeamRows>0", "0", "R9_beyondBeamRows_gt_0"),
    ]

    out: list[str] = []
    for lbl, metric, mval, cond, thr, check_key in specs:
        if lbl == fixed_label:
            continue
        cond_val = rule_checks.get(check_key)
        if cond_val is True:
            out.append(
                f"- {lbl} not selected because although {metric}={mval} meets {cond} (threshold={thr}), a higher-priority rule fired (TriggeredRule: {triggered_rule})."
            )
        else:
            out.append(
                f"- {lbl} rejected because {metric}={mval} does not meet {cond} (threshold={thr})"
            )
    return out


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Prompts
# =============================================================================


PROMPT_JUSTIFY = """You are a staff 5G RAN engineer writing a grounded RCA justification.

You are NOT selecting the label. A deterministic rule-engine already selected the correct label.

Hard rules (must follow):
- The final label is FIXED: __FIXED_LABEL__. Do NOT change it.
- Do NOT introduce ANY numeric value that is not already present in TriggeredRule/Evidence/Rejections below.
- Do NOT invent new metrics/fields. Do NOT add any new key=value pairs besides those already present below.
- LAST line must be EXACTLY: FinalAnswer: \\boxed{__FIXED_LABEL__}

TriggeredRule line (copy EXACTLY; do not edit):
__TRIGGERED_RULE_LINE__

Evidence block (copy EXACTLY; do not edit):
Evidence:
__EVIDENCE_BULLETS__

Rejections block (copy EXACTLY; do not edit):
Rejections:
__REJECTION_BULLETS__

TriggeredRule: __TRIGGERED_RULE__
RuleChecks (booleans; already computed):
__RULE_CHECKS_JSON__
DerivedMetrics (JSON; already computed):
__DERIVED_METRICS_JSON__

Write a short explanation that justifies why __FIXED_LABEL__ is correct.

Output format (STRICT):
__TRIGGERED_RULE_LINE__

Evidence:
__EVIDENCE_BULLETS__

Rejections:
__REJECTION_BULLETS__

Decision:
- <1 short bullet, no new numbers, no new key=value>
FinalAnswer: \\boxed{__FIXED_LABEL__}
"""


PROMPT_REPAIR = """You wrote an invalid trace. Fix it.

You are NOT selecting the label. The FIXED label is: __FIXED_LABEL__

Verifier errors:
__ERRORS__

Hard rules (must follow):
- TriggeredRule/Evidence/Rejections blocks must appear EXACTLY as given.
- Do NOT introduce ANY numeric value not already present in those blocks.
- Do NOT add new key=value pairs besides those blocks.
- LAST line must be EXACTLY: FinalAnswer: \\boxed{__FIXED_LABEL__}

TriggeredRule line (copy EXACTLY; do not edit):
__TRIGGERED_RULE_LINE__

Evidence block (copy EXACTLY; do not edit):
Evidence:
__EVIDENCE_BULLETS__

Rejections block (copy EXACTLY; do not edit):
Rejections:
__REJECTION_BULLETS__

Output format (STRICT):
__TRIGGERED_RULE_LINE__

Evidence:
__EVIDENCE_BULLETS__

Rejections:
__REJECTION_BULLETS__

Decision:
- ...
FinalAnswer: \\boxed{__FIXED_LABEL__}
"""


# =============================================================================
# Verifier
# =============================================================================


BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
BOX_RE_ESCAPED = re.compile(r"\\\\boxed\{([^}]*)\}")
LABEL_RE = re.compile(r"\bC[1-8]\b")
# Numbers not adjacent to letters/underscores (so we ignore identifiers like R1_*_40).
NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
KV_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=")


def last_box_value(text: str) -> Optional[str]:
    matches = list(BOX_RE.finditer(text))
    if not matches:
        matches = list(BOX_RE_ESCAPED.finditer(text))
    if not matches:
        return None
    for m in reversed(matches):
        v = m.group(1).strip()
        if v:
            return v
    return ""


def verify_trace(
    *,
    trace: str,
    fixed_label: str,
    triggered_rule_line: str,
    evidence_bullets: list[str],
    rejection_bullets: list[str],
    allowed_kv_keys: set[str],
    allowed_constants: set[str],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    t = trace or ""

    # Required blocks must be copied exactly.
    if triggered_rule_line not in t:
        errors.append("Missing or modified TriggeredRule line (must copy exactly).")

    for ln in evidence_bullets:
        if ln not in t:
            errors.append("Missing or modified Evidence bullet (must copy exactly).")
            break

    for ln in rejection_bullets:
        if ln not in t:
            errors.append("Missing or modified Rejections bullet (must copy exactly).")
            break

    # Boxed answer must match.
    boxed = last_box_value(t)
    if boxed is None:
        errors.append("Missing \\boxed{...} answer.")
    elif boxed == "":
        errors.append("Empty \\boxed{}.")
    elif boxed.strip() != fixed_label:
        errors.append(f"Boxed answer mismatch: got {boxed!r}, expected {fixed_label!r}.")

    # Must not add key=value fields beyond the Evidence line.
    kv_keys = set(KV_RE.findall(t))
    # 'Evidence' itself can be matched by KV_RE if model writes "Evidence=..."
    kv_keys.discard("Evidence")
    extra = sorted([k for k in kv_keys if k not in allowed_kv_keys])
    if extra:
        errors.append(f"Invented key=value fields: {extra[:12]}{'...' if len(extra) > 12 else ''}")

    # Numeric grounding: only numbers present in Evidence line or allowed constants.
    # We ignore numbers that are part of identifiers via NUM_RE.
    allowed_text = "\n".join([triggered_rule_line, *evidence_bullets, *rejection_bullets])
    nums = NUM_RE.findall(t)
    for n in nums:
        if n in allowed_constants:
            continue
        if n and n in allowed_text:
            continue
        errors.append(f"Un-grounded number mentioned: {n}")
        break

    return (len(errors) == 0), errors


def deterministic_fallback_trace(
    fixed_label: str,
    triggered_rule_line: str,
    evidence_bullets: list[str],
    rejection_bullets: list[str],
) -> str:
    return "\n".join(
        [
            triggered_rule_line,
            "",
            "Evidence:",
            *evidence_bullets,
            "",
            "Rejections:",
            *rejection_bullets,
            "",
            "Decision:",
            f"- TriggeredRule indicates {fixed_label} is selected by the rubric.",
            f"FinalAnswer: \\boxed{{{fixed_label}}}",
        ]
    )


# =============================================================================
# LLM backends
# =============================================================================


class LLMBackend:
    def generate(self, prompt: str, *, seed: int) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class NoneBackend(LLMBackend):
    def generate(self, prompt: str, *, seed: int) -> str:
        raise RuntimeError("BACKEND=NONE does not generate; should never be called.")


class HFLocalBackend(LLMBackend):
    def __init__(self, *, model_name: str, dtype: str, device_map: str):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HF_LOCAL backend requires torch+transformers installed (run on Kaggle/H100)."
            ) from e

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
        self._torch = torch
        self.tok = tok
        self.model = model

    def generate(self, prompt: str, *, seed: int) -> str:
        torch = self._torch
        # Best-effort determinism.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if hasattr(self.tok, "apply_chat_template"):
            try:
                chat = self.tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                chat = prompt
        else:
            chat = prompt

        inputs = self.tok([chat], return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=int(MAX_NEW_TOKENS),
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
            do_sample=bool(DO_SAMPLE),
            num_beams=1,
            repetition_penalty=1.05,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # Try to return only the assistant portion if possible.
        return text.strip()


def build_backend() -> LLMBackend:
    if BACKEND == "NONE":
        return NoneBackend()
    if BACKEND == "HF_LOCAL":
        return HFLocalBackend(model_name=HF_MODEL_NAME, dtype=HF_DTYPE, device_map=HF_DEVICE_MAP)
    raise ValueError(f"Unknown BACKEND={BACKEND!r}")


# =============================================================================
# I/O (JSONL cache)
# =============================================================================


def load_jsonl_by_id(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["ID"])] = obj
    return out


def main() -> None:
    set_reproducible(SEED)
    mod = load_reasoned_predict_module()
    rp = mod.ReasonedPredictor()

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load questions
    input_csv = TRAIN_CSV if RUN_MODE == "TRAIN" else PHASE1_CSV
    log(f"Loading questions from {input_csv}")
    with open(input_csv, encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    if SHUFFLE_IDS:
        random.shuffle(questions)
    
    if LIMIT_BASE_IDS:
        questions = questions[:LIMIT_BASE_IDS]
    
    log(f"Processing {len(questions)} base IDs")

    # Load caches
    evidence_cache = load_jsonl_by_id(EVIDENCE_CACHE_JSONL)
    trace_cache = load_jsonl_by_id(TRACES_JSONL)
    
    backend = build_backend()
    
    with open(EVIDENCE_CACHE_JSONL, "a", encoding="utf-8") as ev_f, \
         open(TRACES_JSONL, "a", encoding="utf-8") as tr_f:
        
        for i, q_row in enumerate(questions, 1):
            qid = str(q_row["ID"])
            q_text = strip_question_wrapper(q_row["question"])
            
            # 1) Rule engine (Evidence)
            if qid in evidence_cache:
                ev_pack = evidence_cache[qid]
            else:
                metrics, status = rp.compute_metrics(q_text)
                fixed_label, rule_reason = rp.decide(metrics, status)
                
                derived_dict = build_derived_metrics_dict(metrics, status)
                rule_checks = build_rule_checks_dict(metrics, status, rp)
                triggered_rule = infer_triggered_rule(fixed_label, rule_reason)
                triggered_rule_line = build_triggered_rule_line(
                    fixed_label=fixed_label,
                    triggered_rule=triggered_rule,
                    derived=derived_dict,
                    rp=rp
                )
                evidence_bullets = build_evidence_bullets(fixed_label=fixed_label, derived=derived_dict)
                rejection_bullets = build_rejection_bullets(
                    fixed_label=fixed_label,
                    triggered_rule=triggered_rule,
                    derived=derived_dict,
                    rule_checks=rule_checks,
                    rp=rp
                )
                
                ev_pack = {
                    "ID": qid,
                    "fixed_label": fixed_label,
                    "triggered_rule": triggered_rule,
                    "triggered_rule_line": triggered_rule_line,
                    "evidence_bullets": evidence_bullets,
                    "rejection_bullets": rejection_bullets,
                    "rule_checks": rule_checks,
                    "derived_metrics": derived_dict,
                }
                ev_f.write(json.dumps(ev_pack, ensure_ascii=False) + "\n")
                ev_f.flush()
                evidence_cache[qid] = ev_pack

            # 2) LLM Trace
            if qid in trace_cache and trace_cache[qid].get("status") == "success":
                continue
            
            log(f"[{i}/{len(questions)}] Justifying {qid} -> {ev_pack['fixed_label']}")
            
            allowed_kv = set(KV_RE.findall("\n".join(ev_pack["evidence_bullets"])))
            # Add basic counters used in logic.
            allowed_constants = {"0", "1", "1.0", "2", "30", "30.0", "40", "160", "1000", "0.75", "-87.0"}

            success = False
            last_trace = ""
            for attempt in range(1, MAX_ATTEMPTS + 1):
                if attempt == 1:
                    prompt = PROMPT_JUSTIFY.replace("__FIXED_LABEL__", ev_pack["fixed_label"]) \
                        .replace("__TRIGGERED_RULE_LINE__", ev_pack["triggered_rule_line"]) \
                        .replace("__EVIDENCE_BULLETS__", "\n".join(ev_pack["evidence_bullets"])) \
                        .replace("__REJECTION_BULLETS__", "\n".join(ev_pack["rejection_bullets"])) \
                        .replace("__TRIGGERED_RULE__", ev_pack["triggered_rule"]) \
                        .replace("__RULE_CHECKS_JSON__", json.dumps(ev_pack["rule_checks"], indent=2)) \
                        .replace("__DERIVED_METRICS_JSON__", json.dumps(ev_pack["derived_metrics"], indent=2))
                else:
                    prompt = PROMPT_REPAIR.replace("__FIXED_LABEL__", ev_pack["fixed_label"]) \
                        .replace("__TRIGGERED_RULE_LINE__", ev_pack["triggered_rule_line"]) \
                        .replace("__EVIDENCE_BULLETS__", "\n".join(ev_pack["evidence_bullets"])) \
                        .replace("__REJECTION_BULLETS__", "\n".join(ev_pack["rejection_bullets"])) \
                        .replace("__ERRORS__", "\n".join(errors))

                last_trace = backend.generate(prompt, seed=SEED + attempt)
                ok, errors = verify_trace(
                    trace=last_trace,
                    fixed_label=ev_pack["fixed_label"],
                    triggered_rule_line=ev_pack["triggered_rule_line"],
                    evidence_bullets=ev_pack["evidence_bullets"],
                    rejection_bullets=ev_pack["rejection_bullets"],
                    allowed_kv_keys=allowed_kv,
                    allowed_constants=allowed_constants
                )
                if ok:
                    success = True
                    break
                else:
                    log(f"  Attempt {attempt} failed: {errors[0]}")

            if not success:
                log(f"  All {MAX_ATTEMPTS} attempts failed for {qid}. Using deterministic fallback.")
                last_trace = deterministic_fallback_trace(
                    fixed_label=ev_pack["fixed_label"],
                    triggered_rule_line=ev_pack["triggered_rule_line"],
                    evidence_bullets=ev_pack["evidence_bullets"],
                    rejection_bullets=ev_pack["rejection_bullets"],
                )

            tr_pack = {
                "ID": qid,
                "status": "success" if success else "fallback",
                "fixed_label": ev_pack["fixed_label"],
                "trace": last_trace,
                "attempts": attempt if success else MAX_ATTEMPTS,
            }
            tr_f.write(json.dumps(tr_pack, ensure_ascii=False) + "\n")
            tr_f.flush()
            trace_cache[qid] = tr_pack


if __name__ == "__main__":
    main()

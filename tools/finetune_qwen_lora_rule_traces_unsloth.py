#!/usr/bin/env python3
"""
LoRA fine-tuning for Qwen-style chat models on the RCA_RULE trace corpus.
**UNSLOTH VERSION** - 2-3x faster training with 60-80% less VRAM.

Goal:
- The deterministic rule-engine picks the correct semantic label (C1..C8).
- The model is trained ONLY to write a grounded justification trace that:
  - preserves TriggeredRule / Evidence / Rejections blocks exactly
  - adds a short Decision
  - ends with: FinalAnswer: \\boxed{C#}

Dataset input:
- tools/rule_trace_corpus/traces_train_v3_patched.csv
  Columns: ID,fixed_label,triggered_rule,trace_source,attempts,trace

================================================================================
📦 KAGGLE USAGE:
================================================================================
1. Create a new Kaggle notebook with GPU (T4 for 7B, P100/A100 for 32B)

2. Install dependencies in the first cell:
   !pip install -q unsloth transformers datasets trl accelerate

3. Upload your trace corpus as a Kaggle dataset named "rca-trace-corpus"
   (or adjust TRACE_CORPUS_CSV path below)

4. Copy this entire script into a cell and run it, OR run via CLI:
   !python finetune_qwen_lora_rule_traces_unsloth.py

5. The script auto-detects Kaggle and adjusts paths automatically!

6. Output LoRA adapter will be saved to /kaggle/working/lora_qwen_rule_traces/
================================================================================

It uses:
- unsloth (FastLanguageModel)
- transformers
- datasets
- trl (SFTTrainer)
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# =============================================================================
# ⚙️ EASY CONFIGURATION (Kaggle-friendly)
# =============================================================================

# Auto-detect Kaggle environment and adjust paths
import os as _os
IS_KAGGLE = _os.path.exists("/kaggle/working")

# Base model to fine-tune (examples):
# - "Qwen/Qwen2.5-7B-Instruct"   (recommended balance)
# - "Qwen/Qwen2.5-1.5B-Instruct" (fast experiments)
# - "Qwen/Qwen3-32B-Instruct"    (best quality, needs A100/H100)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Input corpus (patched TriggeredRule lines + Decision blocks)
# Adjust path based on environment
if IS_KAGGLE:
    # On Kaggle: your dataset at /kaggle/input/aaaaaaa/
    TRACE_CORPUS_CSV = "/kaggle/input/aaaaaaa/traces_train_v3_patched.csv"
    OUTPUT_DIR = "/kaggle/working/lora_qwen_rule_traces"
    PHASE1_QUESTIONS_CSV_DEFAULT = "/kaggle/input/aaaaaaa/phase_1_test.csv"
else:
    # Local paths
    TRACE_CORPUS_CSV = "tools/rule_trace_corpus/traces_train_v3_patched.csv"
    OUTPUT_DIR = "tools/lora_qwen_rule_traces_unsloth"
    PHASE1_QUESTIONS_CSV_DEFAULT = "data/phase_1_test.csv"

# Smoke-test: set to an int (e.g., 32/128) to debug quickly; set None for all.
NUM_SAMPLES: Optional[int] = None

# Evaluation mode:
# - "HOLDOUT": split a small subset from the trace corpus (recommended default).
# - "PHASE1":  build deterministic traces for phase_1_test.csv via the RULE engine.
# - "NONE":    no evaluation during training.
EVAL_MODE = "HOLDOUT"  # "HOLDOUT" | "PHASE1" | "NONE"

# HOLDOUT eval size:
# - set to an int (e.g., 128) for a small fixed eval subset
# - or set to a float fraction (e.g., 0.02)
HOLDOUT_EVAL_SIZE: float | int = 64  # Reduced for small dataset

# PHASE1 evaluation settings
PHASE1_QUESTIONS_CSV = PHASE1_QUESTIONS_CSV_DEFAULT
PHASE1_EVAL_LIMIT: Optional[int] = None  # None = all 864

# Sequence length (your traces are structured/short, 1024 is plenty)
MAX_SEQ_LEN = 1024

# =============================================================================
# ⚙️ MODEL-SPECIFIC HYPERPARAMETERS
# =============================================================================
# These are auto-configured based on MODEL_NAME, but you can override.

# Model size detection (used to set sensible defaults)
def _detect_model_size(model_name: str) -> str:
    """Detect model size from name for auto-config."""
    name_lower = model_name.lower()
    if "32b" in name_lower or "30b" in name_lower:
        return "32B"
    elif "14b" in name_lower:
        return "14B"
    elif "7b" in name_lower or "8b" in name_lower:
        return "7B"
    elif "3b" in name_lower or "4b" in name_lower:
        return "3B"
    elif "1.5b" in name_lower or "1b" in name_lower or "0.5b" in name_lower:
        return "1.5B"
    return "7B"  # default

_MODEL_SIZE = _detect_model_size(MODEL_NAME)

# LoRA hyperparams (model-size dependent to prevent memorization)
# For small datasets (~864 rows), smaller r prevents overfitting
LORA_CONFIGS = {
    "32B":  {"r": 8,  "alpha": 16, "dropout": 0.1},  # Conservative for large model
    "14B":  {"r": 8,  "alpha": 16, "dropout": 0.1},
    "7B":   {"r": 16, "alpha": 32, "dropout": 0.05},  # Standard
    "3B":   {"r": 16, "alpha": 32, "dropout": 0.05},
    "1.5B": {"r": 16, "alpha": 32, "dropout": 0.05},
}
LORA_R = LORA_CONFIGS.get(_MODEL_SIZE, LORA_CONFIGS["7B"])["r"]
LORA_ALPHA = LORA_CONFIGS.get(_MODEL_SIZE, LORA_CONFIGS["7B"])["alpha"]
LORA_DROPOUT = LORA_CONFIGS.get(_MODEL_SIZE, LORA_CONFIGS["7B"])["dropout"]

# Learning rate (model-size dependent to prevent overfitting)
# Larger models need lower LR on small datasets
LR_CONFIGS = {
    "32B":  5e-5,   # Very conservative
    "14B":  5e-5,
    "7B":   1e-4,   # Balanced
    "3B":   1.5e-4,
    "1.5B": 2e-4,   # Can be more aggressive
}
LEARNING_RATE = LR_CONFIGS.get(_MODEL_SIZE, 1e-4)

# Epochs (model-size dependent)
EPOCH_CONFIGS = {
    "32B":  1,
    "14B":  1,
    "7B":   2,
    "3B":   2,
    "1.5B": 3,
}
NUM_EPOCHS = EPOCH_CONFIGS.get(_MODEL_SIZE, 1)

# Training hyperparams
PER_DEVICE_BATCH_SIZE = 2  # Unsloth allows larger batches!
GRAD_ACCUM_STEPS = 8       # Effective batch = 16
WEIGHT_DECAY = 0.01        # Slight regularization
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"

# Precision
# - None: auto-detect (bf16 on modern GPUs)
# - "bfloat16": force bf16
# - "float16": force fp16
DTYPE = None  # Auto-detect

# Quantization (4-bit QLoRA)
# - False: full precision (16-bit) - BETTER QUALITY
# - True: 4-bit quantized - LESS VRAM
USE_4BIT = False

# Logging/saving - FIXED: was 200, but ~54 steps/epoch means it never triggered!
# With 864 rows, batch=2, grad_accum=8 → ~54 steps/epoch
LOGGING_STEPS = 5
SAVE_STEPS = 25
EVAL_STEPS = 25
SAVE_TOTAL_LIMIT = 3

# Use completion-only loss masking (train only on assistant response, not prompt)
# This is CRITICAL for SFT - prevents wasting capacity on prompt prediction
USE_COMPLETION_ONLY_LOSS = True

# Use group split by ID to prevent data leakage in holdout eval
USE_GROUP_SPLIT = True

# Reproducibility
SEED = 42

# =============================================================================
# Prompt template used for training
# =============================================================================

SYSTEM_PROMPT = "You are a staff 5G RAN engineer writing a grounded RCA justification trace."

USER_PROMPT = """A deterministic rule-engine already selected the correct label.

FixedLabel: {fixed_label}

Hard rules (must follow):
- Do NOT change FixedLabel.
- Copy the TriggeredRule/Evidence/Rejections blocks EXACTLY as given (no edits).
- Do NOT introduce any numeric values not already present in those blocks.
- Do NOT invent new metrics/fields or new key=value pairs.
- Add a short Decision (1 bullet).
- The LAST line must be EXACTLY: FinalAnswer: \\\\boxed{{{fixed_label}}}

TriggeredRule line (copy EXACTLY):
{triggered_rule_line}

Evidence block (copy EXACTLY):
Evidence:
{evidence_bullets}

Rejections block (copy EXACTLY):
Rejections:
{rejection_bullets}

Now write the justification in this STRICT format:
{triggered_rule_line}

Evidence:
{evidence_bullets}

Rejections:
{rejection_bullets}

Decision:
- <1 short bullet, no new numbers, no new key=value>
FinalAnswer: \\\\boxed{{{fixed_label}}}
"""


# =============================================================================
# Utilities
# =============================================================================


def _require(deps: list[str]) -> None:
    missing: list[str] = []
    for d in deps:
        try:
            __import__(d)
        except Exception:
            missing.append(d)
    if not missing:
        return
    print("Missing Python packages:", ", ".join(missing))
    print(
        "Install (example):\n"
        "  pip install -U unsloth transformers datasets trl accelerate\n"
        "Then re-run this script."
    )
    raise SystemExit(2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Extra determinism flags
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


@dataclass(frozen=True)
class ParsedBlocks:
    triggered_rule_line: str
    evidence_bullets: list[str]
    rejection_bullets: list[str]


def parse_trace_blocks(trace: str) -> ParsedBlocks:
    """
    Extract:
    - TriggeredRule: ... (single line)
    - Evidence bullet lines (between Evidence: and Rejections:)
    - Rejections bullet lines (between Rejections: and Decision:/FinalAnswer:)
    """
    lines = trace.splitlines()

    trig = ""
    for ln in lines:
        if ln.startswith("TriggeredRule:"):
            trig = ln
            break
    if not trig:
        raise ValueError("Trace missing TriggeredRule line")

    def find_idx(label: str) -> int:
        for i, ln in enumerate(lines):
            if ln.strip() == label:
                return i
        return -1

    idx_e = find_idx("Evidence:")
    idx_r = find_idx("Rejections:")
    if idx_e < 0 or idx_r < 0 or idx_r <= idx_e:
        raise ValueError("Trace missing Evidence/Rejections blocks")

    idx_dec = find_idx("Decision:")
    if idx_dec < 0:
        idx_dec = len(lines)

    ev_lines = [ln for ln in lines[idx_e + 1 : idx_r] if ln.lstrip().startswith("-")]
    rej_lines = [ln for ln in lines[idx_r + 1 : idx_dec] if ln.lstrip().startswith("-")]

    if not ev_lines:
        raise ValueError("No evidence bullets found")
    if not rej_lines:
        raise ValueError("No rejection bullets found")

    return ParsedBlocks(triggered_rule_line=trig, evidence_bullets=ev_lines, rejection_bullets=rej_lines)


def build_training_text(*, tokenizer: Any, fixed_label: str, trace: str) -> str:
    blocks = parse_trace_blocks(trace)
    user = USER_PROMPT.format(
        fixed_label=fixed_label,
        triggered_rule_line=blocks.triggered_rule_line,
        evidence_bullets="\n".join(blocks.evidence_bullets),
        rejection_bullets="\n".join(blocks.rejection_bullets),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": trace},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def _load_reasoned_predict_module() -> Any:
    here = Path(__file__).resolve().parent
    path = here / "reasoned_predict.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing reasoned_predict.py at {path}")
    spec = importlib.util.spec_from_file_location("reasoned_predict", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load reasoned_predict module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _infer_triggered_rule(rule_reason: str) -> str:
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
    return "RULES_unknown"


def _metrics_to_derived_dict(metrics: Any, status: str) -> dict[str, Any]:
    if metrics is None:
        return {"metricsStatus": status}
    m = metrics
    return {
        "metricsStatus": status,
        "nRows": int(getattr(m, "n_rows", 0)),
        "lowRows": int(getattr(m, "n_low", 0)),
        "speedMax": float(getattr(m, "speed_max", 0.0)) if getattr(m, "speed_max", None) is not None else None,
        "avgRB_low": float(getattr(m, "avg_rb_low", 0.0)) if getattr(m, "avg_rb_low", None) is not None else None,
        "handovers": int(getattr(m, "handovers", 0)),
        "maxDist_m": float(getattr(m, "max_dist_m", 0.0)),
        "mod30Frac": float(getattr(m, "mod30_frac", 0.0)),
        "sameGnbCloseFrac": float(getattr(m, "same_gnb_close_frac", 0.0)),
        "diffGnbCloseFrac": float(getattr(m, "diff_gnb_close_frac", 0.0)),
        "beyondBeamRows": int(getattr(m, "beyond_beam_rows", 0)),
        "beyondBeamDenom": int(getattr(m, "beyond_beam_denom", 0)),
        "rsrpMedLow": float(getattr(m, "rsrp_med_low", 0.0)) if getattr(m, "rsrp_med_low", None) is not None else None,
        "tiltMedLow": float(getattr(m, "tilt_med_low", 0.0)) if getattr(m, "tilt_med_low", None) is not None else None,
        "tiltMaxLow": float(getattr(m, "tilt_max_low", 0.0)) if getattr(m, "tilt_max_low", None) is not None else None,
        "c1StrongPcis": list(getattr(m, "c1_strong_pcis", []) or []),
    }


def _fmt_num(v: Any, nd: int) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _build_triggered_rule_line(*, fixed_label: str, triggered_rule: str, derived: dict[str, Any]) -> str:
    # Thresholds aligned with tools/reasoned_predict.py
    if fixed_label == "C7":
        return f"TriggeredRule: speedMax > 40 (speedMax={_fmt_num(derived.get('speedMax'),1)}, threshold=40)"
    if fixed_label == "C8":
        return f"TriggeredRule: avgRB_low < 160 (avgRB_low={_fmt_num(derived.get('avgRB_low'),1)}, threshold=160)"
    if fixed_label == "C2":
        return f"TriggeredRule: maxDist_m > 1000 (maxDist_m={_fmt_num(derived.get('maxDist_m'),1)}, threshold=1000)"
    if fixed_label == "C6":
        return f"TriggeredRule: mod30Frac >= 0.75 (mod30Frac={_fmt_num(derived.get('mod30Frac'),2)}, threshold=0.75)"
    if fixed_label == "C5":
        return f"TriggeredRule: handovers >= 2 (handovers={derived.get('handovers')}, threshold=2)"
    if fixed_label == "C4":
        return f"TriggeredRule: diffGnbCloseFrac == 1 (diffGnbCloseFrac={_fmt_num(derived.get('diffGnbCloseFrac'),2)}, threshold=1)"
    if fixed_label == "C3":
        return f"TriggeredRule: sameGnbCloseFrac == 1.0 (sameGnbCloseFrac={_fmt_num(derived.get('sameGnbCloseFrac'),2)}, threshold=1.0)"
    if fixed_label == "C1":
        bb_rows = derived.get("beyondBeamRows")
        bb_den = derived.get("beyondBeamDenom")
        low_rows = derived.get("lowRows")
        rsrp = derived.get("rsrpMedLow")
        tilt = derived.get("tiltMedLow")
        strong = derived.get("c1StrongPcis") or []

        if triggered_rule == "R7_c1StrongPcis_nonempty":
            return f"TriggeredRule: c1StrongPcis nonempty (c1StrongPcis={strong}, threshold=nonempty)"
        if triggered_rule == "R8_override_C1_allLowBeyond":
            return (
                "TriggeredRule: all low rows beyond far-edge "
                f"(beyondBeamRows={bb_rows}, beyondBeamDenom={bb_den}, lowRows={low_rows}, threshold=all)"
            )
        if triggered_rule == "R8_override_C1_extremeDowntilt":
            return (
                "TriggeredRule: extreme downtilt (tiltMedLow>=30 and beyondBeamRows>0) "
                f"(tiltMedLow={_fmt_num(tilt,1)}, threshold=30; beyondBeamRows={bb_rows}, threshold=0)"
            )
        if triggered_rule == "R8_override_C1_weak_beam_edge":
            return (
                "TriggeredRule: weak+beam-edge (rsrpMedLow<=-87 and beyondBeamRows>0) "
                f"(rsrpMedLow={_fmt_num(rsrp,2)}, threshold=-87; beyondBeamRows={bb_rows}, threshold=0)"
            )
        if triggered_rule == "R10_fallback":
            return "TriggeredRule: fallback (threshold=none)"
        return f"TriggeredRule: beyondBeamRows > 0 (beyondBeamRows={bb_rows}, threshold=0)"

    return f"TriggeredRule: {triggered_rule} (threshold=missing)"


def _build_evidence_bullets(*, fixed_label: str, derived: dict[str, Any]) -> list[str]:
    # Match the 4-line style used in the corpus.
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

    out: list[str] = []
    for k in keys:
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


def _build_rule_checks(*, metrics: Any, status: str, rp: Any) -> dict[str, Any]:
    if metrics is None:
        return {"metricsStatus": status}
    m = metrics
    rsrp_thr = float(getattr(rp, "C3_BLOCK_IF_BEYOND_AND_RSRP_LE", -87.0))

    same1 = abs(float(m.same_gnb_close_frac) - 1.0) < 1e-9
    all_low_beyond = (
        m.beyond_beam_denom > 0 and m.beyond_beam_rows == m.beyond_beam_denom and m.beyond_beam_denom == m.n_low
    )
    extreme_tilt = (m.tilt_med_low is not None) and (float(m.tilt_med_low) >= 30.0) and (m.beyond_beam_rows > 0)
    weak_edge = (m.rsrp_med_low is not None) and (float(m.rsrp_med_low) <= rsrp_thr) and (m.beyond_beam_rows > 0)

    out = {
        "metricsStatus": status,
        "R1_speedMax_gt_40": bool(m.speed_max is not None and m.speed_max > float(getattr(rp, "SPEED_MAX_KMH", 40.0))),
        "R2_avgRB_low_lt_160": bool(m.avg_rb_low is not None and m.avg_rb_low < float(getattr(rp, "RB_MIN", 160.0))),
        "R3_maxDist_m_gt_1000": bool(m.max_dist_m > float(getattr(rp, "OVERSHOOT_M", 1000.0))),
        "R4_mod30Frac_ge_0.75": bool(m.mod30_frac >= float(getattr(rp, "MOD30_FRAC_MIN", 0.75))),
        "R5_handovers_ge_2": bool(m.handovers >= int(getattr(rp, "HANDOVERS_MIN", 2))),
        "R6_diffGnbCloseFrac_eq_1": bool(m.diff_gnb_close_frac >= float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))),
        "R7_c1StrongPcis_nonempty": bool(bool(m.c1_strong_pcis)),
        "R8_sameGnbCloseFrac_eq_1": bool(same1),
        "R8_override_C1_allLowBeyond": bool(same1 and all_low_beyond),
        "R8_override_C1_extremeDowntilt": bool(same1 and extreme_tilt),
        "R8_override_C1_weak_beam_edge": bool(same1 and weak_edge),
        "R9_beyondBeamRows_gt_0": bool(m.beyond_beam_rows > 0),
    }
    # This override is only relevant when diff==1 and same==1 and weak_edge.
    out["R6_override_C1_weak_beam_edge"] = bool(
        out["R6_diffGnbCloseFrac_eq_1"] and same1 and weak_edge
    )
    return out


def _build_rejection_bullets(*, fixed_label: str, triggered_rule: str, derived: dict[str, Any], rule_checks: dict[str, Any]) -> list[str]:
    specs = [
        ("C7", "speedMax", _fmt_num(derived.get("speedMax"), 1), "speedMax>40", "40", "R1_speedMax_gt_40"),
        ("C8", "avgRB_low", _fmt_num(derived.get("avgRB_low"), 1), "avgRB_low<160", "160", "R2_avgRB_low_lt_160"),
        ("C2", "maxDist_m", _fmt_num(derived.get("maxDist_m"), 1), "maxDist_m>1000", "1000", "R3_maxDist_m_gt_1000"),
        ("C6", "mod30Frac", _fmt_num(derived.get("mod30Frac"), 2), "mod30Frac>=0.75", "0.75", "R4_mod30Frac_ge_0.75"),
        ("C5", "handovers", str(derived.get("handovers") if derived.get("handovers") is not None else "NA"), "handovers>=2", "2", "R5_handovers_ge_2"),
        ("C4", "diffGnbCloseFrac", _fmt_num(derived.get("diffGnbCloseFrac"), 2), "diffGnbCloseFrac==1", "1", "R6_diffGnbCloseFrac_eq_1"),
        ("C3", "sameGnbCloseFrac", _fmt_num(derived.get("sameGnbCloseFrac"), 2), "sameGnbCloseFrac==1.0", "1.0", "R8_sameGnbCloseFrac_eq_1"),
        ("C1", "beyondBeamRows", str(derived.get("beyondBeamRows") if derived.get("beyondBeamRows") is not None else "NA"), "beyondBeamRows>0", "0", "R9_beyondBeamRows_gt_0"),
    ]

    out: list[str] = []
    for lbl, metric, mval, cond, thr, key in specs:
        if lbl == fixed_label:
            continue
        if rule_checks.get(key) is True:
            out.append(
                f"- {lbl} not selected because although {metric}={mval} meets {cond} (threshold={thr}), a higher-priority rule fired (TriggeredRule: {triggered_rule})."
            )
        else:
            out.append(f"- {lbl} rejected because {metric}={mval} does not meet {cond} (threshold={thr})")
    return out


def _deterministic_trace(*, fixed_label: str, trig_line: str, ev: list[str], rej: list[str]) -> str:
    return "\n".join(
        [
            trig_line,
            "",
            "Evidence:",
            *ev,
            "",
            "Rejections:",
            *rej,
            "",
            "Decision:",
            f"- TriggeredRule indicates {fixed_label} is selected by the rubric.",
            f"FinalAnswer: \\\\boxed{{{fixed_label}}}",
        ]
    )


def build_phase1_eval_texts(*, tokenizer: Any) -> list[str]:
    rp = _load_reasoned_predict_module()
    df = pd.read_csv(PHASE1_QUESTIONS_CSV)
    if PHASE1_EVAL_LIMIT is not None:
        df = df.head(int(PHASE1_EVAL_LIMIT))

    texts: list[str] = []
    for _, row in df.iterrows():
        qtxt = str(row["question"])
        metrics, status = rp.compute_metrics(qtxt)
        fixed_label, reason = rp.decide(metrics, status)
        trig = _infer_triggered_rule(reason)
        derived = _metrics_to_derived_dict(metrics, status)
        checks = _build_rule_checks(metrics=metrics, status=status, rp=rp)
        trig_line = _build_triggered_rule_line(fixed_label=fixed_label, triggered_rule=trig, derived=derived)
        ev = _build_evidence_bullets(fixed_label=fixed_label, derived=derived)
        rej = _build_rejection_bullets(fixed_label=fixed_label, triggered_rule=trig, derived=derived, rule_checks=checks)
        trace = _deterministic_trace(fixed_label=fixed_label, trig_line=trig_line, ev=ev, rej=rej)
        texts.append(build_training_text(tokenizer=tokenizer, fixed_label=fixed_label, trace=trace))
    return texts


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    _require(["torch", "transformers", "datasets", "trl", "unsloth"])

    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import TrainingArguments  # type: ignore
    from trl import SFTTrainer  # type: ignore
    from unsloth import FastLanguageModel  # type: ignore
    
    # Try to import DataCollatorForCompletionOnlyLM (not available in older TRL versions)
    DataCollatorForCompletionOnlyLM = None
    try:
        from trl import DataCollatorForCompletionOnlyLM as _DataCollator
        DataCollatorForCompletionOnlyLM = _DataCollator
    except ImportError:
        print("⚠️  DataCollatorForCompletionOnlyLM not available in this TRL version.")
        print("   Training will include prompt tokens in loss (less efficient but still works).")
        print("   To enable completion-only loss, upgrade TRL: pip install -U trl>=0.8.0")

    set_seed(SEED)

    csv_path = Path(TRACE_CORPUS_CSV)
    if not csv_path.exists():
        raise SystemExit(f"Missing dataset CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if NUM_SAMPLES is not None:
        df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=SEED).reset_index(drop=True)

    # Calculate expected steps for info
    train_size = len(df) - (int(HOLDOUT_EVAL_SIZE) if EVAL_MODE == "HOLDOUT" else 0)
    effective_batch = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = train_size // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    
    print("=" * 60)
    print("🦥 Qwen LoRA Fine-tune (RCA_RULE traces) - UNSLOTH")
    print("=" * 60)
    print(f"Model: {MODEL_NAME} (detected size: {_MODEL_SIZE})")
    print(f"Dataset: {csv_path} (rows={len(df)})")
    print(f"Output: {OUTPUT_DIR}")
    print(f"EVAL_MODE: {EVAL_MODE}")
    if EVAL_MODE == "HOLDOUT":
        print(f"  HOLDOUT_EVAL_SIZE: {HOLDOUT_EVAL_SIZE}")
        print(f"  USE_GROUP_SPLIT: {USE_GROUP_SPLIT}")
    elif EVAL_MODE == "PHASE1":
        print(f"  PHASE1_QUESTIONS_CSV: {PHASE1_QUESTIONS_CSV}")
        print(f"  PHASE1_EVAL_LIMIT: {PHASE1_EVAL_LIMIT}")
    print(f"USE_4BIT: {USE_4BIT}")
    print(f"USE_COMPLETION_ONLY_LOSS: {USE_COMPLETION_ONLY_LOSS}")
    print(f"MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    print(f"Batch: {PER_DEVICE_BATCH_SIZE}  GradAccum: {GRAD_ACCUM_STEPS}  Effective: {effective_batch}")
    print(f"Epochs: {NUM_EPOCHS}  Steps/epoch: ~{steps_per_epoch}  Total: ~{total_steps}")
    print(f"LR: {LEARNING_RATE}  LoRA(r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})")
    print(f"Eval/Save every {EVAL_STEPS} steps (will trigger ~{total_steps // EVAL_STEPS} times)")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    hparams = {
        "model_name": MODEL_NAME,
        "trace_corpus_csv": TRACE_CORPUS_CSV,
        "num_samples": NUM_SAMPLES,
        "eval": {
            "mode": EVAL_MODE,
            "holdout_eval_size": HOLDOUT_EVAL_SIZE,
            "phase1_questions_csv": PHASE1_QUESTIONS_CSV,
            "phase1_eval_limit": PHASE1_EVAL_LIMIT,
        },
        "max_seq_len": MAX_SEQ_LEN,
        "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "train": {
            "epochs": NUM_EPOCHS,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lr_scheduler": LR_SCHEDULER,
            "seed": SEED,
            "dtype": str(DTYPE),
            "use_4bit": USE_4BIT,
        },
        "logging": {"logging_steps": LOGGING_STEPS, "save_steps": SAVE_STEPS, "eval_steps": EVAL_STEPS},
        "backend": "unsloth",
    }
    with open(Path(OUTPUT_DIR) / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)

    # =========================================================================
    # 🦥 UNSLOTH: Load model and tokenizer
    # =========================================================================
    print("\n🦥 Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=DTYPE,           # None = auto-detect (bf16 on modern GPUs)
        load_in_4bit=USE_4BIT,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # =========================================================================
    # 🦥 UNSLOTH: Add LoRA adapters
    # =========================================================================
    print("🦥 Adding LoRA adapters with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM!
        random_state=SEED,
    )

    # =========================================================================
    # Prepare dataset
    # =========================================================================
    texts: list[str] = []
    for _, row in df.iterrows():
        fixed = str(row["fixed_label"])
        trace = str(row["trace"])
        texts.append(build_training_text(tokenizer=tokenizer, fixed_label=fixed, trace=trace))

    # Build dataset with IDs for group splitting
    ds = Dataset.from_dict({"text": texts, "id": df["ID"].tolist()})

    train_ds = ds
    eval_ds = None
    do_eval = False

    if EVAL_MODE == "HOLDOUT":
        if USE_GROUP_SPLIT and "ID" in df.columns:
            # Group split by ID to prevent data leakage
            unique_ids = df["ID"].unique()
            n_eval = int(HOLDOUT_EVAL_SIZE) if isinstance(HOLDOUT_EVAL_SIZE, int) else int(len(unique_ids) * HOLDOUT_EVAL_SIZE)
            n_eval = min(n_eval, len(unique_ids) - 1)
            
            import numpy as np
            rng = np.random.RandomState(SEED)
            rng.shuffle(unique_ids)
            eval_ids = set(unique_ids[:n_eval])
            
            train_mask = [i for i, row in enumerate(ds) if row["id"] not in eval_ids]
            eval_mask = [i for i, row in enumerate(ds) if row["id"] in eval_ids]
            
            train_ds = ds.select(train_mask)
            eval_ds = ds.select(eval_mask)
            print(f"  Group split: {len(train_ds)} train, {len(eval_ds)} eval (by {len(eval_ids)} unique IDs)")
        else:
            # Simple random split (original behavior)
            split = ds.train_test_split(test_size=HOLDOUT_EVAL_SIZE, seed=SEED, shuffle=True)
            train_ds, eval_ds = split["train"], split["test"]
        do_eval = True
    elif EVAL_MODE == "PHASE1":
        eval_texts = build_phase1_eval_texts(tokenizer=tokenizer)
        eval_ds = Dataset.from_dict({"text": eval_texts})
        do_eval = True
    elif EVAL_MODE == "NONE":
        do_eval = False
    else:
        raise ValueError(f"Unknown EVAL_MODE: {EVAL_MODE!r}")

    eval_strategy = "steps" if do_eval else "no"

    # =========================================================================
    # Training arguments
    # =========================================================================
    # Try fused optimizer if available (faster on H100/A100)
    try:
        import torch
        if hasattr(torch.optim, 'AdamW') and torch.cuda.is_available():
            optim = "adamw_torch_fused"
        else:
            optim = "adamw_8bit"
    except Exception:
        optim = "adamw_8bit"
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        optim=optim,
        bf16=True,           # Unsloth works best with bf16
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy=eval_strategy,  # renamed from evaluation_strategy in newer transformers
        eval_steps=EVAL_STEPS,
        report_to=[],
        remove_unused_columns=False,
    )

    # =========================================================================
    # Completion-only loss masking (IMPORTANT for SFT!)
    # =========================================================================
    # This ensures the model only learns to predict the assistant response,
    # not the system/user prompts. Prevents wasting capacity on prompt tokens.
    data_collator = None
    if USE_COMPLETION_ONLY_LOSS and DataCollatorForCompletionOnlyLM is not None:
        # For Qwen chat templates, the assistant turn starts after this token
        # We need to find the response template token sequence
        # Qwen uses: <|im_start|>assistant\n
        response_template = "<|im_start|>assistant\n"
        try:
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
            print(f"  Using completion-only loss (response_template: {response_template!r})")
        except Exception as e:
            print(f"  Warning: Could not create completion-only collator: {e}")
            print("  Falling back to standard loss (includes prompt tokens)")
            data_collator = None
    elif USE_COMPLETION_ONLY_LOSS and DataCollatorForCompletionOnlyLM is None:
        print("  Skipping completion-only loss (DataCollatorForCompletionOnlyLM not available)")

    # =========================================================================
    # 🦥 Train with SFTTrainer (same as before, but model is Unsloth-optimized)
    # =========================================================================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        data_collator=data_collator,
    )

    print("\n🦥 Starting training...")
    trainer.train()

    # =========================================================================
    # Save the LoRA adapter
    # =========================================================================
    print("\n🦥 Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Optionally save merged model (full weights) for easier deployment
    # Uncomment to merge LoRA into base model:
    # print("\n🦥 Merging LoRA and saving full model...")
    # model.save_pretrained_merged(OUTPUT_DIR + "_merged", tokenizer, save_method="merged_16bit")

    print("=" * 60)
    print("🦥 DONE")
    print(f"Saved LoRA adapter to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

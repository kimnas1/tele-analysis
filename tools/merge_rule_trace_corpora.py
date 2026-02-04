#!/usr/bin/env python3
"""
Merge Phase1/train rule-trace corpus with Phase2 RCA_RULE trace corpus.

Why
---
Phase1 traces usually end with \\boxed{C#}.
Phase2 RCA_RULE traces end with \\boxed{<question-specific option key>} such as:
- digits: 7
- alphanum: Z4
- sometimes C#: C2

This script produces a single merged CSV suitable for SFT/LoRA fine-tuning.

Inputs
------
- --train-csv: Phase1/train trace corpus CSV (must include a 'trace' column)
  If it has 'answer_value' it will be used; else if it has 'fixed_label', we set:
    answer_value = fixed_label and answer_schema = 'phase1_clabel'
- --phase2-csv: Phase2 RCA_RULE trace corpus CSV (LLM or synth)

Output
------
CSV with columns:
  ID,answer_value,answer_schema,triggered_rule,trace_source,attempts,trace
and optional extra columns preserved if present.

All fields are quoted to preserve multiline traces.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Any, List

import pandas as pd


REQUIRED = {"ID", "trace"}


def _ensure_answer_value(df: pd.DataFrame, *, default_schema: str) -> pd.DataFrame:
    df = df.copy()
    if "answer_value" not in df.columns:
        if "fixed_label" in df.columns:
            df["answer_value"] = df["fixed_label"].astype(str)
            df["answer_schema"] = default_schema
        else:
            raise ValueError("Input corpus missing both answer_value and fixed_label; cannot infer answer_value.")
    if "answer_schema" not in df.columns:
        df["answer_schema"] = default_schema
    if "triggered_rule" not in df.columns:
        df["triggered_rule"] = ""
    if "trace_source" not in df.columns:
        df["trace_source"] = ""
    if "attempts" not in df.columns:
        df["attempts"] = 0
    return df


def _validate(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _order_cols(df: pd.DataFrame) -> List[str]:
    base = ["ID", "answer_value", "answer_schema", "triggered_rule", "trace_source", "attempts", "trace"]
    extras = [c for c in df.columns if c not in base]
    return base + extras


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="Train/Phase1 trace corpus CSV (multiline-safe).")
    ap.add_argument(
        "--phase2-csv",
        default="tools/rule_trace_corpus_phase2_llm/traces_phase2_rule_llm.csv",
        help="Phase2 RCA_RULE trace corpus CSV.",
    )
    ap.add_argument(
        "--out",
        default="tools/rule_trace_corpus_merged/traces_train_plus_phase2.csv",
        help="Output merged CSV.",
    )
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    phase2 = pd.read_csv(args.phase2_csv)

    _validate(train, "train")
    _validate(phase2, "phase2")

    train = _ensure_answer_value(train, default_schema="phase1_clabel")
    phase2 = _ensure_answer_value(phase2, default_schema="phase2")

    # Normalize types
    train["ID"] = train["ID"].astype(str)
    phase2["ID"] = phase2["ID"].astype(str)

    # De-dup by ID preferring Phase2 if collision (shouldn't happen usually).
    merged = pd.concat([train, phase2], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ID"], keep="last")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    merged = merged[_order_cols(merged)]
    merged.to_csv(args.out, index=False, quoting=csv.QUOTE_ALL)

    print(f"Wrote: {args.out}")
    print(f"Rows: train={len(train)} phase2={len(phase2)} merged={len(merged)}")


if __name__ == "__main__":
    main()

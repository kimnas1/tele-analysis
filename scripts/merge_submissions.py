#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List

# Allow running as `python3 scripts/merge_submissions.py ...` from repo root or notebooks.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rca.config import load_config
from rca.submission import load_template, row_id_index, write_submission
from rca.util import log


def _overlay_column(dst_rows: List[dict], dst_index: Dict[str, int], src_path: str, column: str) -> int:
    changed = 0
    with open(src_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "ID" not in r.fieldnames or column not in r.fieldnames:
            raise ValueError(f"Bad submission file: {src_path}")
        for row in r:
            rid = str(row.get("ID") or "").strip()
            if not rid or rid not in dst_index:
                continue
            val = row.get(column)
            if val is None or str(val).strip() == "placeholder":
                continue
            dst_rows[dst_index[rid]][column] = val
            changed += 1
    return changed


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: python3 scripts/merge_submissions.py <data_dir> <out_dir> <merged_csv>")

    data_dir = sys.argv[1].strip()
    out_dir = sys.argv[2].strip()
    merged_csv = sys.argv[3].strip()

    cfg = load_config(os.path.join("configs", "pipeline.json")).raw
    sample_submission = os.path.join(data_dir, cfg.get("filenames", {}).get("sample_submission", "SampleSubmission.csv"))
    rows, fieldnames = load_template(sample_submission)
    dst_index = row_id_index(rows)

    log("=" * 60)
    log("Merge per-model submissions")
    log("=" * 60)
    log(f"data_dir: {data_dir}")
    log(f"out_dir:  {out_dir}")
    log(f"output:   {merged_csv}")

    for model_tag, model_cfg in cfg.get("models", {}).items():
        col = model_cfg.get("submission_column")
        if not col or col not in fieldnames:
            continue
        candidate = os.path.join(out_dir, model_tag, f"submission_{model_tag}.csv")
        if not os.path.exists(candidate):
            log(f"Skip {model_tag}: missing {candidate}")
            continue
        changed = _overlay_column(rows, dst_index, candidate, col)
        log(f"Merged {model_tag} -> {col}: {changed} rows")

    write_submission(merged_csv, rows, fieldnames)
    log("Done.")


if __name__ == "__main__":
    main()

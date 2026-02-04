#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

# Allow running as `python3 scripts/run_model.py ...` from repo root or notebooks.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rca.config import load_config
from rca.questions import load_questions_csv, write_questions_csv
from rca.router import load_reasoned_predict, route_questions_phase2
from rca.submission import base_slot_index, force_placeholder_else, load_template, row_id_index, write_submission
from rca.util import ensure_dir, log, sha256_file
from rca.rca_rule.generate import generate_rule_traces
from rca.rca_ood.generate import run_vote3_rca_ood
from rca.general.generate import run_vote3_general


def _safe_join(dir_path: str, filename: str) -> str:
    return os.path.join(dir_path, filename)


def _load_pipeline_config() -> Any:
    cfg_path = os.path.join(_REPO_ROOT, "configs", "pipeline.json")
    return load_config(cfg_path)


def _submission_overlay(
    *,
    dst_rows: List[dict],
    dst_row_index: Dict[str, int],
    src_submission_csv: str,
    submission_column: str,
) -> int:
    import csv

    changed = 0
    with open(src_submission_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "ID" not in r.fieldnames or submission_column not in r.fieldnames:
            raise ValueError(f"Bad submission file: {src_submission_csv}")
        for row in r:
            rid = str(row.get("ID") or "").strip()
            if not rid or rid not in dst_row_index:
                continue
            val = row.get(submission_column)
            if val is None:
                continue
            if str(val).strip() == "placeholder":
                continue
            dst_rows[dst_row_index[rid]][submission_column] = val
            changed += 1
    return changed


def main() -> None:
    if len(sys.argv) not in (4, 5, 6):
        raise SystemExit(
            "Usage: python3 scripts/run_model.py <model_tag> <data_dir> <out_dir> [lora_model_dir] [run_only]"
        )

    model_tag = sys.argv[1].strip()
    data_dir = sys.argv[2].strip()
    out_dir = sys.argv[3].strip()
    override_lora_model_dir: Optional[str] = None
    argv_run_only: Optional[str] = None

    if len(sys.argv) == 5:
        # Backward-compatible:
        # - 5th arg is usually lora_model_dir
        # - but allow passing run_only without LoRA by using a known selector value
        maybe = sys.argv[4].strip()
        if maybe.upper() in {"RCA_RULE_PHASE2", "RCA_RULE_ONLY", "RCA_RULE"}:
            argv_run_only = maybe
        else:
            override_lora_model_dir = maybe
    elif len(sys.argv) == 6:
        override_lora_model_dir = sys.argv[4].strip()
        argv_run_only = sys.argv[5].strip()

    # Optional run selector (so Kaggle runs can target only the needed slice).
    # Values:
    # - "" (default): run Phase2 RCA_OOD + GENERAL + Phase1+Phase2 RCA_RULE
    # - "RCA_RULE_PHASE2": run ONLY Phase2 routed RCA (RULE) questions
    # - "RCA_RULE_ONLY": run ONLY Phase1+Phase2 RCA_RULE questions
    env_run_only = os.environ.get("RUN_ONLY", "").strip()
    RUN_ONLY = (argv_run_only or env_run_only).strip().upper()
    if RUN_ONLY in {"PHASE2_RULE", "RULE_PHASE2"}:
        RUN_ONLY = "RCA_RULE_PHASE2"
    if RUN_ONLY in {"RULE_ONLY", "RCA_RULE_ONLY", "RCA_RULE_ALL", "RCA_RULE_BOTH"}:
        RUN_ONLY = "RCA_RULE_ONLY"
    if RUN_ONLY == "RCA_RULE":
        RUN_ONLY = "RCA_RULE_ONLY"

    cfg = _load_pipeline_config().raw
    if model_tag in cfg.get("models", {}):
        model_cfg = cfg["models"][model_tag]
    else:
        # Dynamic fallback: argument is an HF model string (e.g. "Qwen/Qwen2.5-7B-Instruct")
        log(f"Tag {model_tag!r} not in config; treating as HF model name.")
        # Derive submission column from basename (e.g. "Qwen2.5-7B-Instruct")
        base_name = model_tag.split("/")[-1]
        # Default config structure using this model
        model_cfg = {
            "submission_column": base_name,
            "rca_rule": {"mode": "deterministic"}, # Default to deterministic if no Lora known
            "rca_ood": {"backend": "hf", "hf_model": model_tag},
            "general": {"backend": "hf", "hf_model": model_tag},
        }

    submission_column = model_cfg["submission_column"]

    out_model_dir = os.path.join(out_dir, model_tag)
    ensure_dir(out_model_dir)

    # Resolve dataset paths
    fn = cfg.get("filenames", {})
    sample_submission_path = _safe_join(data_dir, fn.get("sample_submission", "SampleSubmission.csv"))
    phase1_path = _safe_join(data_dir, fn.get("phase1_questions", "phase_1_test.csv"))
    phase2_path = _safe_join(data_dir, fn.get("phase2_questions", "phase_2_test.csv"))

    log("=" * 60)
    log("RCA Pipeline - Run Model")
    log("=" * 60)
    log(f"model_tag:         {model_tag}")
    log(f"submission_column: {submission_column}")
    log(f"data_dir:          {data_dir}")
    log(f"out_dir:           {out_model_dir}")
    if override_lora_model_dir:
        log(f"lora_model_dir:    {override_lora_model_dir} (CLI override)")
    log(f"RUN_ONLY:          {RUN_ONLY or '(default)'}")
    log("=" * 60)

    # Load template and force placeholder in other columns.
    rows, fieldnames = load_template(sample_submission_path)
    if submission_column not in fieldnames:
        raise SystemExit(f"Template missing column {submission_column!r}. Has: {fieldnames}")
    force_placeholder_else(rows, keep_column=submission_column)
    rid_index = row_id_index(rows)
    base_slots = base_slot_index(rows)

    # Load questions
    phase1_q = load_questions_csv(phase1_path) if os.path.exists(phase1_path) else {}
    phase2_q = load_questions_csv(phase2_path) if os.path.exists(phase2_path) else {}
    log(f"Loaded questions: phase1={len(phase1_q)} phase2={len(phase2_q)}")

    # Router + rule engine
    rp = load_reasoned_predict(data_dir)
    phase2_routes = route_questions_phase2(phase2_q, rp=rp) if phase2_q else {}

    # Split Phase2 into RCA (RULES), RCA_OOD, GENERAL
    phase2_rule: List[Tuple[str, str]] = []
    phase2_ood: List[Tuple[str, str]] = []
    phase2_gen: List[Tuple[str, str]] = []
    for bid, meta in phase2_routes.items():
        task_type = meta.get("task_type")
        if task_type == "RCA":
            phase2_rule.append((bid, phase2_q[bid]))
        elif task_type == "RCA_OOD":
            phase2_ood.append((bid, phase2_q[bid]))
        else:
            phase2_gen.append((bid, phase2_q[bid]))

    if RUN_ONLY == "RCA_RULE_PHASE2":
        phase2_ood = []
        phase2_gen = []
        phase1_q = {}
    elif RUN_ONLY == "RCA_RULE_ONLY":
        phase2_ood = []
        phase2_gen = []

    # Optional smoke-test limit (applies per-type)
    limit = cfg.get("run", {}).get("limit_base_ids")
    if limit is not None:
        n = int(limit)
        phase2_rule = phase2_rule[:n]
        phase2_ood = phase2_ood[:n]
        phase2_gen = phase2_gen[:n]
        phase1_items = list(phase1_q.items())[:n]
    else:
        phase1_items = list(phase1_q.items())

    log(f"Phase2 routed counts: RULE={len(phase2_rule)} RCA_OOD={len(phase2_ood)} GENERAL={len(phase2_gen)}")

    # Write routed question subsets (for reproducibility and for vote3 generator inputs)
    routed_rule_csv = os.path.join(out_model_dir, "phase2_rca_rule_questions.csv")
    routed_ood_csv = os.path.join(out_model_dir, "phase2_rca_ood_questions.csv")
    routed_gen_csv = os.path.join(out_model_dir, "phase2_general_questions.csv")
    write_questions_csv(routed_rule_csv, phase2_rule)
    write_questions_csv(routed_ood_csv, phase2_ood)
    write_questions_csv(routed_gen_csv, phase2_gen)

    # 1) RCA_OOD (Phase2)
    vote3_cfg = cfg.get("rca_ood_vote3", {})
    if phase2_ood:
        ood_sub = os.path.join(out_model_dir, "submission_rca_ood.csv")
        run_vote3_rca_ood(
            questions_csv=routed_ood_csv,
            submission_template_csv=sample_submission_path,
            submission_column=submission_column,
            out_submission_csv=ood_sub,
            out_audit_csv=os.path.join(out_model_dir, "audit_rca_ood.csv"),
            out_traces_jsonl=os.path.join(out_model_dir, "traces_rca_ood.jsonl"),
            vote3_cfg=vote3_cfg,
            backend_cfg=model_cfg.get("rca_ood", {}),
        )
        changed = _submission_overlay(
            dst_rows=rows,
            dst_row_index=rid_index,
            src_submission_csv=ood_sub,
            submission_column=submission_column,
        )
        log(f"Overlayed RCA_OOD rows: {changed}")
    else:
        log("No RCA_OOD questions to run.")

    # 2) GENERAL (Phase2)
    if phase2_gen:
        gen_sub = os.path.join(out_model_dir, "submission_general.csv")
        run_vote3_general(
            questions_csv=routed_gen_csv,
            submission_template_csv=sample_submission_path,
            submission_column=submission_column,
            out_submission_csv=gen_sub,
            out_audit_csv=os.path.join(out_model_dir, "audit_general.csv"),
            out_traces_jsonl=os.path.join(out_model_dir, "traces_general.jsonl"),
            vote3_cfg=vote3_cfg,
            backend_cfg=model_cfg.get("general", {}),
        )
        changed = _submission_overlay(
            dst_rows=rows,
            dst_row_index=rid_index,
            src_submission_csv=gen_sub,
            submission_column=submission_column,
        )
        log(f"Overlayed GENERAL rows: {changed}")
    else:
        log("No GENERAL questions to run.")

    # 3) RCA RULE (Phase1 + Phase2 routed RULES)
    if override_lora_model_dir:
        cfg["models"][model_tag].setdefault("rca_rule", {})
        cfg["models"][model_tag]["rca_rule"]["lora_model_dir"] = override_lora_model_dir
    rule_cfg = cfg["models"][model_tag].get("rca_rule", {})
    rule_prefix = vote3_cfg.get("prefix") or ""
    flatten_newlines = bool(cfg.get("csv", {}).get("flatten_newlines_in_cells", True))

    # Build union list of base IDs that are RULES (phase1 assumed RULE; phase2 uses router)
    rule_ids = [bid for bid, _ in phase1_items] + [bid for bid, _ in phase2_rule]
    rule_questions: Dict[str, str] = {bid: q for bid, q in phase1_items}
    for bid, q in phase2_rule:
        rule_questions[bid] = q

    if rule_ids:
        debug_jsonl = os.path.join(out_model_dir, "traces_rca_rule_debug.jsonl")
        traces = generate_rule_traces(
            base_ids=rule_ids,
            questions=rule_questions,
            rp=rp,
            mode_cfg=rule_cfg,
            out_debug_jsonl=debug_jsonl,
            prefix=str(rule_prefix),
            flatten_newlines=flatten_newlines,
        )
        filled = 0
        for bid, slots in traces.items():
            if bid not in base_slots:
                continue
            for k, cell in slots.items():
                idx = base_slots[bid].get(k)
                if idx is None:
                    continue
                rows[idx][submission_column] = cell
                filled += 1
        log(f"Filled RCA_RULE rows: {filled}")
    else:
        log("No RCA_RULE questions to run.")

    # Save final per-model submission
    out_submission = os.path.join(out_model_dir, f"submission_{model_tag}.csv")
    write_submission(out_submission, rows, fieldnames)
    log(f"Wrote per-model submission: {out_submission}")

    # Save reproducibility metadata
    meta = {
        "model_tag": model_tag,
        "submission_column": submission_column,
        "data_dir": os.path.abspath(data_dir),
        "paths": {
            "sample_submission": os.path.abspath(sample_submission_path),
            "phase1_questions": os.path.abspath(phase1_path) if os.path.exists(phase1_path) else None,
            "phase2_questions": os.path.abspath(phase2_path) if os.path.exists(phase2_path) else None,
        },
        "sha256": {
            "sample_submission": sha256_file(sample_submission_path),
            "phase1_questions": sha256_file(phase1_path) if os.path.exists(phase1_path) else None,
            "phase2_questions": sha256_file(phase2_path) if os.path.exists(phase2_path) else None,
        },
        "config_used": cfg,
    }
    with open(os.path.join(out_model_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log("Wrote run_metadata.json")


if __name__ == "__main__":
    main()

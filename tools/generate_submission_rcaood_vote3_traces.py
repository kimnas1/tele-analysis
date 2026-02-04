#!/usr/bin/env python3
"""
RCA_OOD/GENERAL submission generator: N-strategy vote per generation + trace injection.

For each base question ID, we generate 4 submission rows (<ID>_1.._4).
For each row:
  - Run N strategies (e.g., CoT + CoT2) to obtain N answers + N traces.
  - Majority-vote the FINAL answer key (tie-break configurable).
  - Pick ONE trace from a strategy whose answer == voted answer, using priority:
      configurable via CONFIG["trace_priority"]
  - Sanitize the trace so it cannot confuse the scorer:
      - strip ALL \\boxed{...} and \\\\boxed{...}
      - drop lines starting with FinalAnswer:/FINAL:/Final:
      - optionally collapse whitespace to keep the CSV robust
  - Append ONE final boxed answer at the end.

This matches the post-rescore rule: scorer extracts only the value inside \\boxed{...}.

Notes:
  - This script is meant to run on Kaggle/GPUs (transformers+torch).
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


CONFIG = {
    # Inputs
    # Select what to run:
    # - "RCA_OOD": merges sample50+other50 by default
    # - "GENERAL": runs the phase2 general questions
    # - "BOTH": runs RCA_OOD + GENERAL together into one submission file
    "RUN_MODE": "GENERAL",  # "RCA_OOD" | "GENERAL" | "BOTH"

    # Question sources (each CSV must have columns: ID,question)
    "rcaood_question_paths": [
        "qwen7b/phase2_rca_ood_questions_sample50_seed42.csv",
        "qwen7b/phase2_rca_ood_questions_other50_seed42.csv",
    ],
    "general_question_paths": [
        "qwen7b/phase2_general_questions.csv",
    ],
    # Optional override: if set to a list, this exact list is used regardless of RUN_MODE.
    "question_paths_override": None,

    "submission_template_csv": "data/SampleSubmission.csv",
    # Which track column to fill
    "submission_column": "Qwen2.5-7B-Instruct",
    # Output files
    # You can include "{mode}" in output paths; it will be replaced with RUN_MODE.lower().
    "out_submission_csv": "reasoning_experiments/submission_vote3_traces_{mode}.csv",
    "out_audit_csv": "reasoning_experiments/audit_vote3_traces_{mode}.csv",
    "out_traces_jsonl": "reasoning_experiments/traces_vote3_traces_{mode}.jsonl",
    # Backend: "hf" (local)
    "backend": "hf",
    # HF
    "hf_model": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "bfloat16",
    # Generation controls (passed to llm wrapper)
    # NOTE: For large RCA_OOD blobs, too-small token limits often truncates before the model
    # emits a final option, which then collapses to the default first key ("A").
    "max_tokens": 2000,
    "temperature": 0.3,
    "verbose_llm": False,
    # Run controls
    "seed": 42,
    "limit": None,  # Full run
    "n_gens": 4,  # always 4 to match submission format
    # Voting / trace selection priority
    # Fast mode: run a single structured CoT prompt.
    # - "cot2" = shorter/structured CoT prompt (more likely to emit an explicit key early)
    "strategies": ["cot2"],
    # Tie-break order (only relevant if multiple strategies are enabled).
    "answer_tie_break": ["cot2"],
    # If multiple match the voted answer, prefer which trace to keep.
    "trace_priority": ["cot2"],
    # Strategy knobs (passed into get_strategy(...)).
    # Leave empty to use each strategy's default settings.
    "strategy_params": {},
    # If a strategy output is missing an explicit boxed answer, do one low-token finalizer call
    # to recover a boxed key from the draft (so we don't silently fall back to "A").
    "finalize_if_unboxed": True,
    "finalizer_max_tokens": 32,
    "finalizer_temperature_hf": 0.2,
    # Output formatting
    "force_placeholder_else": True,
    "prefix": "",
    "trace_prefix": "",
    # Preserve the reasoning trace formatting by default.
    # Only strip boxed answers / final-answer lines; do NOT collapse whitespace unless you opt in.
    "collapse_whitespace": False,
    # 0 = no truncation
    "max_trace_chars": 0,
}


BOX_RE_ANY = re.compile(r"\\\\boxed\{[^}]*\}|\\boxed\{[^}]*\}")
FINAL_LINE_RE = re.compile(r"^\s*(FinalAnswer|FINAL|Final)\s*:\s*", flags=re.IGNORECASE)
BOX_VALUE_RE = re.compile(r"\\\\boxed\{([^}]*)\}|\\boxed\{([^}]*)\}")


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _maybe_seed_torch(seed: int) -> None:
    try:
        import random

        random.seed(seed)
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass(frozen=True)
class StrategyRun:
    answer: Optional[str]
    raw_output: str
    meta: dict


def load_questions(path: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "ID" not in r.fieldnames or "question" not in r.fieldnames:
            raise SystemExit(f"Questions CSV must have columns: ID,question. Got: {r.fieldnames}")
        for row in r:
            qid = (row.get("ID") or "").strip()
            if not qid:
                continue
            # Normalize to base ID if the input mistakenly contains a generation suffix (<ID>_1.._4).
            base = qid
            parts = qid.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in {"1", "2", "3", "4"}:
                base = parts[0]
            out.append((base, row.get("question") or ""))
    return out


def load_questions_many(paths: Sequence[str]) -> List[Tuple[str, str]]:
    seen: set[str] = set()
    out: List[Tuple[str, str]] = []
    for p in paths:
        qs = load_questions(p)
        for qid, qtext in qs:
            if qid in seen:
                raise ValueError(f"Duplicate question ID across inputs: {qid}")
            seen.add(qid)
            out.append((qid, qtext))
    return out


def load_template(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit(f"Submission template has no header: {path}")
        return list(r.fieldnames), list(r)


def sanitize_trace(text: str, *, collapse_whitespace: bool, max_chars: int) -> str:
    # 1) strip any boxed answers anywhere
    text = BOX_RE_ANY.sub("", text)
    # 2) drop FinalAnswer/FINAL lines
    kept: List[str] = []
    for ln in text.splitlines():
        if FINAL_LINE_RE.match(ln):
            continue
        kept.append(ln)
    text = "\n".join(kept).strip()
    # 3) collapse whitespace (recommended for CSV robustness)
    if collapse_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    # 4) truncate
    if max_chars and len(text) > int(max_chars):
        text = text[: int(max_chars)].rstrip() + "…"
    # 5) make sure no boxed remains
    text = BOX_RE_ANY.sub("", text).strip()
    return text


def majority_vote(
    answers_by_strategy: Dict[str, Optional[str]],
    *,
    answer_tie_break: Sequence[str],
) -> Tuple[str, str]:
    """
    Returns: (picked_answer, reason)
    """
    vals = [v for v in answers_by_strategy.values() if v]
    if not vals:
        return "A", "missing_all(default_A)"
    counts = Counter(vals)
    best, best_n = counts.most_common(1)[0]
    if best_n >= 2:
        return best, "majority"
    # 3-way tie: use configured tie-break order
    for name in answer_tie_break:
        v = answers_by_strategy.get(name)
        if v:
            return v, f"tie_break:{name}"
    return vals[0], "tie_break:first_non_null"


def pick_trace_strategy(
    *,
    voted_answer: str,
    answers_by_strategy: Dict[str, Optional[str]],
    trace_priority: Sequence[str],
) -> Optional[str]:
    """
    Choose which strategy's trace to keep (must match the voted answer).
    """
    for name in trace_priority:
        if answers_by_strategy.get(name) == voted_answer:
            return name
    return None


def render_cell(prefix: str, trace_prefix: str, trace: str, answer: str) -> str:
    # Single boxed answer at the end.
    if trace:
        return f"{trace_prefix}{trace} {prefix}\\boxed{{{answer}}}"
    return f"{prefix}\\boxed{{{answer}}}"


def flatten_cell_for_single_line_csv(text: str) -> str:
    """
    Some evaluation scripts cannot parse CSV fields with embedded newlines
    (even if quoted correctly). Make each cell a single physical line.
    """
    # Normalize all newlines to spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")
    # Collapse repeated whitespace to keep size reasonable.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_traces_cache(path: str) -> tuple[Dict[str, str], Dict[str, dict]]:
    """
    Load a prior traces JSONL file as a cache.

    Returns:
      - fill_map: row_id -> rendered cell text
      - audit_map: row_id -> audit row
    """
    fill_map: Dict[str, str] = {}
    audit_map: Dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return fill_map, audit_map

    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        qid = str(obj.get("ID") or "").strip()
        gen = obj.get("gen")
        if not qid or not isinstance(gen, int):
            continue
        rid = f"{qid}_{gen}"

        cell = obj.get("cell")
        if isinstance(cell, str) and cell:
            fill_map[rid] = cell
        else:
            # Backward compat: reconstruct cell if older cache lines don't include it.
            voted = str(obj.get("voted") or "").strip()
            trace_raw = str(obj.get("trace_raw") or "")
            trace = sanitize_trace(
                trace_raw,
                collapse_whitespace=bool(CONFIG["collapse_whitespace"]),
                max_chars=int(CONFIG["max_trace_chars"]),
            )
            fill_map[rid] = flatten_cell_for_single_line_csv(render_cell(CONFIG["prefix"], CONFIG["trace_prefix"], trace, voted))

        keys = obj.get("keys") or []
        answers_by_strategy = obj.get("answers_by_strategy") or {}
        meta_by_strategy = obj.get("meta_by_strategy") or {}

        audit_row = {
            "ID": qid,
            "gen": gen,
            "keys": ",".join(keys) if isinstance(keys, list) else str(keys),
            "voted": obj.get("voted"),
            "vote_reason": obj.get("vote_reason"),
            "trace_strategy": obj.get("trace_strategy"),
        }
        for name in list(CONFIG["strategies"]):
            audit_row[name] = answers_by_strategy.get(name)
            m = meta_by_strategy.get(name) or {}
            audit_row[f"{name}_had_box"] = m.get("had_box")
            audit_row[f"{name}_used_finalizer"] = m.get("used_finalizer")
        audit_map[rid] = audit_row

    return fill_map, audit_map


def _extract_boxed_key(text: str, keys: Sequence[str]) -> Optional[str]:
    """
    Extract the last boxed value (\\boxed{...}) and normalize to one of keys.
    Returns None if no boxed value.
    """
    from reasoning_strategies.base import normalize_to_keys  # type: ignore

    matches = list(BOX_VALUE_RE.finditer(text))
    if not matches:
        return None
    for m in reversed(matches):
        v = (m.group(1) or m.group(2) or "").strip()
        if v:
            return normalize_to_keys(v, keys)
    return None


def _strict_extract_key(text: str, keys: Sequence[str]) -> Optional[str]:
    """
    Stricter extractor than "last token":
      1) boxed value
      2) explicit final lines / tags
      3) 'Option X' / 'choose X' near the end
    Returns None if not confident.
    """
    from reasoning_strategies.base import normalize_to_keys  # type: ignore

    boxed = _extract_boxed_key(text, keys)
    if boxed:
        return boxed

    m = re.findall(r"^\s*(?:FinalAnswer|FINAL|Final)\s*:\s*([A-Za-z0-9]+)\b", text, flags=re.MULTILINE)
    if m:
        return normalize_to_keys(m[-1], keys)

    for tag in ("CANDIDATE", "PROPOSED", "DRAFT", "REVISED", "ANSWER"):
        m = re.findall(rf"^\s*{tag}\s*:\s*([A-Za-z0-9]+)\b", text, flags=re.MULTILINE)
        if m:
            return normalize_to_keys(m[-1], keys)

    tail = text[-1200:] if len(text) > 1200 else text
    alts = sorted([str(k) for k in keys], key=len, reverse=True)
    alts_re = "|".join(re.escape(k) for k in alts)

    m = re.findall(rf"\boption\s+({alts_re})\b", tail, flags=re.IGNORECASE)
    if m:
        return normalize_to_keys(m[-1], keys)
    m = re.findall(rf"\b(?:choose|pick|select)\s+({alts_re})\b", tail, flags=re.IGNORECASE)
    if m:
        return normalize_to_keys(m[-1], keys)

    return None


def _finalize_box_only(llm, *, backend: str, keys: Sequence[str], draft_text: str) -> Optional[str]:
    """
    If an output is unboxed/truncated, ask for ONLY \\boxed{X} using the draft as context.
    Avoids resending the full (huge) question blob.
    """
    from reasoning_strategies.base import normalize_to_keys  # type: ignore

    keys_str = ", ".join(str(k) for k in keys)
    prompt = (
        "You are a strict formatter.\n"
        f"Allowed option keys: {keys_str}\n\n"
        "Given the DRAFT below (may be truncated), output ONLY the final answer in the form:\n"
        "\\\\boxed{<ONE_OPTION_KEY>}\n\n"
        "Do not output any other text.\n\n"
        "DRAFT:\n"
        f"{draft_text}\n"
    )
    # Finalizer temperature
    temp = float(CONFIG["finalizer_temperature_hf"])

    out = llm.generate(
        prompt,
        system_message="You output only a single boxed option key.",
        max_new_tokens=int(CONFIG["finalizer_max_tokens"]),
        temperature=temp,
    )
    boxed = _extract_boxed_key(out, keys)
    if boxed:
        return boxed
    m = re.search(r"([A-Za-z0-9]+)", out.strip())
    if m:
        return normalize_to_keys(m.group(1), keys)
    return None


def main() -> None:
    from reasoning_strategies import create_llm, extract_option_keys, get_strategy  # type: ignore

    mode = str(CONFIG.get("RUN_MODE", "RCA_OOD")).upper().strip()
    if mode not in {"RCA_OOD", "GENERAL", "BOTH"}:
        raise SystemExit("CONFIG['RUN_MODE'] must be one of: RCA_OOD, GENERAL, BOTH")

    def _fmt_out(p: str) -> str:
        return p.replace("{mode}", mode.lower())

    out_submission = _fmt_out(CONFIG["out_submission_csv"])
    out_audit = _fmt_out(CONFIG.get("out_audit_csv") or "") if CONFIG.get("out_audit_csv") else ""
    out_traces_path = _fmt_out(CONFIG.get("out_traces_jsonl") or "") if CONFIG.get("out_traces_jsonl") else ""
    os.makedirs(os.path.dirname(out_submission), exist_ok=True)

    override_paths = CONFIG.get("question_paths_override")
    if override_paths:
        question_paths = list(override_paths)
    elif mode == "RCA_OOD":
        question_paths = list(CONFIG["rcaood_question_paths"])
    elif mode == "GENERAL":
        question_paths = list(CONFIG["general_question_paths"])
    else:  # BOTH
        question_paths = list(CONFIG["rcaood_question_paths"]) + list(CONFIG["general_question_paths"])

    questions = load_questions_many(question_paths)
    if CONFIG.get("limit"):
        questions = questions[: int(CONFIG["limit"])]

    fieldnames, template_rows = load_template(CONFIG["submission_template_csv"])
    submission_col = CONFIG["submission_column"]
    if submission_col not in fieldnames:
        raise SystemExit(f"Template missing column {submission_col!r}. Has: {fieldnames}")

    log("=" * 60)
    log("Vote + Trace Injection")
    log("=" * 60)
    log(f"RUN_MODE:   {mode}")
    log(f"Questions:  {question_paths} ({len(questions)})")
    log(f"Template:  {CONFIG['submission_template_csv']} ({len(template_rows)} rows)")
    log(f"Column:    {submission_col}")
    log(f"Backend:   {CONFIG['backend']}")
    log("=" * 60)
    backend = str(CONFIG["backend"]).lower()

    llm = create_llm(
        backend="hf",
        model=CONFIG["hf_model"],
        dtype=CONFIG["dtype"],
        max_tokens=int(CONFIG["max_tokens"]),
        temperature=float(CONFIG["temperature"]),
        verbose=bool(CONFIG["verbose_llm"]),
    )

    # Prepare strategy instances.
    strategies = list(CONFIG["strategies"])
    strategy_objs = {name: get_strategy(name, llm=llm, verbose=False, **CONFIG.get("strategy_params", {}).get(name, {})) for name in strategies}  # type: ignore

    # row_id -> cell text
    fill_map: Dict[str, str] = {}
    # row_id -> audit row (prevents duplicates on resume)
    audit_map: Dict[str, dict] = {}

    # Cache logic removed as requested. Always fresh run.

    traces_f = None
    if out_traces_path:
        os.makedirs(os.path.dirname(out_traces_path) or ".", exist_ok=True)
        traces_f = open(
            out_traces_path,
            "w", # Always fresh write
            encoding="utf-8",
        )

    try:
        base_seed = int(CONFIG["seed"])
        n_gens = int(CONFIG["n_gens"])

        for qi, (qid, qtext) in enumerate(questions, 1):
            keys = extract_option_keys(qtext)
            
            # Use deterministic seed for the single generation
            gen = 1 
            rid = f"{qid}_{gen}"
            
            # Check if we already have any slot for this question (resume logic)
            # If we have slot 1, we assume we have everything for this ID
            if rid not in fill_map:
                runs: Dict[str, StrategyRun] = {}
                for si, name in enumerate(strategies):
                    seed = base_seed + (qi * 10_000) + (si * 100) + 7
                    if backend == "hf":
                        _maybe_seed_torch(seed)

                    t0 = time.time()
                    res = strategy_objs[name].solve(qtext, keys)  # type: ignore
                    dt_ms = (time.time() - t0) * 1000
                    raw = str(res.raw_output)
                    boxed_key = _extract_boxed_key(raw, keys)
                    extracted = _strict_extract_key(raw, keys)
                    used_finalizer = False
                    if extracted is None and bool(CONFIG["finalize_if_unboxed"]) and boxed_key is None:
                        extracted = _finalize_box_only(
                            llm,
                            backend=backend,
                            keys=keys,
                            draft_text=raw[-2500:] if len(raw) > 2500 else raw,
                        )
                        used_finalizer = True
                    runs[name] = StrategyRun(
                        answer=extracted,
                        raw_output=raw,
                        meta={
                            "strategy": name,
                            "seed": seed,
                            "keys": list(keys),
                            "llm_calls": int(getattr(res, "num_llm_calls", 0)),
                            "duration_ms": float(getattr(res, "total_duration_ms", dt_ms)),
                            "had_box": bool(boxed_key),
                            "used_finalizer": used_finalizer,
                        },
                    )

                answers_by_strategy: Dict[str, Optional[str]] = {k: v.answer for k, v in runs.items()}
                voted, vote_reason = majority_vote(
                    answers_by_strategy,
                    answer_tie_break=CONFIG["answer_tie_break"],
                )
                chosen = pick_trace_strategy(
                    voted_answer=voted,
                    answers_by_strategy=answers_by_strategy,
                    trace_priority=CONFIG["trace_priority"],
                )
                chosen_trace_raw = runs[chosen].raw_output if chosen else ""
                trace = sanitize_trace(
                    chosen_trace_raw,
                    collapse_whitespace=bool(CONFIG["collapse_whitespace"]),
                    max_chars=int(CONFIG["max_trace_chars"]),
                )
                cell = render_cell(CONFIG["prefix"], CONFIG["trace_prefix"], trace, voted)
                cell = flatten_cell_for_single_line_csv(cell)
                
                # REPLICATE to all 4 slots locally and in the files
                for g in range(1, n_gens + 1):
                    target_rid = f"{qid}_{g}"
                    fill_map[target_rid] = cell
                    
                    audit_row = {
                        "ID": qid,
                        "gen": g,
                        "keys": ",".join(keys),
                        "voted": voted,
                        "vote_reason": vote_reason,
                        "trace_strategy": chosen,
                    }
                    for name in strategies:
                        audit_row[name] = answers_by_strategy.get(name)
                        audit_row[f"{name}_had_box"] = runs[name].meta.get("had_box")
                        audit_row[f"{name}_used_finalizer"] = runs[name].meta.get("used_finalizer")
                    audit_map[target_rid] = audit_row

                    if traces_f:
                        traces_f.write(
                            json.dumps(
                                {
                                    "ID": qid,
                                    "gen": g,
                                    "keys": list(keys),
                                    "answers_by_strategy": answers_by_strategy,
                                    "voted": voted,
                                    "vote_reason": vote_reason,
                                    "trace_strategy": chosen,
                                    "trace_raw": chosen_trace_raw,
                                    "cell": cell,
                                    "meta_by_strategy": {k: v.meta for k, v in runs.items()},
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            done = sum(1 for g in range(1, n_gens + 1) if f"{qid}_{g}" in fill_map)
            log(f"[{qi:04d}/{len(questions)}] {qid} -> Replicated 1 -> {done}/{n_gens} slots")

    finally:
        if traces_f:
            traces_f.close()
            log(f"Wrote traces: {out_traces_path}")

    # Write submission: reset all columns to placeholder; fill only requested column where we have row IDs.
    force_placeholder_else = bool(CONFIG["force_placeholder_else"])

    track_cols = [c for c in fieldnames if c != "ID"]
    filled_rows = 0
    with open(out_submission, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in template_rows:
            rid = row.get("ID") or ""
            for c in track_cols:
                row[c] = "placeholder"
            if rid in fill_map:
                row[submission_col] = fill_map[rid]
                filled_rows += 1
            elif force_placeholder_else:
                row[submission_col] = "placeholder"
            w.writerow(row)

    log(f"Wrote submission: {out_submission} (filled rows: {filled_rows})")

    # Audit CSV
    audit_rows: List[dict] = list(audit_map.values())
    if out_audit:
        os.makedirs(os.path.dirname(out_audit), exist_ok=True)
        with open(out_audit, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()) if audit_rows else ["ID"])
            w.writeheader()
            w.writerows(audit_rows)
        log(f"Wrote audit: {out_audit}")

    log("Done.")


if __name__ == "__main__":
    main()

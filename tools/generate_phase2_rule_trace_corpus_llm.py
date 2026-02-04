#!/usr/bin/env python3
"""
Phase2 RCA_RULE trace corpus generator (LLM-written justifications).

Purpose
-------
You already have high-accuracy deterministic RCA rules (semantic C1..C8).
Phase2 uses *question-specific option keys* (e.g., 7 / Z4 / C2) inside \\boxed{}.

This script:
1) Routes Phase2 questions -> RCA (RULE) using the existing router.
2) Runs the deterministic rule engine to choose the semantic label (internal only).
3) Maps semantic label -> the question's option key (answer_value).
4) Calls an LLM (Cerebras OpenAI-compatible API) to write a grounded trace
   using the exact TriggeredRule/Evidence/Rejections blocks we provide.

Important: the prompt DOES NOT expose the semantic label C# to the model.

Output
------
CSV (quoted) with:
  ID,answer_value,answer_schema,triggered_rule,trace_source,attempts,trace

Resume / crash-safety:
  - evidence cache JSONL (per base ID)
  - traces cache JSONL (per base ID)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# Allow running as `python3 tools/generate_phase2_rule_trace_corpus_llm.py ...`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rca.router import load_reasoned_predict, route_questions_phase2  # noqa: E402
from rca.util import log  # noqa: E402


BOX_RE = re.compile(r"\\\\boxed\{([^}]*)\}|\\boxed\{([^}]*)\}")
_OPT_LINE_RE = re.compile(r"^\s*([A-Za-z]{1,3}\d{1,2}|C[1-8]|[A-I]|[1-9])\s*[:\)]\s*(.+?)\s*$")


SYSTEM_PROMPT = "You are a staff 5G RAN engineer writing a grounded RCA justification trace."

PROMPT_JUSTIFY = """A deterministic rule-engine already selected the correct answer key for THIS question.

CorrectAnswerValue (copy exactly; do not change): __ANSWER_VALUE__

Hard rules (must follow):
- Copy the TriggeredRule/Evidence/Rejections blocks EXACTLY as given.
- Do NOT introduce any numeric values not already present in those blocks.
- Add ONE short Decision bullet (no new numbers).
- The LAST line must contain exactly one boxed answer using CorrectAnswerValue.

TriggeredRule line (copy EXACTLY):
__TRIGGERED_RULE_LINE__

Evidence block (copy EXACTLY):
Evidence:
__EVIDENCE_BULLETS__

Rejections block (copy EXACTLY):
Rejections:
__REJECTION_BULLETS__

Now output in this STRICT format:
__TRIGGERED_RULE_LINE__

Evidence:
__EVIDENCE_BULLETS__

Rejections:
__REJECTION_BULLETS__

Decision:
- <one short grounded decision>

FinalAnswer: \\boxed{__ANSWER_VALUE__}
"""


PROMPT_REPAIR = """You wrote an invalid trace. Fix it.

Errors:
__ERRORS__

Remember:
- Copy TriggeredRule/Evidence/Rejections EXACTLY.
- Do NOT add numbers not present in those blocks.
- Final line must be: FinalAnswer: \\boxed{__ANSWER_VALUE__}

TriggeredRule line (copy EXACTLY):
__TRIGGERED_RULE_LINE__

Evidence block (copy EXACTLY):
Evidence:
__EVIDENCE_BULLETS__

Rejections block (copy EXACTLY):
Rejections:
__REJECTION_BULLETS__
"""


def load_questions_csv(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"Missing ID column: {path}")
    qcol = None
    for c in df.columns:
        if c.lower() in ("question", "prompt", "text"):
            qcol = c
            break
    if qcol is None:
        qcol = next(c for c in df.columns if c != "ID")
    out: Dict[str, str] = {}
    for _, r in df.iterrows():
        out[str(r["ID"]).strip()] = str(r[qcol])
    return out


@dataclass(frozen=True)
class OptionMap:
    keys: List[str]
    key_to_text: Dict[str, str]


def parse_options(question_text: str) -> OptionMap:
    keys: List[str] = []
    key_to_text: Dict[str, str] = {}
    for ln in (question_text or "").splitlines():
        m = _OPT_LINE_RE.match(ln)
        if not m:
            continue
        k = m.group(1).strip()
        t = m.group(2).strip()
        if not k or not t:
            continue
        if k not in key_to_text:
            keys.append(k)
            key_to_text[k] = t
    return OptionMap(keys=keys, key_to_text=key_to_text)


def semantic_from_option_text(text: str) -> Optional[str]:
    # Heuristic aligned with earlier work (same as rca/rca_rule/generate.py)
    t = (text or "").lower()
    if ("speed" in t) and ("40" in t or "40km" in t):
        return "C7"
    if ("scheduled" in t and "rb" in t) and ("160" in t):
        return "C8"
    if ("coverage distance" in t and "1km" in t) or ("exceeds 1km" in t) or ("over-shoot" in t) or ("overshoot" in t):
        return "C2"
    if ("pci" in t and "mod" in t and "30" in t) or ("mod 30" in t):
        return "C6"
    if "frequent handover" in t or ("frequent" in t and "handover" in t):
        return "C5"
    if "non-colocated" in t and ("overlap" in t or "overlapping coverage" in t):
        return "C4"
    if ("neighbor" in t or "neighbour" in t) and "higher throughput" in t:
        return "C3"
    if "downtilt" in t and ("too large" in t or "weak coverage" in t or "far end" in t):
        return "C1"
    return None


def map_label_to_option_key(question_text: str, fixed_label: str) -> Tuple[Optional[str], str]:
    opts = parse_options(question_text)
    if not opts.keys:
        return None, "no_options"
    if fixed_label in opts.key_to_text:
        return fixed_label, "direct_key"
    semantic_to_key: Dict[str, str] = {}
    for k in opts.keys:
        s = semantic_from_option_text(opts.key_to_text.get(k, ""))
        if s and s not in semantic_to_key:
            semantic_to_key[s] = k
    if fixed_label in semantic_to_key:
        return semantic_to_key[fixed_label], "semantic_map"
    return None, "unmapped"


def infer_triggered_rule(rule_reason: str) -> str:
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


def metrics_to_derived_dict(metrics: Any, status: str) -> dict[str, Any]:
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
        "rsrpMedLow": getattr(m, "rsrp_med_low", None),
        "tiltMedLow": getattr(m, "tilt_med_low", None),
        "tiltMaxLow": getattr(m, "tilt_max_low", None),
        "c1StrongPcis": list(getattr(m, "c1_strong_pcis", []) or []),
    }


def _fmt_num(x: Any, ndigits: int) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def build_triggered_rule_line(*, trig_code: str, derived: dict[str, Any], rp: Any) -> str:
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    if trig_code == "R1_speedMax_gt_40":
        return f"TriggeredRule: speedMax > {int(speed_thr)} (speedMax={_fmt_num(derived.get('speedMax'),1)}, threshold={int(speed_thr)})"
    if trig_code == "R2_avgRB_low_lt_160":
        return f"TriggeredRule: avgRB_low < {int(rb_thr)} (avgRB_low={_fmt_num(derived.get('avgRB_low'),1)}, threshold={int(rb_thr)})"
    if trig_code == "R3_maxDist_m_gt_1000":
        return f"TriggeredRule: maxDist_m > {int(dist_thr)} (maxDist_m={_fmt_num(derived.get('maxDist_m'),1)}, threshold={int(dist_thr)})"
    if trig_code == "R4_mod30Frac_ge_0.75":
        return f"TriggeredRule: mod30Frac >= {mod30_thr} (mod30Frac={_fmt_num(derived.get('mod30Frac'),2)}, threshold={mod30_thr})"
    if trig_code == "R5_handovers_ge_2":
        return f"TriggeredRule: handovers >= {ho_thr} (handovers={derived.get('handovers')}, threshold={ho_thr})"
    if trig_code == "R6_diffGnbCloseFrac_eq_1":
        return f"TriggeredRule: diffGnbCloseFrac == {diff_thr} (diffGnbCloseFrac={_fmt_num(derived.get('diffGnbCloseFrac'),2)}, threshold={diff_thr})"
    if trig_code == "R7_c1StrongPcis_nonempty":
        return f"TriggeredRule: c1StrongPcis non-empty (c1StrongPcis={derived.get('c1StrongPcis')}, threshold=nonempty)"
    if trig_code.startswith("R8_"):
        # Same-site / C1 overrides
        return f"TriggeredRule: sameGnbCloseFrac == 1.0 (sameGnbCloseFrac={_fmt_num(derived.get('sameGnbCloseFrac'),2)}, threshold=1.0)"
    if trig_code == "R9_beyondBeamRows_gt_0":
        return f"TriggeredRule: beyondBeamRows > 0 (beyondBeamRows={derived.get('beyondBeamRows')}, threshold=0)"
    return "TriggeredRule: unknown (threshold=unknown)"


def build_evidence_bullets(*, trig_code: str, derived: dict[str, Any]) -> List[str]:
    keys = ["speedMax", "avgRB_low", "maxDist_m", "handovers"]
    if trig_code in ("R4_mod30Frac_ge_0.75",):
        keys[-1] = "mod30Frac"
    elif trig_code in ("R6_diffGnbCloseFrac_eq_1",):
        keys[-1] = "diffGnbCloseFrac"
    elif trig_code.startswith("R8_"):
        keys[-1] = "sameGnbCloseFrac"
    elif trig_code in ("R9_beyondBeamRows_gt_0",):
        keys[-1] = "beyondBeamRows"
    elif trig_code in ("R7_c1StrongPcis_nonempty",):
        keys[-1] = "c1StrongPcis"

    out: List[str] = []
    for k in keys:
        if k == "speedMax":
            out.append(f"- speedMax={_fmt_num(derived.get('speedMax'),1)}")
        elif k == "avgRB_low":
            out.append(f"- avgRB_low={_fmt_num(derived.get('avgRB_low'),1)}")
        elif k == "maxDist_m":
            out.append(f"- maxDist_m={_fmt_num(derived.get('maxDist_m'),1)}")
        elif k == "handovers":
            out.append(f"- handovers={derived.get('handovers')}")
        elif k in ("mod30Frac", "sameGnbCloseFrac", "diffGnbCloseFrac"):
            out.append(f"- {k}={_fmt_num(derived.get(k),2)}")
        elif k == "beyondBeamRows":
            out.append(f"- beyondBeamRows={derived.get('beyondBeamRows')}")
        elif k == "c1StrongPcis":
            out.append(f"- c1StrongPcis={derived.get('c1StrongPcis')}")
        else:
            out.append(f"- {k}={derived.get(k)}")
    return out


def build_rejection_bullets(*, trig_code: str, derived: dict[str, Any], rp: Any, triggered_rule_line: str) -> List[str]:
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    rules = [
        ("R1_speedMax_gt_40", "speedMax", _fmt_num(derived.get("speedMax"), 1), f"speedMax>{int(speed_thr)}", str(int(speed_thr))),
        ("R2_avgRB_low_lt_160", "avgRB_low", _fmt_num(derived.get("avgRB_low"), 1), f"avgRB_low<{int(rb_thr)}", str(int(rb_thr))),
        ("R3_maxDist_m_gt_1000", "maxDist_m", _fmt_num(derived.get("maxDist_m"), 1), f"maxDist_m>{int(dist_thr)}", str(int(dist_thr))),
        ("R4_mod30Frac_ge_0.75", "mod30Frac", _fmt_num(derived.get("mod30Frac"), 2), f"mod30Frac>={mod30_thr}", str(mod30_thr)),
        ("R5_handovers_ge_2", "handovers", str(derived.get("handovers")), f"handovers>={ho_thr}", str(ho_thr)),
        ("R6_diffGnbCloseFrac_eq_1", "diffGnbCloseFrac", _fmt_num(derived.get("diffGnbCloseFrac"), 2), f"diffGnbCloseFrac=={diff_thr}", str(diff_thr)),
    ]

    out: List[str] = []
    for code, metric, mval, cond, thr in rules:
        if code == trig_code:
            continue
        out.append(f"- Rule rejected because {metric}={mval} does not meet {cond} (threshold={thr})")
        if len(out) >= 3:
            break
    out.append(f"- Other rules rejected because a different TriggeredRule fired: {triggered_rule_line}")
    return out


def normalize_newlines(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_boxed(text: str) -> Optional[str]:
    m = BOX_RE.search(text or "")
    if not m:
        return None
    return (m.group(1) or m.group(2) or "").strip()


def validate_trace(*, trace: str, answer_value: str, trig_line: str, ev_lines: List[str], rej_lines: List[str]) -> List[str]:
    errors: List[str] = []
    t = normalize_newlines(trace)

    if f"FinalAnswer: \\boxed{{{answer_value}}}" not in t:
        errors.append("Missing or wrong final boxed answer line.")

    if trig_line not in t:
        errors.append("TriggeredRule line not copied exactly.")

    # Extract blocks to ensure bullets match exactly.
    def _block(label: str) -> Optional[str]:
        idx = t.find(label + "\n")
        return None if idx < 0 else t[idx:]

    if "Evidence:" not in t:
        errors.append("Missing Evidence: block.")
    else:
        # Ensure every provided evidence bullet exists.
        for ln in ev_lines:
            if ln not in t:
                errors.append(f"Evidence bullet missing: {ln}")
                break

    if "Rejections:" not in t:
        errors.append("Missing Rejections: block.")
    else:
        for ln in rej_lines:
            if ln not in t:
                errors.append(f"Rejection bullet missing: {ln}")
                break

    return errors


class CerebrasBackend:
    def __init__(self, *, model: str, api_key: str, base_url: str = "https://api.cerebras.ai/v1") -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat(self, *, system: str, user: str, temperature: float, max_tokens: int, seed: Optional[int]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        # Some deployments support seed; safe to include only if provided.
        if seed is not None:
            payload["seed"] = int(seed)

        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"]["content"])


def load_jsonl_by_id(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    out: Dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[str(obj["ID"])] = obj
    return out


def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Dir used to locate reasoned_predict.py (default: .)")
    ap.add_argument("--phase2", default="phase_2_test.csv", help="Phase2 questions CSV (ID,question)")
    ap.add_argument("--out-dir", default="tools/rule_trace_corpus_phase2_llm", help="Output directory")
    ap.add_argument("--out-csv", default="traces_phase2_rule_llm.csv", help="Output CSV filename (inside out-dir)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of base IDs (RCA only)")
    ap.add_argument("--model", default="qwen3-32b", help="Cerebras model name")
    ap.add_argument("--max-tokens", type=int, default=900, help="Max tokens for the trace generation")
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for trace generation")
    ap.add_argument("--attempts", type=int, default=4, help="Max repair attempts before deterministic fallback")
    ap.add_argument("--seed", type=int, default=42, help="Base seed (best-effort)")
    args = ap.parse_args()

    api_key = os.environ.get("CEREBRAS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing CEREBRAS_API_KEY env var.")

    os.makedirs(args.out_dir, exist_ok=True)
    evidence_jsonl = os.path.join(args.out_dir, "evidence_phase2_rule.jsonl")
    traces_jsonl = os.path.join(args.out_dir, "traces_phase2_rule_llm.jsonl")
    out_csv = os.path.join(args.out_dir, args.out_csv)

    rp = load_reasoned_predict(args.data_dir)
    phase2 = load_questions_csv(args.phase2)
    routes = route_questions_phase2(phase2, rp=rp)
    rca_ids = [bid for bid, meta in routes.items() if meta.get("task_type") == "RCA"]
    rca_ids.sort()
    if args.limit is not None:
        rca_ids = rca_ids[: int(args.limit)]

    log(f"Phase2 total={len(phase2)} routed_RCA={sum(1 for b in routes if routes[b].get('task_type')=='RCA')} using={len(rca_ids)}")

    evidence_cache = load_jsonl_by_id(evidence_jsonl)
    trace_cache = load_jsonl_by_id(traces_jsonl)

    backend = CerebrasBackend(model=args.model, api_key=api_key)

    schema_ctr: Counter[str] = Counter()
    key_style_ctr: Counter[str] = Counter()
    source_ctr: Counter[str] = Counter()

    rows_out: List[dict[str, Any]] = []

    for i, bid in enumerate(tqdm(rca_ids, desc="Phase2 RCA traces"), 1):
        q = phase2[bid]

        if bid in trace_cache:
            obj = trace_cache[bid]
            rows_out.append(obj)
            continue

        # 1) EvidencePack (deterministic)
        if bid in evidence_cache:
            ev = evidence_cache[bid]
        else:
            metrics, status = rp.compute_metrics(q)
            fixed_label, rule_reason = rp.decide(metrics, status)  # semantic, internal
            derived = metrics_to_derived_dict(metrics, status)
            trig_code = infer_triggered_rule(rule_reason)
            trig_line = build_triggered_rule_line(trig_code=trig_code, derived=derived, rp=rp)
            ev_lines = build_evidence_bullets(trig_code=trig_code, derived=derived)
            rej_lines = build_rejection_bullets(trig_code=trig_code, derived=derived, rp=rp, triggered_rule_line=trig_line)

            answer_value, schema = map_label_to_option_key(q, fixed_label)
            if answer_value is None:
                answer_value = fixed_label  # fallback
                schema = "fallback_unmapped"

            ev = {
                "ID": bid,
                "answer_value": str(answer_value),
                "answer_schema": schema,
                "triggered_rule": trig_code,
                "triggered_rule_line": trig_line,
                "evidence_bullets": ev_lines,
                "rejection_bullets": rej_lines,
                "derived": derived,
            }
            append_jsonl(evidence_jsonl, ev)
            evidence_cache[bid] = ev

        answer_value = str(ev["answer_value"])
        schema = str(ev["answer_schema"])
        trig_code = str(ev["triggered_rule"])
        trig_line = str(ev["triggered_rule_line"])
        ev_lines = list(ev["evidence_bullets"])
        rej_lines = list(ev["rejection_bullets"])

        schema_ctr[schema] += 1
        if re.fullmatch(r"[1-9]", answer_value):
            key_style_ctr["digit"] += 1
        elif re.fullmatch(r"[A-I]", answer_value):
            key_style_ctr["letter"] += 1
        elif re.fullmatch(r"C[1-8]", answer_value):
            key_style_ctr["C#"] += 1
        elif re.fullmatch(r"[A-Za-z]{1,3}\d{1,2}", answer_value):
            key_style_ctr["alphanum"] += 1
        else:
            key_style_ctr["other"] += 1

        base_prompt = (
            PROMPT_JUSTIFY.replace("__ANSWER_VALUE__", answer_value)
            .replace("__TRIGGERED_RULE_LINE__", trig_line)
            .replace("__EVIDENCE_BULLETS__", "\n".join(ev_lines))
            .replace("__REJECTION_BULLETS__", "\n".join(rej_lines))
        )

        last = ""
        used = "llm"
        for attempt in range(1, int(args.attempts) + 1):
            seed = int(args.seed) + i * 7 + attempt
            prompt = base_prompt if attempt == 1 else (
                PROMPT_REPAIR.replace("__ERRORS__", "\n".join(errors))
                .replace("__ANSWER_VALUE__", answer_value)
                .replace("__TRIGGERED_RULE_LINE__", trig_line)
                .replace("__EVIDENCE_BULLETS__", "\n".join(ev_lines))
                .replace("__REJECTION_BULLETS__", "\n".join(rej_lines))
            )

            try:
                last = backend.chat(
                    system=SYSTEM_PROMPT,
                    user=prompt,
                    temperature=float(args.temperature),
                    max_tokens=int(args.max_tokens),
                    seed=seed,
                )
            except Exception as e:
                errors = [f"API error: {type(e).__name__}: {e}"]
                time.sleep(0.5)
                continue

            errors = validate_trace(
                trace=last,
                answer_value=answer_value,
                trig_line=trig_line,
                ev_lines=ev_lines,
                rej_lines=rej_lines,
            )
            if not errors:
                used = "llm"
                break

            # brief backoff
            time.sleep(0.2)
        else:
            # Deterministic fallback (still uses answer_value)
            used = "fallback"
            last = (
                f"{trig_line}\n\n"
                "Evidence:\n"
                + "\n".join(ev_lines)
                + "\n\nRejections:\n"
                + "\n".join(rej_lines)
                + "\n\nDecision:\n"
                + "- The TriggeredRule is satisfied for the low-throughput window, so this option key is correct; higher-priority rules were rejected.\n\n"
                + f"FinalAnswer: \\boxed{{{answer_value}}}"
            )

        obj = {
            "ID": bid,
            "answer_value": answer_value,
            "answer_schema": schema,
            "triggered_rule": trig_code,
            "trace_source": used,
            "attempts": int(args.attempts) if used == "fallback" else attempt,
            "trace": normalize_newlines(last),
        }
        append_jsonl(traces_jsonl, obj)
        trace_cache[bid] = obj
        rows_out.append(obj)
        source_ctr[used] += 1

    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL)
    log(f"Wrote CSV: {out_csv} rows={len(df)}")
    log(f"answer_schema: {dict(schema_ctr)}")
    log(f"answer_key_style: {dict(key_style_ctr)}")
    log(f"trace_source: {dict(source_ctr)}")


if __name__ == "__main__":
    main()

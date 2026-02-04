#!/usr/bin/env python3
"""
Generate a *synthetic* RCA_RULE trace corpus from Phase2 questions to teach
format variants (digit / letter / alphanum option keys).

This does NOT change the RCA rule-engine logic. It:
1) routes Phase2 questions using the existing router (RCA vs RCA_OOD vs GENERAL)
2) for RCA questions, computes the semantic label (C1..C8) using reasoned_predict.py
3) parses the question's option list and maps the semantic label -> option key
4) emits deterministic traces in the same block structure:
   TriggeredRule / Evidence / Rejections / Decision / FinalAnswer: \\boxed{<key>}

Output CSV columns:
  ID,fixed_label,answer_value,answer_schema,triggered_rule,trace_source,attempts,trace
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Allow running as `python3 tools/generate_phase2_rule_synth_corpus.py ...`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rca.router import load_reasoned_predict, route_questions_phase2
from rca.util import log


_OPT_LINE_RE = re.compile(r"^\s*([A-Za-z]{1,3}\d{1,2}|C[1-8]|[A-I]|[1-9])\s*[:\)]\s*(.+?)\s*$")


@dataclass(frozen=True)
class OptionMap:
    keys: List[str]
    key_to_text: Dict[str, str]


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
    # Keep heuristic aligned with rca/rca_rule/generate.py
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
    """
    Returns (answer_value, schema) where schema describes the mapping method.
    """
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


def build_triggered_rule_line(*, fixed_label: str, derived: dict[str, Any], rp: Any) -> str:
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    if fixed_label == "C7":
        v = derived.get("speedMax")
        return f"TriggeredRule: speedMax > {int(speed_thr)} (speedMax={_fmt_num(v,1)}, threshold={int(speed_thr)})"
    if fixed_label == "C8":
        v = derived.get("avgRB_low")
        return f"TriggeredRule: avgRB_low < {int(rb_thr)} (avgRB_low={_fmt_num(v,1)}, threshold={int(rb_thr)})"
    if fixed_label == "C2":
        v = derived.get("maxDist_m")
        return f"TriggeredRule: maxDist_m > {int(dist_thr)} (maxDist_m={_fmt_num(v,1)}, threshold={int(dist_thr)})"
    if fixed_label == "C6":
        v = derived.get("mod30Frac")
        return f"TriggeredRule: mod30Frac >= {mod30_thr} (mod30Frac={_fmt_num(v,2)}, threshold={mod30_thr})"
    if fixed_label == "C5":
        v = derived.get("handovers")
        return f"TriggeredRule: handovers >= {ho_thr} (handovers={v}, threshold={ho_thr})"
    if fixed_label == "C4":
        v = derived.get("diffGnbCloseFrac")
        return f"TriggeredRule: diffGnbCloseFrac == {diff_thr} (diffGnbCloseFrac={_fmt_num(v,2)}, threshold={diff_thr})"
    if fixed_label == "C3":
        v = derived.get("sameGnbCloseFrac")
        return f"TriggeredRule: sameGnbCloseFrac == 1.0 (sameGnbCloseFrac={_fmt_num(v,2)}, threshold=1.0)"
    if fixed_label == "C1":
        v = derived.get("beyondBeamRows")
        return f"TriggeredRule: beyondBeamRows > 0 (beyondBeamRows={v}, threshold=0)"
    return "TriggeredRule: missing (threshold=missing)"


def build_evidence_bullets(*, fixed_label: str, derived: dict[str, Any]) -> List[str]:
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

    out: List[str] = []
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        v = derived.get(k)
        if k in {"speedMax", "avgRB_low", "maxDist_m"}:
            out.append(f"- {k}={_fmt_num(v,1)}")
        elif k in {"handovers", "beyondBeamRows"}:
            out.append(f"- {k}={v if v is not None else 'NA'}")
        elif k in {"mod30Frac", "sameGnbCloseFrac", "diffGnbCloseFrac"}:
            out.append(f"- {k}={_fmt_num(v,2)}")
        else:
            out.append(f"- {k}={v if v is not None else 'NA'}")
    return out


def build_rejection_bullets(*, fixed_label: str, triggered_rule_line: str, derived: dict[str, Any], rp: Any) -> List[str]:
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    specs = [
        ("C7", "speedMax", _fmt_num(derived.get("speedMax"), 1), f"speedMax>{int(speed_thr)}", str(int(speed_thr))),
        ("C8", "avgRB_low", _fmt_num(derived.get("avgRB_low"), 1), f"avgRB_low<{int(rb_thr)}", str(int(rb_thr))),
        ("C2", "maxDist_m", _fmt_num(derived.get("maxDist_m"), 1), f"maxDist_m>{int(dist_thr)}", str(int(dist_thr))),
        ("C6", "mod30Frac", _fmt_num(derived.get("mod30Frac"), 2), f"mod30Frac>={mod30_thr}", str(mod30_thr)),
        ("C5", "handovers", str(derived.get("handovers") if derived.get("handovers") is not None else "NA"), f"handovers>={ho_thr}", str(ho_thr)),
        ("C4", "diffGnbCloseFrac", _fmt_num(derived.get("diffGnbCloseFrac"), 2), f"diffGnbCloseFrac=={diff_thr}", str(diff_thr)),
    ]

    out: List[str] = []
    for lbl, metric, mval, cond, thr in specs:
        if lbl == fixed_label:
            continue
        out.append(f"- {lbl} rejected because {metric}={mval} does not meet {cond} (threshold={thr})")
        if len(out) >= 3:
            break
    out.append(f"- Other causes rejected because a different TriggeredRule fired: {triggered_rule_line}")
    return out


def build_trace(*, fixed_label: str, answer_value: str, derived: dict[str, Any], rp: Any, rule_reason: str) -> str:
    trig_line = build_triggered_rule_line(fixed_label=fixed_label, derived=derived, rp=rp)
    ev = build_evidence_bullets(fixed_label=fixed_label, derived=derived)
    rej = build_rejection_bullets(fixed_label=fixed_label, triggered_rule_line=trig_line, derived=derived, rp=rp)
    trig_code = infer_triggered_rule(rule_reason)
    return (
        f"{trig_line}\n\n"
        "Evidence:\n"
        + "\n".join(ev)
        + "\n\nRejections:\n"
        + "\n".join(rej)
        + "\n\nDecision:\n"
        + f"- The TriggeredRule indicates {fixed_label} is the first satisfied condition in the rubric; higher-priority rules were not satisfied (see Rejections). (trigger={trig_code})"
        + "\n\nFinalAnswer: \\boxed{"
        + str(answer_value)
        + "}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Directory used to locate reasoned_predict.py (default: .)")
    ap.add_argument("--phase2", default="phase_2_test.csv", help="Phase2 questions CSV (ID,question)")
    ap.add_argument("--out", default="tools/rule_trace_corpus/traces_phase2_rule_synth.csv")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of base IDs (RCA only)")
    args = ap.parse_args()

    rp = load_reasoned_predict(args.data_dir)
    phase2 = load_questions_csv(args.phase2)
    routes = route_questions_phase2(phase2, rp=rp)

    rca_ids = [bid for bid, meta in routes.items() if meta.get("task_type") == "RCA"]
    rca_ids.sort()
    if args.limit is not None:
        rca_ids = rca_ids[: int(args.limit)]

    total_rca = sum(1 for bid in routes if routes[bid].get("task_type") == "RCA")
    log(f"Phase2 total={len(phase2)} routed_RCA={total_rca} using={len(rca_ids)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    schema_ctr: Counter[str] = Counter()
    key_style_ctr: Counter[str] = Counter()
    unmapped: List[str] = []

    rows: List[dict[str, Any]] = []
    for bid in rca_ids:
        q = phase2[bid]
        metrics, status = rp.compute_metrics(q)
        fixed_label, reason = rp.decide(metrics, status)
        derived = metrics_to_derived_dict(metrics, status)
        triggered_rule = infer_triggered_rule(reason)

        answer_value, schema = map_label_to_option_key(q, fixed_label)
        if answer_value is None:
            answer_value = fixed_label
            unmapped.append(bid)

        schema_ctr[schema] += 1
        if re.fullmatch(r"[1-9]", str(answer_value)):
            key_style_ctr["digit"] += 1
        elif re.fullmatch(r"[A-I]", str(answer_value)):
            key_style_ctr["letter"] += 1
        elif re.fullmatch(r"C[1-8]", str(answer_value)):
            key_style_ctr["C#"] += 1
        elif re.fullmatch(r"[A-Za-z]{1,3}\d{1,2}", str(answer_value)):
            key_style_ctr["alphanum"] += 1
        else:
            key_style_ctr["other"] += 1

        trace = build_trace(
            fixed_label=fixed_label,
            answer_value=str(answer_value),
            derived=derived,
            rp=rp,
            rule_reason=reason,
        )

        rows.append(
            {
                "ID": bid,
                "fixed_label": fixed_label,
                "answer_value": str(answer_value),
                "answer_schema": schema,
                "triggered_rule": triggered_rule,
                "trace_source": "phase2_synth",
                "attempts": 0,
                "trace": trace,
            }
        )

    df = pd.DataFrame(rows)
    # Quote all fields to safely preserve newlines in 'trace'.
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_ALL)

    log(f"Wrote: {args.out} rows={len(df)}")
    log(f"answer_schema: {dict(schema_ctr)}")
    log(f"answer_key_style: {dict(key_style_ctr)}")
    if unmapped:
        log(f"unmapped (showing up to 20): {unmapped[:20]}")


if __name__ == "__main__":
    main()

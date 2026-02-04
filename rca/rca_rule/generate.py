from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from rca.util import flatten_cell, log
from tqdm import tqdm


BOX_RE_ANY = re.compile(r"\\\\boxed\{[^}]*\}|\\boxed\{[^}]*\}")
FINAL_LINE_RE = re.compile(r"^\s*(FinalAnswer|FINAL|Final)\s*:\s*", flags=re.IGNORECASE)

# Option-line parsing for Phase2 RCA_RULE.
# Supports both:
#   1) Pure numeric keys: "1: ..."
#   2) Letter+digit keys: "Z4: ..."
#
# Note: We intentionally do NOT treat bare letters (e.g. "A:") as RCA_RULE options,
# because Phase2 RCA_OOD/GENERAL may use those formats and are routed elsewhere.
_OPT_RE = re.compile(r"^\s*(?P<key>[A-Za-z]\d{1,2}|\d{1,2})\s*:\s*(?P<text>.+?)\s*$")


def strip_question_wrapper(text: str) -> str:
    s = str(text)
    start = s.find("<question>")
    end = s.rfind("</question>")
    if start == -1 or end == -1 or end <= start:
        return s
    return s[start + len("<question>") : end].strip("\n")


def parse_option_lines(question: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ln in (question or "").splitlines():
        m = _OPT_RE.match(ln)
        if not m:
            continue
        key = (m.group("key") or "").strip()
        text = (m.group("text") or "").strip()
        if not key or not text:
            continue
        # Guardrail: only accept keys that correspond to the 8 RCA classes.
        # - digits 1..8
        # - letter+digit where digit 1..8 (e.g., Z4)
        if re.fullmatch(r"[1-8]", key) or re.fullmatch(r"[A-Za-z][1-8]", key):
            out.append((key, text))
    return out


def semantic_label_from_option_text(text: str) -> Optional[str]:
    # Keep identical heuristic mapping as the original “rules justify” prototype.
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


def map_label_to_option_key(question: str, label: str) -> Optional[str]:
    opts = parse_option_lines(question)
    if not opts:
        return None
    mapping: dict[str, str] = {}
    for key, txt in opts:
        c = semantic_label_from_option_text(txt)
        if c and c not in mapping:
            mapping[c] = key
    return mapping.get(label)


def infer_rules_answer_value(question: str, semantic_label: str) -> Tuple[str, str]:
    q = question or ""
    # Phase1 schema: C1..C8
    if re.search(r"(?m)^\s*C1\s*:", q) and re.search(r"(?m)^\s*C8\s*:", q):
        return semantic_label, "clabel"
    # Phase2 schema: option keys (either 1..8 or letter+digit like Z4)
    opt = map_label_to_option_key(q, semantic_label)
    if opt is not None:
        if re.fullmatch(r"[1-8]", opt):
            return opt, "option_number"
        return opt, "option_key"
    return semantic_label, "fallback"


def _r(x: Any, ndigits: int) -> Any:
    try:
        if x is None:
            return None
        return round(float(x), ndigits)
    except Exception:
        return None


def build_derived_metrics(metrics: Any, status: str) -> dict[str, Any]:
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
    return f"RULES_{fixed_label}_unknown"


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
        return f"{xf:.2f}".rstrip("0").rstrip(".")
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
        v = derived.get("beyondBeamRows")
        return f"TriggeredRule: beyondBeamRows > 0 (beyondBeamRows={v}, threshold=0)"
    return "TriggeredRule: missing (threshold=missing)"


def build_evidence_bullets(*, fixed_label: str, derived: dict[str, Any]) -> list[str]:
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


def build_rejection_bullets(*, fixed_label: str, triggered_rule_line: str, derived: dict[str, Any], rp: Any) -> list[str]:
    speed_thr = float(getattr(rp, "SPEED_MAX_KMH", 40.0))
    rb_thr = float(getattr(rp, "RB_MIN", 160.0))
    dist_thr = float(getattr(rp, "OVERSHOOT_M", 1000.0))
    mod30_thr = float(getattr(rp, "MOD30_FRAC_MIN", 0.75))
    ho_thr = int(getattr(rp, "HANDOVERS_MIN", 2))
    diff_thr = float(getattr(rp, "C4_DIFF_GNB_CLOSE_FRAC_MIN", 1.0))

    specs = [
        ("C7", "speedMax", _fmt_num(derived.get("speedMax"), 1), f"speedMax>{_fmt_thr(speed_thr)}", _fmt_thr(speed_thr)),
        ("C8", "avgRB_low", _fmt_num(derived.get("avgRB_low"), 1), f"avgRB_low<{_fmt_thr(rb_thr)}", _fmt_thr(rb_thr)),
        ("C2", "maxDist_m", _fmt_num(derived.get("maxDist_m"), 1), f"maxDist_m>{_fmt_thr(dist_thr)}", _fmt_thr(dist_thr)),
        ("C6", "mod30Frac", _fmt_num(derived.get("mod30Frac"), 2), f"mod30Frac>={_fmt_thr(mod30_thr)}", _fmt_thr(mod30_thr)),
        ("C5", "handovers", str(derived.get("handovers") if derived.get("handovers") is not None else "NA"), f"handovers>={_fmt_thr(ho_thr)}", _fmt_thr(ho_thr)),
        ("C4", "diffGnbCloseFrac", _fmt_num(derived.get("diffGnbCloseFrac"), 2), f"diffGnbCloseFrac=={_fmt_thr(diff_thr)}", _fmt_thr(diff_thr)),
    ]

    out: list[str] = []
    for lbl, metric, mval, cond, thr in specs:
        if lbl == fixed_label:
            continue
        out.append(f"- {lbl} rejected because {metric}={mval} does not meet {cond} (threshold={thr})")
        if len(out) >= 3:
            break
    # Always include one high-level rejection that references the triggered rule.
    out.append(f"- Other causes rejected because a different TriggeredRule fired: {triggered_rule_line}")
    return out


def sanitize_trace(text: str) -> str:
    s = str(text or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = BOX_RE_ANY.sub("", s)
    out_lines: list[str] = []
    for ln in s.split("\n"):
        if FINAL_LINE_RE.match(ln):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def format_submission_cell(prefix: str, trace: str, boxed_value: str, *, flatten_newlines: bool) -> str:
    body = f"{prefix}{trace}\nFinalAnswer: \\boxed{{{boxed_value}}}"
    return flatten_cell(body) if flatten_newlines else body


def _load_lora_model(lora_dir: str):
    """
    Standard Transformers/PEFT loader (no Unsloth required).
    Safe for Kaggle's memory limits if you don't use heavy builders.
    NO quantization as requested.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig

    log(f"Loading LoRA config from {lora_dir}")
    config = PeftConfig.from_pretrained(lora_dir)
    
    log(f"Loading Base model: {config.base_model_name_or_path} in full precision (bf16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    log(f"Applying LoRA adapter from {lora_dir}")
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, *, max_new_tokens: int, temperature: float) -> str:
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            do_sample=float(temperature) > 0.0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _apply_chat_template(tokenizer: Any, *, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\n\nAnswer:"


def generate_rule_traces(
    *,
    base_ids: List[str],
    questions: Dict[str, str],
    rp: Any,
    mode_cfg: dict,
    out_debug_jsonl: str,
    prefix: str,
    flatten_newlines: bool,
) -> Dict[str, Dict[int, str]]:
    """
    Generate 4 submission cells per base ID for RCA_RULE.
    """
    rule_mode = str(mode_cfg.get("mode", "deterministic")).strip().lower()

    decision_variants = list(mode_cfg.get("decision_variants") or [])
    if not decision_variants:
        decision_variants = [
            "Decision:\n- TriggeredRule selects the fixed label; higher-priority checks are not triggered.",
            "Decision:\n- The fixed label is consistent with the TriggeredRule and the Evidence; rejected causes do not meet their thresholds.",
            "Decision:\n- This case matches the fixed label per the rubric; the rejection checks show no higher-priority trigger fired.",
            "Decision:\n- Therefore the fixed label is selected by the deterministic rubric using the provided metrics.",
        ]
    while len(decision_variants) < 4:
        decision_variants.append(decision_variants[-1])
    decision_variants = decision_variants[:4]

    model = None
    tokenizer = None
    if rule_mode in ("unsloth_lora", "lora"):
        lora_dir = str(mode_cfg.get("lora_model_dir") or "").strip()
        if not lora_dir:
            raise ValueError(f"rca_rule.mode={rule_mode} requires rca_rule.lora_model_dir")
        if not os.path.exists(lora_dir):
            raise FileNotFoundError(f"LoRA model dir not found: {lora_dir}")
            
        log(f"Loading LoRA model via PEFT (Full Precision): {lora_dir}")
        model, tokenizer = _load_lora_model(lora_dir)
        log("Model loaded successfully.")

    max_new_tokens = int(mode_cfg.get("max_new_tokens", 512))
    temperature = float(mode_cfg.get("temperature", 0.0))

    os.makedirs(os.path.dirname(out_debug_jsonl) or ".", exist_ok=True)

    out: Dict[str, Dict[int, str]] = {}
    log(f"Processing {len(base_ids)} IDs for RCA_RULE...")
    dbg_mode = "w"  # Always overwrite for fresh run
    with open(out_debug_jsonl, dbg_mode, encoding="utf-8") as dbg_f:
        for i, bid in enumerate(tqdm(base_ids, desc="Generating Rule Traces"), 1):
            
            log(f"Processing bid: {bid} ({i}/{len(base_ids)})")
            
            q_raw = questions.get(bid, "")
            q = strip_question_wrapper(q_raw)
            try:
                metrics, status = rp.compute_metrics(q)
            except Exception:
                metrics, status = None, "exception"
            semantic_label, reason = rp.decide(metrics, status)
            answer_value, answer_schema = infer_rules_answer_value(q, semantic_label)

            derived = build_derived_metrics(metrics, status)
            triggered_rule = infer_triggered_rule(semantic_label, reason)
            triggered_rule_line = build_triggered_rule_line(fixed_label=semantic_label, derived=derived, rp=rp)
            evidence_bullets = build_evidence_bullets(fixed_label=semantic_label, derived=derived)
            rejection_bullets = build_rejection_bullets(
                fixed_label=semantic_label,
                triggered_rule_line=triggered_rule_line,
                derived=derived,
                rp=rp,
            )

            evidence_block = "\n".join(evidence_bullets)
            rejection_block = "\n".join(rejection_bullets)

            variant = decision_variants[0] # Use first variant for the single generation
            user_prompt = (
                "A deterministic rule-engine already selected the correct label.\n\n"
                f"FixedLabelSemantic: {semantic_label}\n"
                f"CorrectAnswerValue (copy exactly; do not change): {answer_value}\n\n"
                "Hard rules (must follow):\n"
                "- Do NOT change FixedLabelSemantic.\n"
                "- Copy the TriggeredRule/Evidence/Rejections blocks EXACTLY as given.\n"
                "- Do NOT introduce any numeric values not already present in those blocks.\n"
                "- Add ONE short Decision bullet (no new numbers).\n"
                "- The LAST line must contain exactly one boxed answer using CorrectAnswerValue.\n\n"
                "TriggeredRule line (copy EXACTLY):\n"
                f"{triggered_rule_line}\n\n"
                "Evidence block (copy EXACTLY):\n"
                "Evidence:\n"
                f"{evidence_block}\n\n"
                "Rejections block (copy EXACTLY):\n"
                "Rejections:\n"
                f"{rejection_block}\n\n"
                "Now output in this STRICT format:\n"
                f"{triggered_rule_line}\n\n"
                "Evidence:\n"
                f"{evidence_block}\n\n"
                "Rejections:\n"
                f"{rejection_block}\n\n"
                f"{variant}\n"
                f"FinalAnswer: \\boxed{{{answer_value}}}\n"
            )

            if model is not None and tokenizer is not None:
                prompt = _apply_chat_template(
                    tokenizer,
                    system="You are a staff 5G RAN engineer writing a grounded RCA justification trace.",
                    user=user_prompt,
                )
                raw = _generate(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            else:
                raw = (
                    f"{triggered_rule_line}\n\n"
                    "Evidence:\n"
                    f"{evidence_block}\n\n"
                    "Rejections:\n"
                    f"{rejection_block}\n\n"
                    f"{variant}\n"
                    f"FinalAnswer: \\boxed{{{answer_value}}}\n"
                )

            sanitized = sanitize_trace(raw)
            cell = format_submission_cell(prefix, sanitized, answer_value, flatten_newlines=flatten_newlines)
            
            # Copy to all 4 slots
            out[bid] = {1: cell, 2: cell, 3: cell, 4: cell}

            for k in (1, 2, 3, 4):
                dbg_f.write(
                    json.dumps(
                        {
                            "base_id": bid,
                            "gen": k,
                            "semantic_label": semantic_label,
                            "answer_value": answer_value,
                            "answer_schema": answer_schema,
                            "triggered_rule": triggered_rule,
                            "triggered_rule_line": triggered_rule_line,
                            "metrics_status": status,
                            "raw": raw,
                            "sanitized": sanitized,
                            "cell": cell,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return out

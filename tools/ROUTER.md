# Router + Output Schemas

Goal: use deterministic rules when the input is *in‑distribution* (the RCA table format), and fall back to an LLM when it’s *out‑of‑distribution* (general questions or different fault formats).

This avoids LLM numeric/logic failures (flipped inequalities, ignored rule order) while still handling non‑RCA questions.

## Router Criteria (fast → accurate)

For each `question` blob:

1) **Cheap string gate (fast)**  
   Treat the blob as “in‑distribution RCA” if it contains:
   - a pipe delimiter `|`
   - “User plane drive test data …”
   - “… parameters data …”
   - “throughput”

   If this fails:
   - If it looks like an RCA classification task (e.g., references `C1..C8`, or “drive test + engineering parameters + root cause” with options like `A:`…`I:`) → route to **LLM / RCA_OOD**
   - Else → route to **LLM / GENERAL**

2) **Deterministic metric extraction (accurate)**
   Call `tools/reasoned_predict.py::compute_metrics(raw_question)`:
   - If `status == "ok"` and metrics are non‑null → route to **RULES / RCA**.
   - Else → route to **LLM / RCA_OOD** (still RCA‑ish, but the extractor couldn’t compute required metrics/columns).

Implementation reference: `tools/router.py`.

## Output Schemas (stable parsing)

To keep parsing consistent across tracks/models, enforce these invariants:

### Invariant (all routes)
- Exactly one line contains: `FinalAnswer: \boxed{...}`  
  (Your evaluator/regex should extract **only** the content inside `\boxed{}`.)

### Schema A: RULES / RCA (in‑distribution)
Use when router selects `route=RULES`.

Required:
- `DerivedMetrics: { ...json... }` (authoritative metrics from `compute_metrics`)
- `DecisionTrace:` numbered steps (can be short)
- `FinalAnswer: \boxed{C#}` where `C# ∈ {C1..C8}`

Example (minimal):
```
DerivedMetrics: {...}
DecisionTrace:
1) routed=RULES (metricsStatus=ok)
2) ruleTriggered=R5(handovers>=2)
FinalAnswer: \boxed{C5}
```

### Schema B: LLM / GENERAL or LLM / RCA_OOD (out‑of‑distribution)
Use when router selects `route=LLM`.

Required:
- `FinalAnswer: \boxed{...}` (single boxed answer token)

Recommended:
- `DecisionTrace:` explain what was missing / how it reasoned
- If it’s still an RCA task (RCA_OOD), keep `FinalAnswer` in `C1..C8` if possible.

Example (GENERAL):
```
DecisionTrace:
1) inputFormat=GENERAL (no RCA tables)
FinalAnswer: \boxed{42}
```

Example (RCA_OOD):
```
DecisionTrace:
1) inputFormat=RCA_OOD (missing engineering table columns)
2) inferred most plausible cause from available evidence
FinalAnswer: \boxed{C2}
```

## Recommended Pipeline

1) Router decides `RULES` vs `LLM`.
2) If `RULES`, compute label deterministically (never let the LLM choose it).
3) If you need traces, call the LLM **only to write the trace**, with the label fixed.
4) If `LLM`, allow normal generation but enforce the `FinalAnswer: \boxed{...}` invariant.

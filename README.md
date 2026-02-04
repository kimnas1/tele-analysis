# 5G RCA Challenge Pipeline

This repo contains:
- A **deterministic RULE engine** (near-perfect on in-distribution RCA) that outputs semantic labels `C1..C8`.
- **LLM pipelines** for `RCA_OOD` and `GENERAL` questions (Phase 2).
- A clean, reproducible **single-command** runner that outputs **SampleSubmission-format** CSVs per model, then merges them.

## Quickstart: inference (recommended)

Run end-to-end inference (Phase 1 + Phase 2; RCA_RULE + RCA_OOD + GENERAL) and produce a SampleSubmission-format CSV.

```bash
python3 scripts/run_model.py <model_tag> <data_dir> <out_dir> [lora_model_dir]
```

Example (Kaggle):
```bash
python3 scripts/run_model.py qwen7b /kaggle/input/rca-5g /kaggle/working/out /kaggle/input/qwen7b-lora-rule-traces
```

Output:
- `<out_dir>/<model_tag>/submission_<model_tag>.csv`
  - Only that model’s submission column is filled; all other columns remain `placeholder`.

### Merge per-model submissions

If you run multiple models (one per command), merge them into one final `submission.csv`:
```bash
python3 scripts/merge_submissions.py <data_dir> <out_dir> <merged_csv>
```

Example (Kaggle):
```bash
python3 scripts/merge_submissions.py /kaggle/input/rca-5g /kaggle/working/out /kaggle/working/submission.csv
```

## Install (local dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU notes:
- Install **PyTorch** separately for your CUDA version.
- LoRA inference uses **Transformers + PEFT**.
- (Optional) LoRA training can use **Transformers/TRL** or **Unsloth** (see scripts in `tools/`).

## Data layout

For the runner scripts below, pass `data_dir` that contains (filenames are configurable in `configs/pipeline.json`):
- `SampleSubmission.csv`
- `phase_1_test.csv`
- `phase_2_test.csv`
- (optional) `reasoned_predict.py` (if absent, the repo’s `tools/reasoned_predict.py` is used)

## Model artifacts

LoRA adapters are expected as directories:
- Any PEFT adapter directory (must contain `adapter_config.json` + `adapter_model.safetensors`)

Note: `models/` is intentionally **gitignored** (large weights). Download/place adapters locally (or host them on HuggingFace / GitHub release assets) and point `configs/pipeline.json` at the local paths.

If you only have zips, unzip them into `models/` and update `configs/pipeline.json` accordingly.

## Optional: run only a slice (debug)

`scripts/run_model.py` supports an optional `run_only` selector (or env var `RUN_ONLY`) for faster debugging:
- `RCA_RULE_PHASE2`: only Phase 2 routed RCA_RULE questions
- `RCA_RULE_ONLY`: only RCA_RULE (Phase 1 + Phase 2), everything else placeholder

```bash
python3 scripts/run_model.py <model_tag> <data_dir> <out_dir> [lora_model_dir] RCA_RULE_ONLY
```

## Reproduce from scratch (optional)

### 1) Generate RULE trace corpus (train set)

This builds `EvidencePack` from the deterministic rule engine and uses an LLM only to **justify** the fixed label.

Edit the config block at the top of:
- `tools/generate_rule_trace_corpus.py`

Then run:
```bash
python3 tools/generate_rule_trace_corpus.py
```

Outputs go to:
- `tools/rule_trace_corpus/`

### 2) Fine-tune LoRA on rule traces

Edit the config block at the top of:
- `tools/finetune_qwen_lora_rule_traces.py` (Transformers/TRL)
- or `tools/finetune_qwen_lora_rule_traces_unsloth.py` (Unsloth)

Then run:
```bash
python3 tools/finetune_qwen_lora_rule_traces.py
```

The resulting adapter directory can be used by the runner via `configs/pipeline.json`.

## Cerebras (for RCA_OOD / GENERAL)

If you run `RCA_OOD` / `GENERAL` using Cerebras, export your API key:
```bash
export CEREBRAS_API_KEY="..."
```

The runner uses the same vote/trace pipeline as:
- `tools/generate_submission_rcaood_vote3_traces.py`

## Config

Main config:
- `configs/pipeline.json`

You typically change:
- `models.<model_tag>.submission_column`
- `models.<model_tag>.rca_rule.*` (LoRA path / generation params)
- `rca_ood_vote3.*` (RCA_OOD/GENERAL reasoning strategy settings)

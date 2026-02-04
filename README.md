# 5G RCA Challenge Pipeline

This repo contains:
- A **deterministic RULE engine** (near-perfect on in-distribution RCA) that outputs semantic labels `C1..C8`.
- **LLM pipelines** for `RCA_OOD` and `GENERAL` questions (Phase 2).
- A clean, reproducible **single-command** runner that outputs **SampleSubmission-format** CSVs per model, then merges them.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU notes:
- Install **PyTorch** separately for your CUDA version.
- LoRA train/infer in this repo uses **Unsloth** (`import unsloth`). Install it per Unsloth’s official instructions for your platform.

## Data layout

For the runner scripts below, pass `data_dir` that contains (filenames are configurable in `configs/pipeline.json`):
- `SampleSubmission.csv`
- `phase_1_test.csv`
- `phase_2_test.csv`
- (optional) `reasoned_predict.py` (if absent, the repo’s `tools/reasoned_predict.py` is used)

## Model artifacts

LoRA adapters are expected as directories:
- `models/qwen7b_lora_rule_traces/`
- `models/qwen1.5b_lora_rule_traces/`

Note: `models/` is intentionally **gitignored** (large weights). Download/place adapters locally (or host them on HuggingFace / GitHub release assets) and point `configs/pipeline.json` at the local paths.

If you only have zips, unzip them into `models/` and update `configs/pipeline.json` accordingly.

## One-command generation (per model)

Run one model and get a per-model submission file (only that model column is filled; all others remain `placeholder`):

```bash
python3 scripts/run_model.py <model_tag> <data_dir> <out_dir>
```

Examples:
```bash
python3 scripts/run_model.py qwen7b   /kaggle/input/rca-5g  /kaggle/working/out
python3 scripts/run_model.py qwen1.5b /kaggle/input/rca-5g  /kaggle/working/out
python3 scripts/run_model.py qwen32b  /kaggle/input/rca-5g  /kaggle/working/out
```

Outputs:
- `out_dir/<model_tag>/submission_<model_tag>.csv`
- Routing/debug files in `out_dir/<model_tag>/` (Phase2 split into RULE/RCA_OOD/GENERAL).

### Merge per-model submissions

```bash
python3 scripts/merge_submissions.py <data_dir> <out_dir> <merged_csv>
```

Example:
```bash
python3 scripts/merge_submissions.py /kaggle/input/rca-5g /kaggle/working/out /kaggle/working/submission.csv
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

### 2) Fine-tune LoRA on rule traces (Unsloth)

Edit the config block at the top of:
- `tools/finetune_qwen_lora_rule_traces_unsloth.py`

Then run:
```bash
python3 tools/finetune_qwen_lora_rule_traces_unsloth.py
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

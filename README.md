# PAC-Zero MLX Validation

This repository contains a compact PAC-Zero / ZPL validation package for Apple MLX using `mlx-community/SmolLM-135M-4bit`.

It is intended as a reproducible software artifact. It preserves the validation scripts, selected result JSON files, logs, and aggregate report for a smoke-scale MLX adaptation. It does **not** claim to reproduce the full paper-scale OPT utility results.

## What is included

- PAC-Zero / ZPL transcript-audit logic with `M = 126` subsets.
- Balanced candidate subset construction with each example appearing in `M / 2 = 63` subsets.
- Per-sample two-point zeroth-order finite differences.
- Rank-8 / alpha-16 LoRA adapters.
- `q_proj + v_proj` adapters across SmolLM layers.
- SST-2 and SQuAD validation paths.
- Non-private zeroth-order utility controls.
- A negative-control audit that deliberately makes the release depend on `S*` and checks that the audit catches it.
- Preserved benchmark logs and result JSON files from the release snapshot.

## Main preserved outputs

Aggregate JSON:

```text
benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_aggregate_results.json
```

Human-readable aggregate report:

```text
benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_report.md
```

Negative-control result:

```text
benchmark-results/paczero-smollm-validation-aggregate/zpl_negative_control_results.json
```

Task-level validation outputs:

```text
benchmark-results/paczero-smollm-validation/smollm-135m-4bit-sst2/smollm_validation_results.json
benchmark-results/paczero-smollm-validation/smollm-135m-4bit-squad/smollm_validation_results.json
```

Non-private utility-control outputs:

```text
benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-sst2/nonprivate_utility_control_results.json
benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-squad/nonprivate_utility_control_results.json
```

## Requirements

For the full MLX run:

- macOS on Apple Silicon, or a GitHub-hosted macOS runner with MLX support.
- Python 3.11.
- Git.
- Network access to download model and dataset dependencies.

Python packages used by the GitHub Actions workflow:

```bash
python -m pip install --upgrade pip
python -m pip install --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
```

For lightweight inspection of preserved outputs, only Python 3.11 is needed.

## Run on GitHub Actions

The main workflow is:

```text
.github/workflows/paczero-smollm-validation.yml
```

It can be run from the GitHub web UI:

1. Push this repository to GitHub with Actions enabled.
2. Open the repository's **Actions** tab.
3. Select **PACZero SmolLM validation workflow**.
4. Choose **Run workflow**.
5. Use the default inputs for the release-scale smoke run, or adjust them if needed.

Default workflow inputs:

```text
steps = 30
train_examples = 8
dev_examples = 8
eval_examples = 32
layers = all
```

The workflow runs both validation tasks:

```text
sst2
squad
```

and both non-private utility-control tasks:

```text
sst2
squad
```

The workflow writes outputs under:

```text
benchmark-results/
benchmark-logs/
```

The workflow also commits generated result files back to the repository. Commit messages include skip markers such as `[skip paczero-smollm-validation]` to avoid recursively triggering the same validation workflow.

### Notes on GitHub workflow permissions

If the workflow itself is added, removed, or modified, GitHub may require a token or account with workflow-write permission. Ordinary result-file commits should only need repository contents write permission.

If you are reconstructing this repository from a Zenodo archive, restore `.github/workflows/paczero-smollm-validation.yml` manually through GitHub or with a token that has workflow permission.

## Run locally on Apple Silicon

Create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
```

Sanity-check the scripts:

```bash
python -m py_compile scripts/*.py
```

### Rebuild the aggregate report from included results

This is the fastest local reproducibility check because it uses the preserved JSON files already included in the repository:

```bash
python scripts/paczero_smollm_validation_aggregate.py
```

Expected outputs:

```text
benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_aggregate_results.json
benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_report.md
```

### Run the ZPL negative control locally

```bash
python scripts/paczero_zpl_negative_control.py
```

This writes or updates:

```text
benchmark-results/paczero-smollm-validation-aggregate/zpl_negative_control_results.json
```

### Run a local SmolLM validation task

SST-2 example:

```bash
mkdir -p benchmark-results/paczero-smollm-validation/smollm-135m-4bit-sst2 \
         benchmark-results/paczero-smollm-validation-adapters/smollm-135m-4bit-sst2

python scripts/paczero_mlxlm_faithful_adaptation.py \
  --model mlx-community/SmolLM-135M-4bit \
  --slug smollm-135m-4bit \
  --task sst2 \
  --projections q_proj,v_proj \
  --layers all \
  --rank 8 \
  --alpha 16.0 \
  --seed 20260615 \
  --steps 30 \
  --train-examples 8 \
  --dev-examples 8 \
  --eval-examples 32 \
  --num-subsets 126 \
  --mu 0.05 \
  --lr 0.05 \
  --clip 25.0 \
  --eval-every 5 \
  --json-out benchmark-results/paczero-smollm-validation/smollm-135m-4bit-sst2/smollm_validation_results.json \
  --adapter-out benchmark-results/paczero-smollm-validation-adapters/smollm-135m-4bit-sst2/all_layers_qv_lora_rank8_alpha16.npz
```

SQuAD example:

```bash
mkdir -p benchmark-results/paczero-smollm-validation/smollm-135m-4bit-squad \
         benchmark-results/paczero-smollm-validation-adapters/smollm-135m-4bit-squad

python scripts/paczero_mlxlm_faithful_adaptation.py \
  --model mlx-community/SmolLM-135M-4bit \
  --slug smollm-135m-4bit \
  --task squad \
  --projections q_proj,v_proj \
  --layers all \
  --rank 8 \
  --alpha 16.0 \
  --seed 20260616 \
  --steps 30 \
  --train-examples 8 \
  --dev-examples 8 \
  --eval-examples 32 \
  --num-subsets 126 \
  --mu 0.05 \
  --lr 0.05 \
  --clip 25.0 \
  --eval-every 5 \
  --json-out benchmark-results/paczero-smollm-validation/smollm-135m-4bit-squad/smollm_validation_results.json \
  --adapter-out benchmark-results/paczero-smollm-validation-adapters/smollm-135m-4bit-squad/all_layers_qv_lora_rank8_alpha16.npz
```

### Run a local non-private utility control

SST-2 example:

```bash
mkdir -p benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-sst2 \
         benchmark-results/paczero-smollm-utility-control-adapters/smollm-135m-4bit-sst2

python scripts/paczero_smollm_nonprivate_utility_control.py \
  --model mlx-community/SmolLM-135M-4bit \
  --slug smollm-135m-4bit \
  --task sst2 \
  --projections q_proj,v_proj \
  --layers all \
  --rank 8 \
  --alpha 16.0 \
  --seed 20260618 \
  --steps 30 \
  --train-examples 8 \
  --dev-examples 8 \
  --eval-examples 32 \
  --mu 0.05 \
  --lr 0.05 \
  --eval-every 5 \
  --json-out benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-sst2/nonprivate_utility_control_results.json \
  --adapter-out benchmark-results/paczero-smollm-utility-control-adapters/smollm-135m-4bit-sst2/nonprivate_all_layers_qv_lora_rank8_alpha16.npz
```

SQuAD is the same command with:

```text
--task squad
--seed 20260619
--json-out benchmark-results/paczero-smollm-utility-control/smollm-135m-4bit-squad/nonprivate_utility_control_results.json
--adapter-out benchmark-results/paczero-smollm-utility-control-adapters/smollm-135m-4bit-squad/nonprivate_all_layers_qv_lora_rank8_alpha16.npz
```

After running task-level validation and utility-control jobs, regenerate the aggregate report:

```bash
python scripts/paczero_smollm_validation_aggregate.py
```

## Expected runtime and storage notes

The default configuration is intentionally small:

```text
train/dev/eval = 8/8/32
steps = 30
M = 126
LoRA = rank 8 / alpha 16
projections = q_proj + v_proj
layers = all
```

It is meant for reproducibility and mechanism validation, not for large-scale utility benchmarking. Adapter `.npz` files may be generated locally or by Actions, but they can be omitted from compact archival uploads if preserved JSON/log evidence is sufficient.

## Scope and limitations

This package does not claim:

- paper-scale OPT-1.3B / OPT-6.7B reproduction;
- paper-scale `1000/500/1000` data or 1000 ZPL steps;
- generated SQuAD EM/F1;
- utility parity with the PAC-Zero paper;
- differential privacy.

The MLX demo uses normalized zeroth-order directions and fixed `mu/lr` for fast, stable execution. It preserves and audits the PAC-Zero / ZPL release-mechanism structure at smoke scale, but it is not a byte-for-byte optimizer reproduction of a reference trainer.

## Suggested citation metadata

For archival releases, use Zenodo or another DOI-backed archive. A modest initial version such as `v0.0.1` is appropriate for a first reproducibility snapshot.

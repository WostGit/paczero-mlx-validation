#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
PAC-Zero MLX validation helper

Usage:
  ./run.sh quick
  ./run.sh aggregate
  ./run.sh negative-control
  ./run.sh install
  ./run.sh local-sst2
  ./run.sh local-squad
  ./run.sh local-control-sst2
  ./run.sh local-control-squad
  ./run.sh local-all

Recommended first check:
  ./run.sh quick

Modes:
  quick                Compile scripts, run the negative control, rebuild the aggregate report from included JSON results.
  aggregate            Rebuild the aggregate report from included JSON results only.
  negative-control     Run the ZPL audit negative control only.
  install              Install the Python packages used by the workflow into the active environment.
  local-sst2           Run the PAC-Zero/ZPL SmolLM validation task for SST-2.
  local-squad          Run the PAC-Zero/ZPL SmolLM validation task for SQuAD.
  local-control-sst2   Run the non-private utility control for SST-2.
  local-control-squad  Run the non-private utility control for SQuAD.
  local-all            Run both validation tasks, both utility controls, the negative control, then aggregate.

Notes:
  - Full local validation requires macOS on Apple Silicon with MLX support.
  - The quick and aggregate modes are suitable for inspecting the archived result package.
  - You can override defaults with environment variables, for example:
      STEPS=10 TRAIN_EXAMPLES=4 DEV_EXAMPLES=4 EVAL_EXAMPLES=16 ./run.sh local-sst2
USAGE
}

MODE="${1:-quick}"
MODEL="${MODEL:-mlx-community/SmolLM-135M-4bit}"
SLUG="${SLUG:-smollm-135m-4bit}"
STEPS="${STEPS:-30}"
TRAIN_EXAMPLES="${TRAIN_EXAMPLES:-8}"
DEV_EXAMPLES="${DEV_EXAMPLES:-8}"
EVAL_EXAMPLES="${EVAL_EXAMPLES:-32}"
LAYERS="${LAYERS:-all}"
PROJECTIONS="${PROJECTIONS:-q_proj,v_proj}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16.0}"
MU="${MU:-0.05}"
LR="${LR:-0.05}"
CLIP="${CLIP:-25.0}"
EVAL_EVERY="${EVAL_EVERY:-5}"
NUM_SUBSETS="${NUM_SUBSETS:-126}"

ensure_repo_root() {
  if [ ! -d scripts ]; then
    echo "run.sh must be executed from the repository root." >&2
    exit 1
  fi
}

install_deps() {
  python -m pip install --upgrade pip
  python -m pip install --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
}

compile_scripts() {
  python -m py_compile scripts/*.py
}

run_negative_control() {
  python scripts/paczero_zpl_negative_control.py
}

run_aggregate() {
  python scripts/paczero_smollm_validation_aggregate.py
}

run_validation_task() {
  task="$1"
  seed="$2"
  out_dir="benchmark-results/paczero-smollm-validation/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-validation-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  python scripts/paczero_mlxlm_faithful_adaptation.py \
    --model "$MODEL" \
    --slug "$SLUG" \
    --task "$task" \
    --projections "$PROJECTIONS" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --steps "$STEPS" \
    --train-examples "$TRAIN_EXAMPLES" \
    --dev-examples "$DEV_EXAMPLES" \
    --eval-examples "$EVAL_EXAMPLES" \
    --num-subsets "$NUM_SUBSETS" \
    --mu "$MU" \
    --lr "$LR" \
    --clip "$CLIP" \
    --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/smollm_validation_results.json" \
    --adapter-out "$adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"
}

run_utility_control_task() {
  task="$1"
  seed="$2"
  out_dir="benchmark-results/paczero-smollm-utility-control/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-utility-control-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  python scripts/paczero_smollm_nonprivate_utility_control.py \
    --model "$MODEL" \
    --slug "$SLUG" \
    --task "$task" \
    --projections "$PROJECTIONS" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --steps "$STEPS" \
    --train-examples "$TRAIN_EXAMPLES" \
    --dev-examples "$DEV_EXAMPLES" \
    --eval-examples "$EVAL_EXAMPLES" \
    --mu "$MU" \
    --lr "$LR" \
    --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/nonprivate_utility_control_results.json" \
    --adapter-out "$adapter_dir/nonprivate_all_layers_qv_lora_rank8_alpha16.npz"
}

ensure_repo_root

case "$MODE" in
  help|-h|--help)
    usage
    ;;
  install)
    install_deps
    ;;
  aggregate)
    run_aggregate
    ;;
  negative-control)
    run_negative_control
    ;;
  quick)
    compile_scripts
    run_negative_control
    run_aggregate
    ;;
  local-sst2)
    run_validation_task sst2 20260615
    ;;
  local-squad)
    run_validation_task squad 20260616
    ;;
  local-control-sst2)
    run_utility_control_task sst2 20260618
    ;;
  local-control-squad)
    run_utility_control_task squad 20260619
    ;;
  local-all)
    run_validation_task sst2 20260615
    run_validation_task squad 20260616
    run_utility_control_task sst2 20260618
    run_utility_control_task squad 20260619
    run_negative_control
    run_aggregate
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac

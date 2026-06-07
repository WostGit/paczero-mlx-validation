#!/usr/bin/env bash
set -euo pipefail

# SST-2 utility-smoke sweep for PAC-Zero MLX validation.
#
# Goal:
#   Find the smallest short run that gives a defensible SST-2 utility signal.
#
# Definition used here:
#   A run is "convincing_smoke" if selected eval accuracy is at least
#   max(0.60, baseline_eval_accuracy + 0.05).
#
# Why this threshold:
#   The current 8/8/32 run sits at chance-level 0.50 accuracy. A tiny change in
#   loss alone is not persuasive. This sweep looks for a clear accuracy lift.

MODEL="${MODEL:-mlx-community/SmolLM-135M-4bit}"
SLUG="${SLUG:-smollm-135m-4bit}"
LAYERS="${LAYERS:-all}"
PROJECTIONS="${PROJECTIONS:-q_proj,v_proj}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16.0}"
MU="${MU:-0.05}"
LR="${LR:-0.05}"
CLIP="${CLIP:-25.0}"
EVAL_EVERY="${EVAL_EVERY:-10}"
NUM_SUBSETS="${NUM_SUBSETS:-126}"
BASE_SEED="${BASE_SEED:-20260630}"
STOP_ON_CONVINCING="${STOP_ON_CONVINCING:-1}"

hr() { printf '\n%s\n' "----------------------------------------------------------------------"; }
log() { printf '[paczero-sweep] %s\n' "$*"; }
fail() { printf '\n[paczero-sweep][error] %s\n' "$*" >&2; exit 1; }

if [ ! -d scripts ]; then
  fail "Run this from the package root; ./scripts was not found."
fi

log "Installing/checking MLX dependencies through run.sh."
bash run.sh install

if [ -x .venv/bin/python ]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  fail "No Python found after install."
fi

log "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

summary_dir="benchmark-results/paczero-sst2-utility-sweep"
mkdir -p "$summary_dir"
summary_jsonl="$summary_dir/sweep_results.jsonl"
summary_txt="$summary_dir/sweep_summary.txt"
: > "$summary_jsonl"
: > "$summary_txt"

# Format: label steps train dev eval
# The first row reproduces the current smoke-scale setting.
configs=(
  "tiny_current 30 8 8 32"
  "small_32 60 32 32 64"
  "candidate_64 100 64 64 128"
  "candidate_128 200 128 128 256"
  "stronger_256 350 256 256 512"
)

cat <<'INTRO' | tee -a "$summary_txt"
PAC-Zero MLX SST-2 utility-smoke sweep

Criterion for a convincing short utility signal:
  selected_eval_accuracy >= max(0.60, baseline_eval_accuracy + 0.05)

Interpretation:
  - PASS means this is the smallest tested run so far that gives a clear SST-2 accuracy lift.
  - If all rows fail, the artifact still supports mechanism validation, but not short-run utility reproduction.
INTRO

found="0"

for row in "${configs[@]}"; do
  read -r label steps train_examples dev_examples eval_examples <<< "$row"
  seed=$((BASE_SEED + steps + train_examples))
  out_dir="$summary_dir/$label"
  adapter_dir="$summary_dir/${label}-adapters"
  json_out="$out_dir/smollm_validation_results.json"
  adapter_out="$adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"
  mkdir -p "$out_dir" "$adapter_dir"

  hr | tee -a "$summary_txt"
  log "Running $label: steps=$steps train/dev/eval=$train_examples/$dev_examples/$eval_examples seed=$seed" | tee -a "$summary_txt"

  start_epoch=$(date +%s)
  "$PYTHON_BIN" scripts/paczero_mlxlm_faithful_adaptation.py \
    --model "$MODEL" \
    --slug "$SLUG" \
    --task sst2 \
    --projections "$PROJECTIONS" \
    --layers "$LAYERS" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --seed "$seed" \
    --steps "$steps" \
    --train-examples "$train_examples" \
    --dev-examples "$dev_examples" \
    --eval-examples "$eval_examples" \
    --num-subsets "$NUM_SUBSETS" \
    --mu "$MU" \
    --lr "$LR" \
    --clip "$CLIP" \
    --eval-every "$EVAL_EVERY" \
    --json-out "$json_out" \
    --adapter-out "$adapter_out"
  end_epoch=$(date +%s)
  wall_seconds=$((end_epoch - start_epoch))

  set +e
  "$PYTHON_BIN" - "$json_out" "$label" "$wall_seconds" "$summary_jsonl" "$summary_txt" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
label = sys.argv[2]
wall_seconds = int(sys.argv[3])
summary_jsonl = Path(sys.argv[4])
summary_txt = Path(sys.argv[5])

data = json.loads(json_path.read_text())
baseline = data.get('baseline') or {}
selected = data.get('selected_best_adapter_eval') or data.get('best_checkpoint') or {}
checks = data.get('checks') or {}
privacy = data.get('privacy_accounting') or {}

def f(x):
    return None if x is None else float(x)

base_eval_acc = f(baseline.get('eval_accuracy'))
sel_eval_acc = f(selected.get('eval_accuracy'))
base_dev_acc = f(baseline.get('dev_accuracy'))
sel_dev_acc = f(selected.get('dev_accuracy'))
base_eval_loss = f(baseline.get('eval_loss'))
sel_eval_loss = f(selected.get('eval_loss'))
base_dev_loss = f(baseline.get('dev_loss'))
sel_dev_loss = f(selected.get('dev_loss'))

threshold = None
convincing = False
if base_eval_acc is not None and sel_eval_acc is not None:
    threshold = max(0.60, base_eval_acc + 0.05)
    convincing = bool(sel_eval_acc >= threshold)

record = {
    'label': label,
    'success': bool(data.get('success')),
    'wall_seconds': wall_seconds,
    'script_elapsed_seconds': data.get('elapsed_seconds'),
    'steps': data.get('steps'),
    'train_examples': data.get('train_examples'),
    'dev_examples': data.get('dev_examples'),
    'eval_examples': data.get('eval_examples'),
    'baseline_eval_accuracy': base_eval_acc,
    'selected_eval_accuracy': sel_eval_acc,
    'eval_accuracy_delta': None if base_eval_acc is None or sel_eval_acc is None else sel_eval_acc - base_eval_acc,
    'convincing_eval_accuracy_threshold': threshold,
    'convincing_smoke': convincing,
    'baseline_dev_accuracy': base_dev_acc,
    'selected_dev_accuracy': sel_dev_acc,
    'baseline_eval_loss': base_eval_loss,
    'selected_eval_loss': sel_eval_loss,
    'eval_loss_delta_lower_is_better': None if base_eval_loss is None or sel_eval_loss is None else sel_eval_loss - base_eval_loss,
    'baseline_dev_loss': base_dev_loss,
    'selected_dev_loss': sel_dev_loss,
    'dev_loss_delta_lower_is_better': None if base_dev_loss is None or sel_dev_loss is None else sel_dev_loss - base_dev_loss,
    'privacy_transcript_audit_passed': bool(checks.get('privacy_transcript_audit_passed')),
    'release_rule_violation_count': (privacy.get('privacy_audit') or {}).get('release_rule_violation_count'),
    'fd_finite_rate_ok': bool(checks.get('fd_finite_rate_ok')),
    'fd_signal_rate_ok': bool(checks.get('fd_signal_rate_ok')),
    'json_path': str(json_path),
}

with summary_jsonl.open('a') as fh:
    fh.write(json.dumps(record, sort_keys=True) + '\n')

lines = []
lines.append(f"Result for {label}:")
lines.append(f"  success: {'PASS' if record['success'] else 'FAIL'}")
lines.append(f"  wall_seconds: {wall_seconds}")
lines.append(f"  steps/train/dev/eval: {record['steps']}/{record['train_examples']}/{record['dev_examples']}/{record['eval_examples']}")
lines.append(f"  eval_accuracy: {base_eval_acc} -> {sel_eval_acc} (delta {record['eval_accuracy_delta']})")
lines.append(f"  convincing_threshold: {threshold}")
lines.append(f"  convincing_smoke: {'PASS' if convincing else 'FAIL'}")
lines.append(f"  dev_accuracy: {base_dev_acc} -> {sel_dev_acc}")
lines.append(f"  eval_loss: {base_eval_loss} -> {sel_eval_loss} (lower is better; delta {record['eval_loss_delta_lower_is_better']})")
lines.append(f"  dev_loss: {base_dev_loss} -> {sel_dev_loss} (lower is better; delta {record['dev_loss_delta_lower_is_better']})")
lines.append(f"  privacy_audit: {'PASS' if record['privacy_transcript_audit_passed'] else 'FAIL'}")
lines.append(f"  release_rule_violation_count: {record['release_rule_violation_count']}")
lines.append(f"  result_json: {json_path}")
text = '\n'.join(lines)
print(text)
with summary_txt.open('a') as fh:
    fh.write(text + '\n')

raise SystemExit(0 if convincing else 2)
PY
  status=$?
  set -e

  if [ "$status" = "0" ]; then
    found="1"
    log "Found first convincing short utility-smoke run: $label" | tee -a "$summary_txt"
    if [ "$STOP_ON_CONVINCING" = "1" ]; then
      break
    fi
  elif [ "$status" = "2" ]; then
    log "$label did not meet the convincing-smoke threshold; continuing." | tee -a "$summary_txt"
  else
    fail "Result parser failed for $label with status $status."
  fi

done

hr | tee -a "$summary_txt"
if [ "$found" = "1" ]; then
  log "Sweep result: found a convincing utility-smoke configuration." | tee -a "$summary_txt"
else
  log "Sweep result: no tested configuration reached the convincing utility-smoke threshold." | tee -a "$summary_txt"
  log "Interpretation: publish mechanism validation, but do not claim utility reproduction from this sweep." | tee -a "$summary_txt"
fi

log "Summary JSONL: $summary_jsonl" | tee -a "$summary_txt"
log "Summary text:  $summary_txt" | tee -a "$summary_txt"
cat "$summary_txt"

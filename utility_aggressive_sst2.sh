#!/usr/bin/env bash
set -euo pipefail

# Experimental aggressive SST-2 utility sweep for PAC-Zero MLX validation.
#
# This script is intentionally more aggressive than utility_sweep_sst2.sh.
# It tries larger learning rates, perturbation scales, multiple seeds, and
# optionally fewer adapted layers. It is meant to answer:
#
#   Can this MLX LoRA/ZO setup produce a visible SST-2 accuracy lift at all?
#
# It does NOT weaken the success criterion. A run is counted as convincing only
# if selected eval accuracy is at least max(0.60, baseline_eval_accuracy + 0.05).
#
# Interpretation:
#   - If non-private runs pass but ZPL runs fail: utility machinery works, but
#     the audited ZPL release is too weak/noisy at this scale.
#   - If neither non-private nor ZPL runs pass: the current prompt/model/scoring
#     or LoRA/ZO setup is not producing SST-2 utility reproduction.
#   - If a ZPL run passes: that is useful small-scale SST-2 utility-smoke evidence.

MODEL="${MODEL:-mlx-community/SmolLM-135M-4bit}"
SLUG="${SLUG:-smollm-135m-4bit}"
PROJECTIONS="${PROJECTIONS:-q_proj,v_proj}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16.0}"
CLIP="${CLIP:-25.0}"
NUM_SUBSETS="${NUM_SUBSETS:-126}"
BASE_SEED="${BASE_SEED:-20260710}"
STOP_ON_CONVINCING="${STOP_ON_CONVINCING:-1}"
RUN_NONPRIVATE="${RUN_NONPRIVATE:-1}"
RUN_ZPL="${RUN_ZPL:-1}"

hr() { printf '\n%s\n' "----------------------------------------------------------------------"; }
log() { printf '[paczero-aggressive] %s\n' "$*"; }
fail() { printf '\n[paczero-aggressive][error] %s\n' "$*" >&2; exit 1; }

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

summary_dir="benchmark-results/paczero-sst2-aggressive-utility"
mkdir -p "$summary_dir"
summary_jsonl="$summary_dir/aggressive_results.jsonl"
summary_txt="$summary_dir/aggressive_summary.txt"
: > "$summary_jsonl"
: > "$summary_txt"

cat <<'INTRO' | tee -a "$summary_txt"
PAC-Zero MLX aggressive SST-2 utility sweep

Criterion for convincing utility-smoke evidence:
  selected_eval_accuracy >= max(0.60, baseline_eval_accuracy + 0.05)

This script is experimental. It is more aggressive than the archival validation
path and is intended to discover whether any short configuration produces a
visible SST-2 accuracy lift.
INTRO

# Format:
# label kind steps train dev eval layers lr mu eval_every seed_offset
# kind = nonprivate or zpl
#
# Order matters: first cheap non-private sanity checks, then increasingly
# aggressive ZPL settings. Fewer layers can improve signal/noise and speed.
configs=(
  "np_layers4_lr02_mu05 nonprivate 150 128 128 256 4 0.20 0.05 10 1"
  "np_layers8_lr02_mu05 nonprivate 150 128 128 256 8 0.20 0.05 10 2"
  "np_all_lr02_mu05 nonprivate 200 128 128 256 all 0.20 0.05 10 3"
  "np_all_lr05_mu05 nonprivate 200 128 128 256 all 0.50 0.05 10 4"
  "np_all_lr02_mu10 nonprivate 200 128 128 256 all 0.20 0.10 10 5"
  "zpl_layers4_lr02_mu05 zpl 200 128 128 256 4 0.20 0.05 10 11"
  "zpl_layers8_lr02_mu05 zpl 200 128 128 256 8 0.20 0.05 10 12"
  "zpl_all_lr02_mu05 zpl 250 128 128 256 all 0.20 0.05 10 13"
  "zpl_all_lr05_mu05 zpl 250 128 128 256 all 0.50 0.05 10 14"
  "zpl_all_lr02_mu10 zpl 250 128 128 256 all 0.20 0.10 10 15"
  "zpl_layers8_lr01_mu05_big zpl 350 256 256 512 8 0.10 0.05 10 21"
  "zpl_all_lr01_mu05_big zpl 350 256 256 512 all 0.10 0.05 10 22"
)

found_nonprivate="0"
found_zpl="0"

parse_result() {
  json_out="$1"
  label="$2"
  kind="$3"
  wall_seconds="$4"
  set +e
  "$PYTHON_BIN" - "$json_out" "$label" "$kind" "$wall_seconds" "$summary_jsonl" "$summary_txt" <<'PY'
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
label = sys.argv[2]
kind = sys.argv[3]
wall_seconds = int(sys.argv[4])
summary_jsonl = Path(sys.argv[5])
summary_txt = Path(sys.argv[6])

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
    'kind': kind,
    'success': bool(data.get('success')),
    'convincing_smoke': convincing,
    'wall_seconds': wall_seconds,
    'script_elapsed_seconds': data.get('elapsed_seconds'),
    'steps': data.get('steps'),
    'train_examples': data.get('train_examples'),
    'dev_examples': data.get('dev_examples'),
    'eval_examples': data.get('eval_examples'),
    'layers': data.get('layers'),
    'rank': data.get('rank'),
    'alpha': data.get('alpha'),
    'lr': data.get('lr'),
    'mu': data.get('mu'),
    'baseline_eval_accuracy': base_eval_acc,
    'selected_eval_accuracy': sel_eval_acc,
    'eval_accuracy_delta': None if base_eval_acc is None or sel_eval_acc is None else sel_eval_acc - base_eval_acc,
    'convincing_eval_accuracy_threshold': threshold,
    'baseline_dev_accuracy': base_dev_acc,
    'selected_dev_accuracy': sel_dev_acc,
    'dev_accuracy_delta': None if base_dev_acc is None or sel_dev_acc is None else sel_dev_acc - base_dev_acc,
    'baseline_eval_loss': base_eval_loss,
    'selected_eval_loss': sel_eval_loss,
    'eval_loss_delta_lower_is_better': None if base_eval_loss is None or sel_eval_loss is None else sel_eval_loss - base_eval_loss,
    'baseline_dev_loss': base_dev_loss,
    'selected_dev_loss': sel_dev_loss,
    'dev_loss_delta_lower_is_better': None if base_dev_loss is None or sel_dev_loss is None else sel_dev_loss - base_dev_loss,
    'privacy_transcript_audit_passed': bool(checks.get('privacy_transcript_audit_passed')) if kind == 'zpl' else None,
    'release_rule_violation_count': (privacy.get('privacy_audit') or {}).get('release_rule_violation_count') if kind == 'zpl' else None,
    'fd_finite_rate': data.get('fd_finite_rate'),
    'fd_signal_rate': data.get('fd_signal_rate'),
    'json_path': str(json_path),
}

with summary_jsonl.open('a') as fh:
    fh.write(json.dumps(record, sort_keys=True) + '\n')

lines = []
lines.append(f"Result for {label} [{kind}]:")
lines.append(f"  success: {'PASS' if record['success'] else 'FAIL'}")
lines.append(f"  convincing_smoke: {'PASS' if convincing else 'FAIL'}")
lines.append(f"  wall_seconds: {wall_seconds}")
lines.append(f"  steps/train/dev/eval: {record['steps']}/{record['train_examples']}/{record['dev_examples']}/{record['eval_examples']}")
lines.append(f"  layers/rank/alpha: {record['layers']}/{record['rank']}/{record['alpha']}")
lines.append(f"  lr/mu: {record['lr']}/{record['mu']}")
lines.append(f"  eval_accuracy: {base_eval_acc} -> {sel_eval_acc} (delta {record['eval_accuracy_delta']})")
lines.append(f"  convincing_threshold: {threshold}")
lines.append(f"  dev_accuracy: {base_dev_acc} -> {sel_dev_acc} (delta {record['dev_accuracy_delta']})")
lines.append(f"  eval_loss: {base_eval_loss} -> {sel_eval_loss} (lower is better; delta {record['eval_loss_delta_lower_is_better']})")
lines.append(f"  dev_loss: {base_dev_loss} -> {sel_dev_loss} (lower is better; delta {record['dev_loss_delta_lower_is_better']})")
if kind == 'zpl':
    lines.append(f"  privacy_audit: {'PASS' if record['privacy_transcript_audit_passed'] else 'FAIL'}")
    lines.append(f"  release_rule_violation_count: {record['release_rule_violation_count']}")
lines.append(f"  fd_finite/signal: {record['fd_finite_rate']}/{record['fd_signal_rate']}")
lines.append(f"  result_json: {json_path}")
text = '\n'.join(lines)
print(text)
with summary_txt.open('a') as fh:
    fh.write(text + '\n')

raise SystemExit(0 if convincing else 2)
PY
  status=$?
  set -e
  return "$status"
}

for row in "${configs[@]}"; do
  read -r label kind steps train_examples dev_examples eval_examples layers lr mu eval_every seed_offset <<< "$row"

  if [ "$kind" = "nonprivate" ] && [ "$RUN_NONPRIVATE" != "1" ]; then
    continue
  fi
  if [ "$kind" = "zpl" ] && [ "$RUN_ZPL" != "1" ]; then
    continue
  fi

  seed=$((BASE_SEED + seed_offset))
  out_dir="$summary_dir/$label"
  adapter_dir="$summary_dir/${label}-adapters"
  mkdir -p "$out_dir" "$adapter_dir"
  json_out="$out_dir/smollm_validation_results.json"
  adapter_out="$adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"

  hr | tee -a "$summary_txt"
  log "Running $label kind=$kind steps=$steps train/dev/eval=$train_examples/$dev_examples/$eval_examples layers=$layers lr=$lr mu=$mu seed=$seed" | tee -a "$summary_txt"

  start_epoch=$(date +%s)
  if [ "$kind" = "nonprivate" ]; then
    "$PYTHON_BIN" scripts/paczero_smollm_nonprivate_utility_control.py \
      --model "$MODEL" \
      --slug "$SLUG" \
      --task sst2 \
      --projections "$PROJECTIONS" \
      --layers "$layers" \
      --rank "$RANK" \
      --alpha "$ALPHA" \
      --seed "$seed" \
      --steps "$steps" \
      --train-examples "$train_examples" \
      --dev-examples "$dev_examples" \
      --eval-examples "$eval_examples" \
      --mu "$mu" \
      --lr "$lr" \
      --eval-every "$eval_every" \
      --json-out "$json_out" \
      --adapter-out "$adapter_out"
  else
    "$PYTHON_BIN" scripts/paczero_mlxlm_faithful_adaptation.py \
      --model "$MODEL" \
      --slug "$SLUG" \
      --task sst2 \
      --projections "$PROJECTIONS" \
      --layers "$layers" \
      --rank "$RANK" \
      --alpha "$ALPHA" \
      --seed "$seed" \
      --steps "$steps" \
      --train-examples "$train_examples" \
      --dev-examples "$dev_examples" \
      --eval-examples "$eval_examples" \
      --num-subsets "$NUM_SUBSETS" \
      --mu "$mu" \
      --lr "$lr" \
      --clip "$CLIP" \
      --eval-every "$eval_every" \
      --json-out "$json_out" \
      --adapter-out "$adapter_out"
  fi
  end_epoch=$(date +%s)
  wall_seconds=$((end_epoch - start_epoch))

  if parse_result "$json_out" "$label" "$kind" "$wall_seconds"; then
    if [ "$kind" = "nonprivate" ]; then
      found_nonprivate="1"
    else
      found_zpl="1"
    fi
    log "Found convincing $kind utility-smoke run: $label" | tee -a "$summary_txt"
    if [ "$STOP_ON_CONVINCING" = "1" ]; then
      break
    fi
  else
    status=$?
    if [ "$status" = "2" ]; then
      log "$label did not meet the convincing-smoke threshold; continuing." | tee -a "$summary_txt"
    else
      fail "Result parser failed for $label with status $status."
    fi
  fi

done

hr | tee -a "$summary_txt"
if [ "$found_zpl" = "1" ]; then
  log "Aggressive sweep result: found a convincing ZPL utility-smoke configuration." | tee -a "$summary_txt"
elif [ "$found_nonprivate" = "1" ]; then
  log "Aggressive sweep result: found a convincing non-private utility configuration, but not a ZPL one." | tee -a "$summary_txt"
  log "Interpretation: utility machinery can move SST-2, but the audited ZPL release remains the bottleneck." | tee -a "$summary_txt"
else
  log "Aggressive sweep result: no tested configuration reached the convincing utility-smoke threshold." | tee -a "$summary_txt"
  log "Interpretation: keep the artifact framed as mechanism validation, not utility reproduction." | tee -a "$summary_txt"
fi

log "Summary JSONL: $summary_jsonl" | tee -a "$summary_txt"
log "Summary text:  $summary_txt" | tee -a "$summary_txt"
cat "$summary_txt"

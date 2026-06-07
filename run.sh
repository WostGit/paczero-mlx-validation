#!/usr/bin/env bash
set -euo pipefail

# PAC-Zero MLX validation helper.
# Fast path: quick/aggregate/negative-control use the built-in macOS python3 when possible.
# MLX path: install/local-* first try the available Apple Silicon python3 with --user packages,
# then fall back to Homebrew/Python 3.11 only if that fails.

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
VENV_DIR="${VENV_DIR:-.venv}"
AUTO_INSTALL_HOMEBREW="${AUTO_INSTALL_HOMEBREW:-ask}"
AUTO_BREW_INSTALL_PYTHON="${AUTO_BREW_INSTALL_PYTHON:-1}"
PACZERO_SKIP_VENV="${PACZERO_SKIP_VENV:-0}"
PACZERO_USE_SYSTEM_PYTHON="${PACZERO_USE_SYSTEM_PYTHON:-1}"

PYTHON_BIN=""
BREW_BIN=""

hr() { printf '\n%s\n' "----------------------------------------------------------------------"; }
log() { printf '[paczero] %s\n' "$*"; }
warn() { printf '[paczero][warning] %s\n' "$*" >&2; }
fail() { printf '\n[paczero][error] %s\n' "$*" >&2; exit 1; }
run_cmd() { log "Running: $*"; "$@"; }

usage() {
  cat <<'USAGE'
PAC-Zero MLX validation helper

Fast archive check, no Homebrew needed on standard macOS:
  bash run.sh quick

Full local MLX run, now also tries macOS python3 first:
  bash run.sh install
  bash run.sh local-all

Modes:
  quick                 Use available python3, compile scripts, run negative control, rebuild aggregate report, print final summary.
  aggregate             Use available python3, rebuild aggregate report from included JSON files, print final summary.
  negative-control      Use available python3, run only the ZPL negative-control check.
  doctor                Print diagnostics only.
  bootstrap             Install/check MLX dependencies.
  install               Install/check MLX dependencies.
  local-sst2            Run local MLX validation on SST-2.
  local-squad           Run local MLX validation on SQuAD.
  local-control-sst2    Run local non-private utility control on SST-2.
  local-control-squad   Run local non-private utility control on SQuAD.
  local-all             Run all local MLX tasks, rebuild aggregate report, print final summary.

Environment knobs:
  PACZERO_USE_SYSTEM_PYTHON=1   default; try /usr/bin/python3 + --user packages for MLX first.
  PACZERO_USE_SYSTEM_PYTHON=0   force venv/Python 3.11+ path.
  AUTO_INSTALL_HOMEBREW=1       allow Homebrew install without prompt if fallback is needed.

Notes:
  - quick/aggregate/negative-control intentionally accept Python 3.9+.
  - On recent Apple Silicon macOS, mlx/mlx-lm may install successfully under the system Python 3.9 user site.
  - If the system Python path fails, install/local-* fall back to Python 3.11+ via Homebrew.
USAGE
}

ensure_repo_root() {
  hr
  log "Checking repository layout."
  log "Current directory: $(pwd)"
  [ -d scripts ] || fail "Run from repository root; ./scripts was not found."
  [ -f scripts/paczero_smollm_validation_aggregate.py ] || fail "Missing scripts/paczero_smollm_validation_aggregate.py"
  log "Repository layout looks OK."
}

print_system_info() {
  hr
  log "System diagnostics"
  log "Date UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "Shell: ${SHELL:-unknown}"
  log "PATH: ${PATH:-empty}"
  command -v uname >/dev/null 2>&1 && log "uname: $(uname -a)"
  command -v sw_vers >/dev/null 2>&1 && log "macOS: $(sw_vers -productName) $(sw_vers -productVersion) build $(sw_vers -buildVersion)"
  if [ "$(uname -s 2>/dev/null || echo unknown)" = "Darwin" ] && [ "$(uname -m 2>/dev/null || echo unknown)" = "arm64" ]; then
    log "Apple Silicon detected: yes"
  else
    warn "Apple Silicon was not detected. Full MLX runs may not work."
  fi
}

find_brew() {
  if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
  elif [ -x /opt/homebrew/bin/brew ]; then
    BREW_BIN="/opt/homebrew/bin/brew"
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [ -x /usr/local/bin/brew ]; then
    BREW_BIN="/usr/local/bin/brew"
    eval "$(/usr/local/bin/brew shellenv)"
  else
    BREW_BIN=""
  fi
}

print_tool_locations() {
  hr
  log "Tool diagnostics"
  find_brew
  for tool in git bash zsh curl xcode-select python3 python3.11 python pip3; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "$tool: $(command -v "$tool")"
    else
      warn "$tool: not found"
    fi
  done
  [ -n "$BREW_BIN" ] && log "brew: $BREW_BIN" || warn "brew: not found"
}

find_any_python() {
  if command -v python3.11 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3.11)";
  elif [ -x /opt/homebrew/bin/python3.11 ]; then PYTHON_BIN="/opt/homebrew/bin/python3.11";
  elif [ -x /usr/local/bin/python3.11 ]; then PYTHON_BIN="/usr/local/bin/python3.11";
  elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)";
  elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)";
  else PYTHON_BIN=""; fi
}

find_system_python() {
  if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)";
  elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)";
  else PYTHON_BIN=""; fi
}

python_at_least() {
  min_major="$1"; min_minor="$2"
  [ -n "$PYTHON_BIN" ] || return 1
  "$PYTHON_BIN" - "$min_major" "$min_minor" <<'PY'
import sys
maj = int(sys.argv[1]); minor = int(sys.argv[2])
raise SystemExit(0 if sys.version_info >= (maj, minor) else 1)
PY
}

ensure_light_python() {
  hr
  log "Checking lightweight Python for quick/archive checks."
  find_any_python
  [ -n "$PYTHON_BIN" ] || fail "No Python found. Install Python 3 or run xcode-select --install."
  log "Using Python candidate: $PYTHON_BIN"
  log "Version: $($PYTHON_BIN --version 2>&1)"
  python_at_least 3 9 || fail "Python 3.9+ is required for quick/archive checks."
  log "Python is sufficient for quick/archive checks. No Homebrew bootstrap needed."
}

ask_yes_no() {
  question="$1"; default_answer="$2"
  [ -t 0 ] || { [ "$default_answer" = "yes" ]; return $?; }
  printf '%s ' "$question"
  read -r answer
  answer="${answer:-$default_answer}"
  case "$answer" in y|Y|yes|YES|Yes) return 0 ;; *) return 1 ;; esac
}

ensure_homebrew() {
  find_brew
  if [ -n "$BREW_BIN" ]; then log "Homebrew available: $BREW_BIN"; return 0; fi
  warn "Homebrew is missing. It is only needed if the no-Homebrew Python path fails."
  if [ "$AUTO_INSTALL_HOMEBREW" = "0" ]; then fail "Homebrew install disabled."; fi
  if [ "$AUTO_INSTALL_HOMEBREW" = "1" ] || ask_yes_no "Install Homebrew now? [y/N]" "no"; then
    run_cmd /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    find_brew
    [ -n "$BREW_BIN" ] || fail "Homebrew still not discoverable. Open a new terminal and retry."
  else
    fail "Homebrew installation declined."
  fi
}

ensure_python311() {
  hr
  log "Checking Python 3.11+ fallback path."
  find_any_python
  if [ -n "$PYTHON_BIN" ]; then log "Candidate: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"; fi
  if python_at_least 3 11; then log "Python 3.11+ is available."; return 0; fi
  [ "$AUTO_BREW_INSTALL_PYTHON" = "1" ] || fail "Python 3.11+ missing and auto install disabled."
  ensure_homebrew
  run_cmd "$BREW_BIN" install python@3.11
  eval "$($BREW_BIN shellenv)"
  find_any_python
  python_at_least 3 11 || fail "Python 3.11+ still missing after brew install."
}

ensure_venv() {
  if [ "$PACZERO_SKIP_VENV" = "1" ]; then log "PACZERO_SKIP_VENV=1; using $PYTHON_BIN directly."; return 0; fi
  hr
  log "Preparing virtual environment: $VENV_DIR"
  [ -d "$VENV_DIR" ] || run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
  [ -x "$VENV_DIR/bin/python" ] || fail "Broken venv: $VENV_DIR/bin/python missing."
  PYTHON_BIN="$VENV_DIR/bin/python"
  log "Using venv Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
}

ensure_pip() {
  hr
  log "Checking pip."
  "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || run_cmd "$PYTHON_BIN" -m ensurepip --upgrade
  log "pip: $($PYTHON_BIN -m pip --version)"
}

check_mlx_imports() {
  "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ['mlx', 'mlx_lm', 'huggingface_hub', 'safetensors', 'numpy', 'datasets'] if importlib.util.find_spec(name) is None]
if missing:
    print('missing:', ', '.join(missing))
    raise SystemExit(1)
print('MLX/runtime imports: ok')
PY
}

install_mlx_user_site() {
  find_system_python
  [ -n "$PYTHON_BIN" ] || return 1
  log "Trying no-Homebrew MLX install using system Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
  python_at_least 3 9 || return 1
  ensure_pip
  run_cmd "$PYTHON_BIN" -m pip install --user --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
  check_mlx_imports
}

install_mlx_venv() {
  ensure_python311
  ensure_venv
  ensure_pip
  log "Installing MLX/runtime dependencies into managed Python environment."
  run_cmd "$PYTHON_BIN" -m pip install --upgrade pip
  run_cmd "$PYTHON_BIN" -m pip install --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
  check_mlx_imports
}

install_deps() {
  hr
  log "Installing/checking MLX dependencies."
  if [ "$PACZERO_USE_SYSTEM_PYTHON" = "1" ]; then
    if install_mlx_user_site; then
      log "No-Homebrew system Python MLX setup succeeded."
      return 0
    fi
    warn "System Python MLX setup failed; falling back to managed Python 3.11+ path."
  fi
  install_mlx_venv
}

print_final_summary() {
  ensure_light_python
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path('benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_aggregate_results.json')
report = Path('benchmark-results/paczero-smollm-validation-aggregate/smollm_validation_report.md')
neg_path = Path('benchmark-results/paczero-smollm-validation-aggregate/zpl_negative_control_results.json')

print('\n' + '=' * 72)
print('PAC-ZERO MLX FINAL RESULT SUMMARY')
print('=' * 72)

if not path.exists():
    print(f'Aggregate JSON missing: {path}')
    print('=' * 72)
    raise SystemExit(0)

data = json.loads(path.read_text())

def yn(value):
    return 'PASS' if bool(value) else 'FAIL'

def fmt(value):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)

print(f'Overall success: {yn(data.get("success"))}')
print(f'Claim scope: {data.get("claim", "n/a")}')
print()

limitations = data.get('non_claims_limitations') or []
if limitations:
    print('Important limitations / non-claims:')
    for item in limitations:
        print(f'  - {item}')
    print()

checks = data.get('aggregate_checks') or {}
if checks:
    print('Aggregate checks:')
    for key in sorted(checks):
        print(f'  - {key}: {yn(checks[key])}')
    print()

neg = data.get('negative_control') or {}
if neg:
    print('Negative control:')
    print(f'  - success: {yn(neg.get("success"))}')
    print(f'  - M: {fmt(neg.get("num_subsets_M"))}')
    print(f'  - membership counts: {fmt(neg.get("membership_column_counts_unique"))}')
    print(f'  - good ZPL release passes audit: {yn((neg.get("checks") or {}).get("good_zpl_release_passes_audit"))}')
    print(f'  - bad secret-dependent release fails audit: {yn((neg.get("checks") or {}).get("bad_secret_dependent_release_fails_audit"))}')
    print(f'  - conclusion: {neg.get("conclusion", "n/a")}')
    print()

print('PAC-Zero/ZPL validation tasks:')
for task in data.get('tasks') or []:
    name = task.get('task', 'unknown')
    print(f'  - {name}: {yn(task.get("success"))}')
    print(f'      elapsed_seconds: {fmt(task.get("elapsed_seconds"))}')
    print(f'      model: {task.get("model", "n/a")}')
    print(f'      examples train/dev/eval: {fmt(task.get("train_examples"))}/{fmt(task.get("dev_examples"))}/{fmt(task.get("eval_examples"))}')
    print(f'      steps: {fmt(task.get("steps"))}')
    print(f'      M: {fmt(task.get("num_subsets_M"))}')
    print(f'      membership counts: {fmt(task.get("membership_counts"))}')
    print(f'      LoRA rank/alpha: {fmt(task.get("rank"))}/{fmt(task.get("alpha"))}')
    print(f'      q/v target count: {fmt(task.get("target_count"))}')
    print(f'      projections: {fmt(task.get("projections"))}')
    print(f'      FD finite/signal rates: {fmt(task.get("fd_finite_rate"))}/{fmt(task.get("fd_signal_rate"))}')
    print(f'      privacy audit: {yn(task.get("privacy_transcript_audit_passed"))}')
    print(f'      release-rule violations: {fmt(task.get("release_rule_violation_count"))}')
    print(f'      transcript independent by construction: {yn(task.get("transcript_independent_by_construction"))}')
    print(f'      ZPL utility not worse than baseline: {yn(task.get("utility_not_worse_than_baseline"))}')
    print(f'      adapter saved: {yn(task.get("adapter_saved"))}')
    baseline = task.get('baseline') or {}
    best = task.get('selected_best_adapter_eval') or task.get('best_checkpoint') or {}
    if baseline or best:
        print('      baseline vs selected checkpoint:')
        for metric in ['train_loss', 'train_accuracy', 'dev_loss', 'dev_accuracy', 'eval_loss', 'eval_accuracy']:
            if metric in baseline or metric in best:
                print(f'        {metric}: {fmt(baseline.get(metric))} -> {fmt(best.get(metric))}')
        if 'eval_answer_likelihood_metric' in baseline or 'eval_answer_likelihood_metric' in best:
            print(f'        metric note: {baseline.get("eval_answer_likelihood_metric") or best.get("eval_answer_likelihood_metric")}')
    print()

print('Non-private utility controls:')
for ctrl in data.get('nonprivate_utility_controls') or []:
    name = ctrl.get('task', 'unknown')
    print(f'  - {name}: {yn(ctrl.get("success"))}')
    print(f'      elapsed_seconds: {fmt(ctrl.get("elapsed_seconds"))}')
    print(f'      steps: {fmt(ctrl.get("steps"))}')
    print(f'      LoRA rank/alpha: {fmt(ctrl.get("rank"))}/{fmt(ctrl.get("alpha"))}')
    print(f'      q/v target count: {fmt(ctrl.get("target_count"))}')
    print(f'      FD finite/signal rates: {fmt(ctrl.get("fd_finite_rate"))}/{fmt(ctrl.get("fd_signal_rate"))}')
    print(f'      utility not worse than baseline: {yn(ctrl.get("utility_not_worse_than_baseline"))}')
    print(f'      adapter saved: {yn(ctrl.get("adapter_saved"))}')
    baseline = ctrl.get('baseline') or {}
    best = ctrl.get('selected_best_adapter_eval') or ctrl.get('best_checkpoint') or {}
    if baseline or best:
        print('      baseline vs selected checkpoint:')
        for metric in ['train_loss', 'train_accuracy', 'dev_loss', 'dev_accuracy', 'eval_loss', 'eval_accuracy']:
            if metric in baseline or metric in best:
                print(f'        {metric}: {fmt(baseline.get(metric))} -> {fmt(best.get(metric))}')
        if 'eval_answer_likelihood_metric' in baseline or 'eval_answer_likelihood_metric' in best:
            print(f'        metric note: {baseline.get("eval_answer_likelihood_metric") or best.get("eval_answer_likelihood_metric")}')
    print()

print('Output files:')
for p in [path, report, neg_path]:
    if p.exists():
        print(f'  - {p} ({p.stat().st_size} bytes)')
    else:
        print(f'  - {p} (missing)')
print('=' * 72)
PY
}

compile_scripts() { ensure_light_python; run_cmd "$PYTHON_BIN" -m py_compile scripts/*.py; }
run_negative_control() { ensure_light_python; run_cmd "$PYTHON_BIN" scripts/paczero_zpl_negative_control.py; }
run_aggregate() { ensure_light_python; run_cmd "$PYTHON_BIN" scripts/paczero_smollm_validation_aggregate.py; ls -lh benchmark-results/paczero-smollm-validation-aggregate || true; print_final_summary; }

print_run_config() {
  hr
  log "Run config: MODEL=$MODEL SLUG=$SLUG STEPS=$STEPS TRAIN/DEV/EVAL=$TRAIN_EXAMPLES/$DEV_EXAMPLES/$EVAL_EXAMPLES LAYERS=$LAYERS"
}

run_validation_task() {
  install_deps
  print_run_config
  task="$1"; seed="$2"
  out_dir="benchmark-results/paczero-smollm-validation/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-validation-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  run_cmd "$PYTHON_BIN" scripts/paczero_mlxlm_faithful_adaptation.py \
    --model "$MODEL" --slug "$SLUG" --task "$task" --projections "$PROJECTIONS" --layers "$LAYERS" \
    --rank "$RANK" --alpha "$ALPHA" --seed "$seed" --steps "$STEPS" \
    --train-examples "$TRAIN_EXAMPLES" --dev-examples "$DEV_EXAMPLES" --eval-examples "$EVAL_EXAMPLES" \
    --num-subsets "$NUM_SUBSETS" --mu "$MU" --lr "$LR" --clip "$CLIP" --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/smollm_validation_results.json" \
    --adapter-out "$adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"
}

run_utility_control_task() {
  install_deps
  print_run_config
  task="$1"; seed="$2"
  out_dir="benchmark-results/paczero-smollm-utility-control/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-utility-control-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_nonprivate_utility_control.py \
    --model "$MODEL" --slug "$SLUG" --task "$task" --projections "$PROJECTIONS" --layers "$LAYERS" \
    --rank "$RANK" --alpha "$ALPHA" --seed "$seed" --steps "$STEPS" \
    --train-examples "$TRAIN_EXAMPLES" --dev-examples "$DEV_EXAMPLES" --eval-examples "$EVAL_EXAMPLES" \
    --mu "$MU" --lr "$LR" --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/nonprivate_utility_control_results.json" \
    --adapter-out "$adapter_dir/nonprivate_all_layers_qv_lora_rank8_alpha16.npz"
}

bootstrap() { ensure_repo_root; print_system_info; print_tool_locations; install_deps; log "Bootstrap complete."; }
doctor() { ensure_repo_root; print_system_info; print_tool_locations; find_any_python; [ -n "$PYTHON_BIN" ] && log "Python candidate: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))" || true; }

ensure_repo_root
case "$MODE" in
  help|-h|--help) usage ;;
  doctor) doctor ;;
  bootstrap|install) bootstrap ;;
  aggregate) run_aggregate ;;
  negative-control) run_negative_control ;;
  quick) print_system_info; print_tool_locations; compile_scripts; run_negative_control; run_aggregate; log "Quick check complete." ;;
  local-sst2) run_validation_task sst2 20260615 ;;
  local-squad) run_validation_task squad 20260616 ;;
  local-control-sst2) run_utility_control_task sst2 20260618 ;;
  local-control-squad) run_utility_control_task squad 20260619 ;;
  local-all) run_validation_task sst2 20260615; run_validation_task squad 20260616; run_utility_control_task sst2 20260618; run_utility_control_task squad 20260619; run_negative_control; run_aggregate; log "Full local run complete." ;;
  *) echo "Unknown mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

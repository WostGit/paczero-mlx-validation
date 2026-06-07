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
  quick                 Use available python3, compile scripts, run negative control, rebuild aggregate report.
  aggregate             Use available python3, rebuild aggregate report from included JSON files.
  negative-control      Use available python3, run only the ZPL negative-control check.
  doctor                Print diagnostics only.
  bootstrap             Install/check MLX dependencies.
  install               Install/check MLX dependencies.
  local-sst2            Run local MLX validation on SST-2.
  local-squad           Run local MLX validation on SQuAD.
  local-control-sst2    Run local non-private utility control on SST-2.
  local-control-squad   Run local non-private utility control on SQuAD.
  local-all             Run all local MLX tasks and rebuild aggregate report.

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

compile_scripts() { ensure_light_python; run_cmd "$PYTHON_BIN" -m py_compile scripts/*.py; }
run_negative_control() { ensure_light_python; run_cmd "$PYTHON_BIN" scripts/paczero_zpl_negative_control.py; }
run_aggregate() { ensure_light_python; run_cmd "$PYTHON_BIN" scripts/paczero_smollm_validation_aggregate.py; ls -lh benchmark-results/paczero-smollm-validation-aggregate || true; }

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

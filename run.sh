#!/usr/bin/env bash
set -euo pipefail

# PAC-Zero MLX validation helper.
# Designed to be very explicit on a clean macOS machine.

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

Clean macOS quick start:
  git clone https://github.com/WostGit/paczero-mlx-validation.git
  cd paczero-mlx-validation
  bash run.sh quick

Useful modes:
  bash run.sh quick                 Set up Python if needed, compile scripts, run negative control, rebuild aggregate report.
  bash run.sh bootstrap             Set up Homebrew/Python/.venv/dependencies only.
  bash run.sh doctor                Print detailed diagnostics only.
  bash run.sh aggregate             Rebuild aggregate report from included JSON files.
  bash run.sh negative-control      Run only the ZPL negative-control check.
  bash run.sh install               Install Python dependencies into .venv.
  bash run.sh local-sst2            Run local MLX validation on SST-2.
  bash run.sh local-squad           Run local MLX validation on SQuAD.
  bash run.sh local-control-sst2    Run local non-private utility control on SST-2.
  bash run.sh local-control-squad   Run local non-private utility control on SQuAD.
  bash run.sh local-all             Run all local MLX tasks and rebuild aggregate report.

Fresh-machine behavior:
  - The script looks for Homebrew on PATH, /opt/homebrew/bin/brew, and /usr/local/bin/brew.
  - If Python 3.11+ is missing, it installs python@3.11 with Homebrew.
  - If Homebrew is missing, it asks before running the official Homebrew installer.
  - To allow Homebrew install without a prompt:
      AUTO_INSTALL_HOMEBREW=1 bash run.sh quick
  - To forbid Homebrew install:
      AUTO_INSTALL_HOMEBREW=0 bash run.sh quick

Runtime overrides:
  STEPS=10 TRAIN_EXAMPLES=4 DEV_EXAMPLES=4 EVAL_EXAMPLES=16 bash run.sh local-sst2

Notes:
  - Full MLX runs require macOS on Apple Silicon.
  - The quick mode is lightweight and mostly checks the archived result package.
USAGE
}

ensure_repo_root() {
  hr
  log "Checking repository layout."
  log "Current directory: $(pwd)"
  if [ ! -d scripts ]; then
    fail "run.sh must be executed from the repository root. Expected ./scripts but it was not found."
  fi
  if [ ! -f scripts/paczero_smollm_validation_aggregate.py ]; then
    fail "Missing expected script: scripts/paczero_smollm_validation_aggregate.py"
  fi
  log "Repository layout looks OK."
}

print_system_info() {
  hr
  log "System diagnostics"
  log "Date UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  log "Shell: ${SHELL:-unknown}"
  log "PATH: ${PATH:-empty}"
  if command -v uname >/dev/null 2>&1; then
    log "uname: $(uname -a)"
    log "machine: $(uname -m)"
  fi
  if command -v sw_vers >/dev/null 2>&1; then
    log "macOS: $(sw_vers -productName) $(sw_vers -productVersion) build $(sw_vers -buildVersion)"
  else
    warn "sw_vers not found; this may not be macOS."
  fi
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
  if [ -n "$BREW_BIN" ]; then
    log "brew: $BREW_BIN"
  else
    warn "brew: not found on PATH, /opt/homebrew/bin, or /usr/local/bin"
  fi
  if command -v xcode-select >/dev/null 2>&1; then
    if xcode-select -p >/dev/null 2>&1; then
      log "Xcode command line tools path: $(xcode-select -p)"
    else
      warn "Xcode command line tools are missing. Install with: xcode-select --install"
    fi
  fi
}

ask_yes_no() {
  question="$1"
  default_answer="$2"
  if [ ! -t 0 ]; then
    [ "$default_answer" = "yes" ]
    return $?
  fi
  printf '%s ' "$question"
  read -r answer
  answer="${answer:-$default_answer}"
  case "$answer" in
    y|Y|yes|YES|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_homebrew() {
  find_brew
  if [ -n "$BREW_BIN" ]; then
    log "Homebrew available: $BREW_BIN"
    return 0
  fi

  warn "Homebrew is not installed or not discoverable."
  if [ "$AUTO_INSTALL_HOMEBREW" = "0" ]; then
    cat >&2 <<'EOF'

Install Homebrew manually, then rerun:

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  bash run.sh quick

EOF
    fail "Homebrew is required to install Python 3.11 automatically."
  fi

  if [ "$AUTO_INSTALL_HOMEBREW" = "1" ] || ask_yes_no "Install Homebrew now using the official installer? [y/N]" "no"; then
    hr
    log "Installing Homebrew. This may ask for your macOS password and may take several minutes."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    find_brew
    [ -n "$BREW_BIN" ] || fail "Homebrew installer finished, but brew is still not discoverable. Open a new terminal and retry."
    log "Homebrew installed/found: $BREW_BIN"
  else
    fail "Homebrew installation declined. Install Homebrew or Python 3.11 manually, then rerun."
  fi
}

find_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  elif [ -x /opt/homebrew/bin/python3.11 ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3.11"
  elif [ -x /usr/local/bin/python3.11 ]; then
    PYTHON_BIN="/usr/local/bin/python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN=""
  fi
}

python_version_ok() {
  [ -n "$PYTHON_BIN" ] || return 1
  "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
}

ensure_python() {
  hr
  log "Checking for Python 3.11+."
  find_python
  if [ -n "$PYTHON_BIN" ]; then
    log "Candidate Python: $PYTHON_BIN"
    log "Candidate version: $($PYTHON_BIN --version 2>&1)"
  else
    warn "No Python executable found."
  fi

  if python_version_ok; then
    log "Python is acceptable: $($PYTHON_BIN --version 2>&1)"
    return 0
  fi

  warn "Python 3.11+ is required; current candidate is missing or too old."
  if [ "$AUTO_BREW_INSTALL_PYTHON" != "1" ]; then
    fail "AUTO_BREW_INSTALL_PYTHON=0, so Python will not be installed automatically."
  fi

  ensure_homebrew
  hr
  log "Installing Python 3.11 via Homebrew."
  run_cmd "$BREW_BIN" install python@3.11
  eval "$($BREW_BIN shellenv)"
  find_python
  python_version_ok || fail "Python 3.11+ still not available after Homebrew install."
  log "Using Python: $PYTHON_BIN"
}

ensure_venv() {
  if [ "$PACZERO_SKIP_VENV" = "1" ]; then
    hr
    log "PACZERO_SKIP_VENV=1: using current Python directly: $PYTHON_BIN"
    return 0
  fi
  hr
  log "Preparing virtual environment at $VENV_DIR."
  if [ ! -d "$VENV_DIR" ]; then
    run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    log "Virtual environment already exists: $VENV_DIR"
  fi
  [ -x "$VENV_DIR/bin/python" ] || fail "Virtual environment is missing $VENV_DIR/bin/python. Remove $VENV_DIR and retry."
  PYTHON_BIN="$VENV_DIR/bin/python"
  log "Using virtualenv Python: $PYTHON_BIN"
  log "Virtualenv version: $($PYTHON_BIN --version 2>&1)"
}

ensure_pip() {
  hr
  log "Checking pip for $PYTHON_BIN."
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    warn "pip missing; trying ensurepip."
    run_cmd "$PYTHON_BIN" -m ensurepip --upgrade
  fi
  log "pip: $($PYTHON_BIN -m pip --version)"
}

prepare_python_env() {
  ensure_python
  ensure_venv
  ensure_pip
}

install_deps() {
  prepare_python_env
  hr
  log "Installing Python dependencies."
  run_cmd "$PYTHON_BIN" -m pip install --upgrade pip
  run_cmd "$PYTHON_BIN" -m pip install --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
  hr
  log "Dependency import check."
  "$PYTHON_BIN" - <<'PY'
import importlib.util, sys
print('python:', sys.version)
for name in ['mlx', 'mlx_lm', 'huggingface_hub', 'safetensors', 'numpy', 'datasets']:
    print(f'{name}:', 'ok' if importlib.util.find_spec(name) else 'missing')
PY
}

ensure_runtime_for_lightweight_modes() {
  prepare_python_env
}

compile_scripts() {
  ensure_runtime_for_lightweight_modes
  hr
  log "Compiling Python scripts."
  run_cmd "$PYTHON_BIN" -m py_compile scripts/*.py
}

run_negative_control() {
  ensure_runtime_for_lightweight_modes
  hr
  log "Running ZPL negative-control audit."
  run_cmd "$PYTHON_BIN" scripts/paczero_zpl_negative_control.py
}

run_aggregate() {
  ensure_runtime_for_lightweight_modes
  hr
  log "Rebuilding aggregate report from included JSON results."
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_validation_aggregate.py
  hr
  log "Aggregate directory contents:"
  ls -lh benchmark-results/paczero-smollm-validation-aggregate || true
}

print_run_config() {
  hr
  log "Run configuration"
  log "MODEL=$MODEL"
  log "SLUG=$SLUG"
  log "STEPS=$STEPS"
  log "TRAIN_EXAMPLES=$TRAIN_EXAMPLES"
  log "DEV_EXAMPLES=$DEV_EXAMPLES"
  log "EVAL_EXAMPLES=$EVAL_EXAMPLES"
  log "LAYERS=$LAYERS"
  log "PROJECTIONS=$PROJECTIONS"
  log "RANK=$RANK"
  log "ALPHA=$ALPHA"
  log "MU=$MU"
  log "LR=$LR"
  log "CLIP=$CLIP"
  log "EVAL_EVERY=$EVAL_EVERY"
  log "NUM_SUBSETS=$NUM_SUBSETS"
}

run_validation_task() {
  install_deps
  print_run_config
  task="$1"
  seed="$2"
  out_dir="benchmark-results/paczero-smollm-validation/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-validation-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  hr
  log "Running PAC-Zero/ZPL SmolLM validation task: $task"
  run_cmd "$PYTHON_BIN" scripts/paczero_mlxlm_faithful_adaptation.py \
    --model "$MODEL" --slug "$SLUG" --task "$task" \
    --projections "$PROJECTIONS" --layers "$LAYERS" \
    --rank "$RANK" --alpha "$ALPHA" --seed "$seed" \
    --steps "$STEPS" --train-examples "$TRAIN_EXAMPLES" --dev-examples "$DEV_EXAMPLES" --eval-examples "$EVAL_EXAMPLES" \
    --num-subsets "$NUM_SUBSETS" --mu "$MU" --lr "$LR" --clip "$CLIP" --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/smollm_validation_results.json" \
    --adapter-out "$adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"
}

run_utility_control_task() {
  install_deps
  print_run_config
  task="$1"
  seed="$2"
  out_dir="benchmark-results/paczero-smollm-utility-control/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-utility-control-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  hr
  log "Running non-private utility-control task: $task"
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_nonprivate_utility_control.py \
    --model "$MODEL" --slug "$SLUG" --task "$task" \
    --projections "$PROJECTIONS" --layers "$LAYERS" \
    --rank "$RANK" --alpha "$ALPHA" --seed "$seed" \
    --steps "$STEPS" --train-examples "$TRAIN_EXAMPLES" --dev-examples "$DEV_EXAMPLES" --eval-examples "$EVAL_EXAMPLES" \
    --mu "$MU" --lr "$LR" --eval-every "$EVAL_EVERY" \
    --json-out "$out_dir/nonprivate_utility_control_results.json" \
    --adapter-out "$adapter_dir/nonprivate_all_layers_qv_lora_rank8_alpha16.npz"
}

bootstrap() {
  ensure_repo_root
  print_system_info
  print_tool_locations
  install_deps
  hr
  log "Bootstrap complete. Next: bash run.sh quick"
}

doctor() {
  ensure_repo_root
  print_system_info
  print_tool_locations
  find_python
  if [ -n "$PYTHON_BIN" ]; then
    log "Python candidate: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
  else
    warn "No Python candidate found."
  fi
  hr
  log "Repository preview:"
  find . -maxdepth 3 -type f | sort | sed -n '1,160p'
}

ensure_repo_root

case "$MODE" in
  help|-h|--help)
    usage
    ;;
  doctor)
    doctor
    ;;
  bootstrap)
    bootstrap
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
    print_system_info
    print_tool_locations
    compile_scripts
    run_negative_control
    run_aggregate
    hr
    log "Quick check complete."
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
    hr
    log "Full local run complete."
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac

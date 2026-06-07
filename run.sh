#!/usr/bin/env bash
set -euo pipefail

# PAC-Zero MLX validation helper.
# This script is intentionally verbose so a fresh macOS user can see exactly
# what is being checked, installed, and run.

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
AUTO_INSTALL_HOMEBREW="${AUTO_INSTALL_HOMEBREW:-0}"
AUTO_BREW_INSTALL_PYTHON="${AUTO_BREW_INSTALL_PYTHON:-1}"
PACZERO_SKIP_VENV="${PACZERO_SKIP_VENV:-0}"

PYTHON_BIN=""
PIP_BIN=""

hr() {
  printf '\n%s\n' "----------------------------------------------------------------------"
}

log() {
  printf '[paczero] %s\n' "$*"
}

warn() {
  printf '[paczero][warning] %s\n' "$*" >&2
}

fail() {
  printf '\n[paczero][error] %s\n' "$*" >&2
  exit 1
}

run_cmd() {
  log "Running: $*"
  "$@"
}

usage() {
  cat <<'USAGE'
PAC-Zero MLX validation helper

Usage:
  bash run.sh quick
  bash run.sh bootstrap
  bash run.sh doctor
  bash run.sh aggregate
  bash run.sh negative-control
  bash run.sh install
  bash run.sh local-sst2
  bash run.sh local-squad
  bash run.sh local-control-sst2
  bash run.sh local-control-squad
  bash run.sh local-all

Recommended first check on any machine:
  bash run.sh quick

Recommended clean macOS setup:
  git clone https://github.com/WostGit/paczero-mlx-validation.git
  cd paczero-mlx-validation
  bash run.sh bootstrap
  bash run.sh quick

Recommended full local MLX run on Apple Silicon:
  bash run.sh bootstrap
  bash run.sh local-all

Modes:
  bootstrap            Detect macOS/Homebrew/Python, create .venv, install dependencies, then print environment details.
  doctor               Print detailed environment diagnostics without installing Python packages.
  quick                Compile scripts, run the negative control, rebuild the aggregate report from included JSON results.
  aggregate            Rebuild the aggregate report from included JSON results only.
  negative-control     Run the ZPL audit negative control only.
  install              Install Python package dependencies into the active or managed environment.
  local-sst2           Run the PAC-Zero/ZPL SmolLM validation task for SST-2.
  local-squad          Run the PAC-Zero/ZPL SmolLM validation task for SQuAD.
  local-control-sst2   Run the non-private utility control for SST-2.
  local-control-squad  Run the non-private utility control for SQuAD.
  local-all            Run both validation tasks, both utility controls, the negative control, then aggregate.

Environment overrides:
  STEPS=10 TRAIN_EXAMPLES=4 DEV_EXAMPLES=4 EVAL_EXAMPLES=16 bash run.sh local-sst2
  VENV_DIR=.venv bash run.sh bootstrap
  PACZERO_SKIP_VENV=1 bash run.sh quick
  AUTO_INSTALL_HOMEBREW=1 bash run.sh bootstrap
  AUTO_BREW_INSTALL_PYTHON=0 bash run.sh bootstrap

Notes:
  - Full local validation requires macOS on Apple Silicon with MLX support.
  - quick and aggregate modes are suitable for inspecting the archived result package.
  - bootstrap can install Python 3.11 with Homebrew if Homebrew is already installed.
  - bootstrap will not install Homebrew unless AUTO_INSTALL_HOMEBREW=1 is set.
USAGE
}

ensure_repo_root() {
  hr
  log "Checking repository layout."
  log "Current directory: $(pwd)"
  if [ ! -d scripts ]; then
    fail "run.sh must be executed from the repository root. Expected to find ./scripts."
  fi
  if [ ! -f scripts/paczero_smollm_validation_aggregate.py ]; then
    fail "Expected script missing: scripts/paczero_smollm_validation_aggregate.py"
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
    log "macOS product: $(sw_vers -productName) $(sw_vers -productVersion)"
    log "macOS build: $(sw_vers -buildVersion)"
  else
    warn "sw_vers not found. This does not look like macOS. Full MLX validation is intended for macOS/Apple Silicon."
  fi
  if [ "$(uname -s 2>/dev/null || echo unknown)" = "Darwin" ]; then
    if [ "$(uname -m 2>/dev/null || echo unknown)" = "arm64" ]; then
      log "Apple Silicon detected: yes"
    else
      warn "Apple Silicon not detected. MLX may not work or may be unsupported."
    fi
  fi
}

print_tool_locations() {
  hr
  log "Tool diagnostics"
  for tool in git bash zsh curl xcode-select brew python3 python3.11 python pip3; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "$tool: $(command -v "$tool")"
    else
      warn "$tool: not found"
    fi
  done
  if command -v xcode-select >/dev/null 2>&1; then
    if xcode-select -p >/dev/null 2>&1; then
      log "Xcode command line tools path: $(xcode-select -p)"
    else
      warn "Xcode command line tools are not installed. Install with: xcode-select --install"
    fi
  fi
}

install_homebrew_if_requested() {
  if command -v brew >/dev/null 2>&1; then
    log "Homebrew is already installed: $(command -v brew)"
    return 0
  fi

  warn "Homebrew is not installed."
  if [ "$AUTO_INSTALL_HOMEBREW" = "1" ]; then
    hr
    log "AUTO_INSTALL_HOMEBREW=1 set, so attempting Homebrew installation."
    log "This uses the official Homebrew installer and may ask for your macOS password."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -x /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
      eval "$(/usr/local/bin/brew shellenv)"
    fi
  else
    cat >&2 <<'EOF'

Homebrew is recommended for a clean macOS setup.
Install it manually with:

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Then reopen your terminal, cd back into this repository, and run:

  bash run.sh bootstrap

Alternatively, run this script with AUTO_INSTALL_HOMEBREW=1:

  AUTO_INSTALL_HOMEBREW=1 bash run.sh bootstrap

EOF
    fail "Homebrew not available, and automatic Homebrew installation was not requested."
  fi

  command -v brew >/dev/null 2>&1 || fail "Homebrew installation did not make brew available on PATH. Reopen terminal and retry."
  log "Homebrew is now available: $(command -v brew)"
}

find_python() {
  if [ -n "$PYTHON_BIN" ] && [ -x "$PYTHON_BIN" ]; then
    return 0
  fi
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
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
  log "Checking Python."
  find_python
  if [ -n "$PYTHON_BIN" ]; then
    log "Candidate Python: $PYTHON_BIN"
    log "Python version: $($PYTHON_BIN --version 2>&1)"
  else
    warn "No Python executable found."
  fi

  if python_version_ok; then
    log "Python version is acceptable: $($PYTHON_BIN --version 2>&1)"
    return 0
  fi

  warn "Python 3.11+ is required."
  if [ "$AUTO_BREW_INSTALL_PYTHON" != "1" ]; then
    fail "Python 3.11+ not available and AUTO_BREW_INSTALL_PYTHON=0. Install Python 3.11+ and retry."
  fi

  install_homebrew_if_requested
  hr
  log "Installing Python 3.11 with Homebrew."
  run_cmd brew install python@3.11

  if [ -x /opt/homebrew/bin/python3.11 ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3.11"
  elif [ -x /usr/local/bin/python3.11 ]; then
    PYTHON_BIN="/usr/local/bin/python3.11"
  else
    find_python
  fi

  python_version_ok || fail "Python 3.11+ still not available after installation attempt."
  log "Using Python: $PYTHON_BIN"
  log "Python version: $($PYTHON_BIN --version 2>&1)"
}

ensure_venv() {
  if [ "$PACZERO_SKIP_VENV" = "1" ]; then
    hr
    log "PACZERO_SKIP_VENV=1: using the current Python environment directly."
    PIP_BIN="$PYTHON_BIN -m pip"
    return 0
  fi

  hr
  log "Preparing virtual environment: $VENV_DIR"
  if [ ! -d "$VENV_DIR" ]; then
    run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    log "Virtual environment already exists: $VENV_DIR"
  fi

  if [ ! -x "$VENV_DIR/bin/python" ]; then
    fail "Virtual environment exists but $VENV_DIR/bin/python is missing. Remove $VENV_DIR and retry."
  fi

  PYTHON_BIN="$VENV_DIR/bin/python"
  PIP_BIN="$PYTHON_BIN -m pip"
  log "Virtualenv Python: $PYTHON_BIN"
  log "Virtualenv Python version: $($PYTHON_BIN --version 2>&1)"
}

ensure_pip() {
  hr
  log "Checking pip."
  if ! $PYTHON_BIN -m pip --version >/dev/null 2>&1; then
    warn "pip is not available for $PYTHON_BIN. Trying ensurepip."
    run_cmd "$PYTHON_BIN" -m ensurepip --upgrade
  fi
  log "pip: $($PYTHON_BIN -m pip --version)"
}

install_deps() {
  ensure_python
  ensure_venv
  ensure_pip
  hr
  log "Installing Python package dependencies."
  log "This may take a while on a fresh machine."
  run_cmd "$PYTHON_BIN" -m pip install --upgrade pip
  run_cmd "$PYTHON_BIN" -m pip install --upgrade mlx mlx-lm huggingface_hub hf_transfer safetensors numpy datasets
  hr
  log "Installed package summary."
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
print('python:', sys.version)
for name in ['mlx', 'mlx_lm', 'huggingface_hub', 'safetensors', 'numpy', 'datasets']:
    spec = importlib.util.find_spec(name)
    print(f'{name}:', 'ok' if spec else 'missing')
PY
}

bootstrap() {
  ensure_repo_root
  print_system_info
  print_tool_locations
  ensure_python
  ensure_venv
  ensure_pip
  install_deps
  hr
  log "Bootstrap complete."
  log "Next quick check: bash run.sh quick"
  log "Full local run:   bash run.sh local-all"
}

doctor() {
  ensure_repo_root
  print_system_info
  print_tool_locations
  ensure_python || true
  if [ -n "${PYTHON_BIN:-}" ]; then
    log "Final Python candidate: $PYTHON_BIN"
  fi
  hr
  log "Repository files preview:"
  find . -maxdepth 3 -type f | sort | sed -n '1,160p'
}

use_existing_or_managed_python() {
  ensure_python
  if [ "$PACZERO_SKIP_VENV" = "1" ]; then
    log "Using direct Python environment because PACZERO_SKIP_VENV=1."
  elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
    log "Using existing virtual environment: $PYTHON_BIN"
  else
    log "No virtual environment found. Creating one automatically."
    ensure_venv
  fi
  ensure_pip
}

compile_scripts() {
  use_existing_or_managed_python
  hr
  log "Compiling Python scripts."
  run_cmd "$PYTHON_BIN" -m py_compile scripts/*.py
}

run_negative_control() {
  use_existing_or_managed_python
  hr
  log "Running ZPL negative-control audit."
  run_cmd "$PYTHON_BIN" scripts/paczero_zpl_negative_control.py
}

run_aggregate() {
  use_existing_or_managed_python
  hr
  log "Rebuilding aggregate report from included result JSON files."
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_validation_aggregate.py
  hr
  log "Aggregate outputs:"
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
  use_existing_or_managed_python
  print_run_config
  task="$1"
  seed="$2"
  out_dir="benchmark-results/paczero-smollm-validation/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-validation-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  hr
  log "Running PAC-Zero/ZPL SmolLM validation task: $task"
  log "Output JSON: $out_dir/smollm_validation_results.json"
  log "Adapter output: $adapter_dir/all_layers_qv_lora_rank8_alpha16.npz"
  run_cmd "$PYTHON_BIN" scripts/paczero_mlxlm_faithful_adaptation.py \
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
  use_existing_or_managed_python
  print_run_config
  task="$1"
  seed="$2"
  out_dir="benchmark-results/paczero-smollm-utility-control/${SLUG}-${task}"
  adapter_dir="benchmark-results/paczero-smollm-utility-control-adapters/${SLUG}-${task}"
  mkdir -p "$out_dir" "$adapter_dir"
  hr
  log "Running non-private utility-control task: $task"
  log "Output JSON: $out_dir/nonprivate_utility_control_results.json"
  log "Adapter output: $adapter_dir/nonprivate_all_layers_qv_lora_rank8_alpha16.npz"
  run_cmd "$PYTHON_BIN" scripts/paczero_smollm_nonprivate_utility_control.py \
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

#!/usr/bin/env bash
# Set up AgentGarten on a CUDA 13.2 host (DGX): venv, CUDA torch, the wm package
# and the released weights. One step per call; nothing runs implicitly.
#
#   bash scripts/setup.sh setup      venv, CUDA 13.2 torch, wm, GPU check
#   bash scripts/setup.sh download   renderer checkpoint + Wan 2.2 VAE, fills .env
#   bash scripts/setup.sh all        setup, then download
#
# Overrides (environment): WEIGHTS (default <repo>/weights), PYTHON_VERSION
# (3.11), CUDA_TAG (cu132 = CUDA 13.2), TORCH_SPEC (the torch and torchvision
# pins; they must exist on that CUDA index).
# Opt-in extras (set to 1): WITH_BASE downloads Cosmos3-Nano too (16B, training
# only), WITH_DATA installs torchcodec, WITH_TE installs transformer-engine
# (compiled against the CUDA toolkit on the host), WITH_WANDB installs wandb.
# A Hugging Face token, if a repo asks for one, is read from HF_TOKEN by
# huggingface_hub and is never printed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

WEIGHTS="${WEIGHTS:-${ROOT}/weights}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_TAG="${CUDA_TAG:-cu132}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.12.0 torchvision==0.27.0}"
WITH_BASE="${WITH_BASE:-0}"
WITH_DATA="${WITH_DATA:-0}"
WITH_TE="${WITH_TE:-0}"
WITH_WANDB="${WITH_WANDB:-0}"

# cu132 -> 13.2, cu130 -> 13.0
cuda_digits="${CUDA_TAG#cu}"
WANT_CUDA="${cuda_digits%?}.${cuda_digits: -1}"

die() {
  echo "error: $*" >&2
  exit 1
}

check_driver() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found; needs an NVIDIA driver"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  local driver_cuda
  driver_cuda="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
  [[ -n "${driver_cuda}" ]] || die "could not read the driver CUDA version"
  echo "driver supports CUDA ${driver_cuda}, building for CUDA ${WANT_CUDA}"
  local lowest
  lowest="$(printf '%s\n%s\n' "${WANT_CUDA}" "${driver_cuda}" | sort -V | head -n1)"
  [[ "${lowest}" == "${WANT_CUDA}" ]] \
    || die "the driver supports CUDA ${driver_cuda}, below the ${WANT_CUDA} needed"
}

check_torch() {
  .venv/bin/python - "${WANT_CUDA}" <<'PY'
import sys

import torch

want = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a GPU: " + torch.__version__)
if torch.version.cuda != want:
    raise SystemExit(
        f"torch was built for CUDA {torch.version.cuda}, expected {want}: "
        f"{torch.__version__}"
    )
print(torch.__version__, "CUDA", torch.version.cuda, torch.cuda.get_device_name(0))
PY
}

cmd_setup() {
  command -v uv >/dev/null || die "uv not found on PATH"
  check_driver

  # uv venv wipes an existing environment, so only create it when missing.
  [[ -d .venv ]] || uv venv --python "${PYTHON_VERSION}" .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate

  echo "--- CUDA torch from the ${CUDA_TAG} index: ${TORCH_SPEC} ---"
  # TORCH_SPEC is split into separate requirements on purpose.
  # shellcheck disable=SC2086
  uv pip install ${TORCH_SPEC} --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    || die "no torch build on ${CUDA_TAG} for: ${TORCH_SPEC} (set CUDA_TAG or TORCH_SPEC)"
  check_torch

  local extras=media
  [[ "${WITH_DATA}" == 1 ]] && extras="${extras},data"
  [[ "${WITH_TE}" == 1 ]] && extras="${extras},te"
  [[ "${WITH_WANDB}" == 1 ]] && extras="${extras},wandb"
  echo "--- wm with extras: ${extras} ---"
  uv pip install huggingface_hub "transformers>=4.57"
  uv pip install -e ".[${extras}]"
  # The package install must not have replaced the CUDA torch with another build.
  check_torch
  [[ -f .env ]] || cp .env.example .env
}

cmd_download() {
  [[ -x .venv/bin/python ]] || die "no .venv; run: bash scripts/setup.sh setup"
  mkdir -p "${WEIGHTS}"
  echo "weights: ${WEIGHTS}, free: $(df -BG --output=avail "${WEIGHTS}" | tail -1 | tr -d ' ')"
  local base_flag=()
  [[ "${WITH_BASE}" == 1 ]] && base_flag=(--with-base)
  .venv/bin/python scripts/download_weights.py "${WEIGHTS}" \
    --env-file .env --env-example .env.example --python "${ROOT}/.venv/bin/python" \
    "${base_flag[@]}"
}

case "${1:-}" in
  setup) cmd_setup ;;
  download) cmd_download ;;
  all) cmd_setup && cmd_download ;;
  *) die "usage: bash scripts/setup.sh setup|download|all" ;;
esac

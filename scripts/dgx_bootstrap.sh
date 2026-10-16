#!/usr/bin/env bash
# Copy this AgentGarten checkout to the DGX and run its own scripts/setup.sh there (setup, then weight download).
# Wraps the project's real setup path (README "On a CUDA 13.2 host" section) - it does not invent any inference logic;
# the README's Cosmos3Stream example needs a real depth/normal-annotated clip manifest, which this does not build.
#
# Usage:   bash scripts/dgx_bootstrap.sh [--host user@host] [--remote-dir PATH] [--with-base]
#   --with-base   also download Cosmos3-Nano (needed for training, not for using a released renderer checkpoint)
# Checks free GPU memory on the DGX first (needs at least MIN_FREE_MIB, default 30000); the GPU there is shared.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

host="kschmid@10.74.11.118"
remote_dir="/localhome/kschmid/AgentGarten"
with_base=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="${2:?--host needs user@host}"; shift 2 ;;
    --remote-dir) remote_dir="${2:?--remote-dir needs a path}"; shift 2 ;;
    --with-base) with_base=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

min_free="${MIN_FREE_MIB:-30000}"
free_mib="$(ssh -o BatchMode=yes "${host}" 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1' | tr -d ' ')"
echo "GPU free on ${host}: ${free_mib} MiB (needs at least ${min_free})"
if [[ "${free_mib}" -lt "${min_free}" ]]; then
  echo "ERROR: not enough free GPU memory; the GPU is shared. Not starting." >&2
  exit 1
fi

echo "Copying ${root} -> ${host}:${remote_dir} (excluding .git, weights, outputs)"
ssh -o BatchMode=yes "${host}" "mkdir -p ${remote_dir}"
rsync -az --exclude .git --exclude weights --exclude outputs --exclude samples --exclude '__pycache__' \
  "${root}/" "${host}:${remote_dir}/"

echo "Running scripts/setup.sh setup on ${host}"
ssh -o BatchMode=yes "${host}" "cd ${remote_dir} && sed -i 's/\r\$//' scripts/*.sh && bash scripts/setup.sh setup"

echo "Running scripts/setup.sh download on ${host}${with_base:+ (WITH_BASE=1)}"
if [[ "${with_base}" -eq 1 ]]; then
  ssh -o BatchMode=yes "${host}" "cd ${remote_dir} && WITH_BASE=1 bash scripts/setup.sh download"
else
  ssh -o BatchMode=yes "${host}" "cd ${remote_dir} && bash scripts/setup.sh download"
fi

echo "Done. Renderer checkpoint and Wan 2.2 VAE are under ${remote_dir}/weights on ${host}."
echo "The README's Cosmos3Stream inference example still needs a real video/depth/normal manifest; nothing here builds one."

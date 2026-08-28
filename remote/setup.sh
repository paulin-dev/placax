#!/usr/bin/env bash
# One-time setup for a rented GPU box (e.g. vast.ai). Clone this repo onto the instance
# first (git clone https://github.com/paulin-dev/placax.git && cd placax), then run:
#
#   CUDA_EXTRA=cuda ./remote/setup.sh   (or CUDA_EXTRA=cuda12 for older drivers)
#
# Unlike Colab, these boxes don't ship a preinstalled CUDA-enabled JAX, so - unlike
# colab/run_maskplace.ipynb - this installs placax's own [cuda]/[cuda12] extra rather
# than reusing something already on the image. Check `nvidia-smi` (driver/CUDA version)
# if unsure which extra your instance needs; jax[cuda13] (this repo's default `cuda`
# extra) needs a fairly recent driver, jax[cuda12] is the safer fallback on older images.
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_EXTRA="${CUDA_EXTRA:-cuda}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python3 -m venv venv
source venv/bin/activate
pip install -q -U -e ".[${CUDA_EXTRA},resnet,viz]"

python3 -c "import jax; print(jax.devices())"

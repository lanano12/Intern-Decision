#!/usr/bin/env bash
# Environment for scripts/play_doom.sh. Torch stays the one already installed
# for this GPU (ROCm or CUDA). Transformers is pinned to the checkpoint's version.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_WITH_TORCH="${PYTHON_WITH_TORCH:-/opt/pytorch-rocm/bin/python}"
if [[ ! -x "$PYTHON_WITH_TORCH" ]]; then
  echo "Set PYTHON_WITH_TORCH to a Python that already imports torch with GPU support." >&2
  exit 1
fi
"$PYTHON_WITH_TORCH" -m venv --system-site-packages .venv-doom
SITE=$(.venv-doom/bin/python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
TORCH_SITE=$("$PYTHON_WITH_TORCH" -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parents[1])")
printf '%s\n' "$TORCH_SITE" > "$SITE/torch-prefix.pth"
.venv-doom/bin/python -m pip install -q --upgrade pip
.venv-doom/bin/python -m pip install -q --no-deps 'transformers==5.14.1'
.venv-doom/bin/python -m pip install -q \
  'huggingface-hub>=1.5,<2' 'regex>=2025.10.22' 'safetensors>=0.8' \
  'tokenizers>=0.22,<=0.23.0' einops vizdoom
# Triton kernels for Qwen3.5 linear attention. --no-deps keeps the ROCm torch in place.
# causal-conv1d is a CUDA extension and is not installed here; the conv stays on PyTorch.
.venv-doom/bin/python -m pip install -q --no-deps 'flash-linear-attention==0.5.2' 'fla-core==0.5.2'
.venv-doom/bin/python - << 'PY'
import torch, transformers, vizdoom
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} gpu={name}")
print(f"transformers {transformers.__version__} vizdoom {vizdoom.__version__}")
PY
echo "Next: .venv-doom/bin/python -c \"from huggingface_hub import snapshot_download; snapshot_download('internlm/Intern-Decision-4B', local_dir='checkpoints/Intern-Decision-4B')\""

#!/usr/bin/env bash
# install.sh - one-shot environment setup for AGD (mixture-augmented GazeReward)
#
# Run once from the repo root:
#     bash install.sh
#
# Environment variables:
#     SKIP_TORCH=1        # skip all torch/torchvision/nvidia-* installs (default: auto-detect)
#     FORCE_TORCH=1       # force-reinstall torch==2.2.2 stack (dangerous — may break existing CUDA)
#     SKIP_ET2_DL=1       # skip pre-downloading ET2 weights from HuggingFace

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "  AGD - one-shot install"
echo "  Repo root: $REPO_ROOT"
echo "================================================================"

# ─── Detect existing torch ──────────────────────────────────────────
TORCH_ALREADY=0
if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
    echo ""
    echo "Detected existing working PyTorch:"
    echo "  torch:       $TORCH_VERSION"
    echo "  CUDA:        $CUDA_VERSION"
    echo "  GPU:         $GPU_NAME"
    TORCH_ALREADY=1
fi

# Decide whether to skip the pinned torch stack
SKIP_TORCH="${SKIP_TORCH:-}"
FORCE_TORCH="${FORCE_TORCH:-}"
if [[ -n "$FORCE_TORCH" ]]; then
    echo ""
    echo "  FORCE_TORCH=1 set — will reinstall torch==2.2.2 stack."
    SKIP_TORCH_STACK=0
elif [[ -n "$SKIP_TORCH" ]]; then
    echo ""
    echo "  SKIP_TORCH=1 set — skipping torch/torchvision/nvidia-* installs."
    SKIP_TORCH_STACK=1
elif [[ "$TORCH_ALREADY" == "1" ]]; then
    echo ""
    echo "  Working torch+CUDA detected — will NOT touch torch/nvidia-* stack."
    echo "  (Set FORCE_TORCH=1 if you explicitly want to reinstall torch==2.2.2.)"
    SKIP_TORCH_STACK=1
else
    echo ""
    echo "  No working torch detected — will install torch==2.2.2 stack from requirements.txt."
    SKIP_TORCH_STACK=0
fi

pip_install() {
    echo ""
    echo ">>> pip install $*"
    pip install --quiet "$@"
}

# ─── 1. Core requirements ────────────────────────────────────────────
echo ""
echo "[1/9] Installing requirements.txt ..."

# Always filter the GitHub-installed packages from requirements.txt
# (they're installed separately with specific flags below).
FILTERS="eyetrackpy\\|tokenizer_aligner"

# Conditionally also filter the torch/torchvision/nvidia-* stack when the
# caller already has a working torch — reinstalling would either downgrade
# their torch or trigger a ~5GB CUDA-lib download just to satisfy a pinned
# version that the code doesn't actually require.
if [[ "$SKIP_TORCH_STACK" == "1" ]]; then
    FILTERS="$FILTERS\\|^torch\\|^torchvision\\|^torchviz\\|^nvidia-"
    echo "  (skipping pinned torch/torchvision/nvidia-* — using existing install)"
fi

grep -v "$FILTERS" requirements.txt | pip install --quiet -r /dev/stdin

# ─── 2. bitsandbytes (QLoRA) ─────────────────────────────────────────
echo ""
echo "[2/9] bitsandbytes (QLoRA) ..."
if python -c "import bitsandbytes" 2>/dev/null; then
    echo "  already installed — skip"
else
    pip_install bitsandbytes
fi

# ─── 3. safetensors ──────────────────────────────────────────────────
echo ""
echo "[3/9] safetensors ..."
if python -c "import safetensors" 2>/dev/null; then
    echo "  already installed — skip"
else
    pip_install safetensors
fi

# ─── 4. optuna ───────────────────────────────────────────────────────
echo ""
echo "[4/9] optuna (Hyperband tuning) ..."
if python -c "import optuna" 2>/dev/null; then
    echo "  already installed — skip"
else
    pip_install "optuna>=3.6" optuna-dashboard
fi

# ─── 5. nvidia-ml-py3 (VRAM watcher) ─────────────────────────────────
echo ""
echo "[5/9] nvidia-ml-py3 (VRAM watcher) ..."
if python -c "import pynvml" 2>/dev/null; then
    echo "  already installed — skip"
else
    pip_install nvidia-ml-py3
fi

# ─── 6. eyetrackpy (ET1 predictor) ───────────────────────────────────
echo ""
echo "[6/9] eyetrackpy from GitHub ..."
if python -c "import eyetrackpy" 2>/dev/null; then
    echo "  already installed — skip"
else
    pip_install "git+https://github.com/angelalopezcardona/eyetrackpy.git@v1.0.0"
fi

# ─── 7. tokenizer_aligner (ET1/ET2 subword mapping) ──────────────────
# Uses --no-deps because tokenizer_aligner 0.1 pins datasets==3.1.0 but
# requirements.txt needs datasets==3.2.0 (fully backward compatible).
echo ""
echo "[7/9] tokenizer_aligner from GitHub (--no-deps) ..."
if python -c "import tokenizer_aligner" 2>/dev/null; then
    echo "  already installed — skip"
else
    pip install --quiet --no-deps "git+https://github.com/angelalopezcardona/tokenizer_aligner.git@v1.0.0"
fi

# ─── 8. ET2 weights pre-download ─────────────────────────────────────
echo ""
echo "[8/9] Pre-downloading ET2 weights from HuggingFace ..."
if [[ -n "${SKIP_ET2_DL:-}" ]]; then
    echo "  SKIP_ET2_DL=1 — skipping."
    echo "  Set ET2_CHECKPOINT_PATH to your checkpoint, or run setup_et_models.py later."
else
    python - <<'PYEOF'
try:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="skboy/et_prediction_2",
        filename="et_predictor2_seed123.safetensors",
    )
    print(f"    ET2 weights cached at: {path}")
except Exception as e:
    print(f"    WARNING: ET2 download failed: {e}")
    print(f"    You can retry later with: python setup_et_models.py")
    print(f"    Or set ET2_CHECKPOINT_PATH to your own checkpoint.")
PYEOF
fi

# ─── 9. Smoke test ───────────────────────────────────────────────────
echo ""
echo "[9/9] Smoke-testing imports ..."
python - <<'PYEOF'
import sys
ok = True

def check(name, importer):
    global ok
    try:
        importer()
        print(f"  OK    {name}")
    except Exception as e:
        print(f"  FAIL  {name}  -- {e}")
        ok = False

check("PyTorch",      lambda: __import__("torch"))
check("Transformers", lambda: __import__("transformers"))
check("PEFT",         lambda: __import__("peft"))
check("TRL",          lambda: __import__("trl"))
check("Datasets",     lambda: __import__("datasets"))
check("BitsAndBytes", lambda: __import__("bitsandbytes"))
check("Safetensors",  lambda: __import__("safetensors"))
check("Optuna",       lambda: __import__("optuna"))
check("pynvml",       lambda: __import__("pynvml"))
check("LMDB",         lambda: __import__("lmdb"))
check("eyetrackpy",   lambda: __import__("eyetrackpy"))
check("WandB",        lambda: __import__("wandb"))
check("sklearn",      lambda: __import__("sklearn"))

import torch
gpu = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if gpu else "(not found)"
print(f"  {'OK' if gpu else 'FAIL'}  CUDA GPU ({name})")
if not ok:
    sys.exit(1)
PYEOF

echo ""
echo "================================================================"
echo "  Install complete."
echo ""
echo "  Quick-start:"
echo "    python scripts/h1_bic_sanity_check.py    # FIRST: verify mixture hypothesis"
echo "    python rlhf_rw/main.py ...               # see README.md for full command"
echo "    python optuna_tune.py --mode mixture --n_trials 20"
echo ""
echo "  See README.md for full usage, training commands, and hyperparameter guide."
echo "================================================================"

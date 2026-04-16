#!/usr/bin/env bash
# ============================================================
#  install.sh  one-shot environment setup for GazeReward
#
#  Run once from the repo root (personnel-main/):
#      bash install.sh
#
#  Install order:
#    1. requirements.txt  (GitHub packages filtered out — see note)
#    2. bitsandbytes
#    3. safetensors
#    4. optuna + optuna-dashboard
#    5. nvidia-ml-py3
#    6. eyetrackpy          (GitHub, normal install)
#    7. tokenizer_aligner   (GitHub, --no-deps  <-- KEY FIX)
#    8. ET2 weights pre-download from HuggingFace
#    9. Smoke test
#
#  WHY --no-deps for tokenizer_aligner:
#    tokenizeraligner 0.1 requires datasets==3.1.0
#    requirements.txt pins  datasets==3.2.0  -> pip conflict
#    --no-deps skips the conflicting pin.
#    datasets==3.2.0 is fully backward compatible.
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "================================================================"
echo "  GazeReward - one-shot install"
echo "  Repo root: $REPO_ROOT"
echo "================================================================"

pip_install() {
    echo ""
    echo ">>> pip install $*"
    pip install --quiet "$@"
}

# 1. Core requirements — filter out the two GitHub lines
echo ""
echo "[1/9] Installing requirements.txt (filtering GitHub packages) ..."
grep -v "eyetrackpy\|tokenizer_aligner" requirements.txt \
    | pip install --quiet -r /dev/stdin

# 2. bitsandbytes
echo ""
echo "[2/9] bitsandbytes (QLoRA) ..."
pip_install bitsandbytes

# 3. safetensors
echo ""
echo "[3/9] safetensors ..."
pip_install safetensors

# 4. optuna
echo ""
echo "[4/9] optuna (Hyperband tuning) ..."
pip_install "optuna>=3.6" "optuna-dashboard"

# 5. pynvml
echo ""
echo "[5/9] nvidia-ml-py3 (VRAM watcher) ..."
pip_install nvidia-ml-py3

# 6. eyetrackpy
echo ""
echo "[6/9] eyetrackpy from GitHub ..."
pip_install "git+https://github.com/angelalopezcardona/eyetrackpy.git@v1.0.0"

# 7. tokenizer_aligner with --no-deps
echo ""
echo "[7/9] tokenizer_aligner from GitHub (--no-deps to avoid datasets conflict) ..."
pip install --quiet --no-deps \
    "git+https://github.com/angelalopezcardona/tokenizer_aligner.git@v1.0.0"

# 8. Pre-warm ET2 weights
echo ""
echo "[8/9] Pre-downloading ET2 weights from HuggingFace ..."
python3 - <<'PYEOF'
try:
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download(
        repo_id="skboy/et_prediction_2",
        filename="et_predictor2_seed123.safetensors",
    )
    print(f"    ET2 weights cached at: {ckpt}")
except Exception as e:
    print(f"    [WARNING] {e}")
    print("    et2_wrapper.py will retry at first run.")
PYEOF

# 9. Smoke test
echo ""
echo "[9/9] Smoke-testing imports ..."
python3 - <<'PYEOF'
import sys
ok = True
checks = [
    ("torch",        "PyTorch"),
    ("transformers", "Transformers"),
    ("peft",         "PEFT"),
    ("trl",          "TRL"),
    ("datasets",     "Datasets"),
    ("bitsandbytes", "BitsAndBytes"),
    ("safetensors",  "Safetensors"),
    ("optuna",       "Optuna"),
    ("pynvml",       "pynvml"),
    ("lmdb",         "LMDB"),
    ("eyetrackpy",   "eyetrackpy"),
    ("wandb",        "WandB"),
]
for mod, name in checks:
    try:
        __import__(mod)
        print(f"  OK    {name}")
    except ImportError as e:
        print(f"  FAIL  {name} -- {e}")
        ok = False
import torch
gpu = torch.cuda.is_available()
print(f"  {'OK' if gpu else 'FAIL'}  CUDA GPU "
      f"{'(' + torch.cuda.get_device_name(0) + ')' if gpu else '(not found)'}")
if not ok:
    sys.exit(1)
PYEOF

echo ""
echo "================================================================"
echo "  Install complete."
echo ""
echo "  Quick-start:"
echo "    python scripts/h1_bic_sanity_check.py    # FIRST: verify mixture hypothesis"
echo "    bash run_modes.sh mixture                # mixture (per-response GMM)"
echo "    bash run_modes.sh lopez                  # Lopez-Cardona fcomb2.5 baseline"
echo "    python optuna_tune.py --mode mixture --n_trials 20"
echo ""
echo "  See README.md for full usage and hyperparameter guide."
echo "================================================================"

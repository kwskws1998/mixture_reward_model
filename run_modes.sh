#!/usr/bin/env bash
# ============================================================
#  run_modes.sh  –  GazeReward multi-feature run launcher
#
#  Experiment families:
#
#    MIXTURE       : fmv=2, all 5 features + per-response GMM
#                    summary token prepended to the input sequence.
#                    Replaces the deprecated TRT redistribution
#                    mechanism with a generative treatment of ET
#                    features as a mixture process.
#
#    MIXTURE_FULL  : same as MIXTURE but with full covariance GMM
#                    (captures inter-feature correlations).
#
#    LOPEZ         : fmv=2, fcomb2.5 style – all 5 features, no
#                    mixture token. Direct paper-baseline replication.
#
#    LOPEZ_22      : fmv=2, fcomb2.2 style – TRT + FFD only.
#    LOPEZ_1       : fmv=1, single-feature (TRT) baseline.
#    BASELINE      : no ET, pure reward model.
#    EVAL          : eval mode (set CKPT_DIR below).
#
#  Usage:
#    bash run_modes.sh mixture        # default mixture mode
#    bash run_modes.sh mixture_full   # full-covariance mixture
#    bash run_modes.sh lopez          # paper baseline (no mixture)
#    bash run_modes.sh baseline       # no ET
#    bash run_modes.sh eval           # eval mode (set CKPT_DIR)
#
#  Override any arg inline, e.g.:
#    DATASET=nvidia/HelpSteer2 MODEL=meta-llama/Meta-Llama-3-8B \
#        bash run_modes.sh mixture
#
#  Mixture-specific overrides:
#    MIX_K=4 MIX_COV=full bash run_modes.sh mixture
# ============================================================

set -euo pipefail

# ─── Configurable defaults (override via env) ───────────────
DATASET="${DATASET:-OpenAssistant/oasst1}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
BATCH="${BATCH:-8}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
LR_SCHED="${LR_SCHED:-cosine_with_min_lr}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SEED="${SEED:-42}"
FP_DROPOUT="${FP_DROPOUT:-0.1,0.3}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LOG_STEPS="${LOG_STEPS:-50}"
MAX_LEN="${MAX_LEN:-10000}"
CKPT_DIR="${CKPT_DIR:-./models_save/eval_run}"   # for eval mode only

# ─── Mixture-specific defaults (override via env) ───────────
MIX_K="${MIX_K:-3}"
MIX_COV="${MIX_COV:-diag}"
MIX_PROJ_HIDDEN="${MIX_PROJ_HIDDEN:-128}"
MIX_DROPOUT="${MIX_DROPOUT:-0.1}"
MIX_LOG_TRANSFORM="${MIX_LOG_TRANSFORM:-True}"

# ─── Derived paths ──────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="${SCRIPT_DIR}/rlhf_rw/main.py"

if [[ ! -f "$MAIN_PY" ]]; then
    echo "[ERROR] Cannot find rlhf_rw/main.py relative to $SCRIPT_DIR"
    echo "        Run this script from the repo root (personnel-main/)."
    exit 1
fi

# ─── VRAM monitor (background) ──────────────────────────────
VRAM_LOG="${SCRIPT_DIR}/vram_log.jsonl"
VRAM_THRESHOLD_GB=35

if python3 -c "import pynvml" 2>/dev/null; then
    python3 - <<'PYEOF' &
import time, json, os, datetime
try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    threshold = float(os.environ.get("VRAM_THRESHOLD_GB", 35)) * 1024**3
    logfile = os.environ.get("VRAM_LOG", "vram_log.jsonl")
    while True:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024**3
        record = {"ts": datetime.datetime.utcnow().isoformat(),
                  "used_gb": round(used_gb, 2), "total_gb": round(info.total/1024**3,2)}
        with open(logfile, "a") as f:
            f.write(json.dumps(record) + "\n")
        if info.used > threshold:
            print(f"[VRAM WARNING] {used_gb:.1f} GB used (>{os.environ.get('VRAM_THRESHOLD_GB','35')} GB threshold)", flush=True)
        time.sleep(60)
except Exception as e:
    print(f"[vram_monitor] disabled: {e}")
PYEOF
    export VRAM_LOG VRAM_THRESHOLD_GB
    echo "[vram_monitor] background monitor started (log: $VRAM_LOG, threshold: ${VRAM_THRESHOLD_GB} GB)"
else
    echo "[vram_monitor] pynvml not found – skipping background VRAM monitor"
    echo "               Install with:  pip install nvidia-ml-py3"
fi

# ─── Common args shared by all run modes ────────────────────
COMMON_ARGS=(
    -d "$DATASET"
    -m "$MODEL"
    --batch_size "$BATCH"
    --train_epochs "$EPOCHS"
    --learning_rate "$LR"
    --lr_scheduler_type "$LR_SCHED"
    --min_lr_ratio "$MIN_LR_RATIO"
    --weight_decay "$WEIGHT_DECAY"
    --seed "$SEED"
    --fp_dropout "$FP_DROPOUT"
    --gradient_acum_steps "$GRAD_ACCUM"
    --logging_steps "$LOG_STEPS"
    --max_length "$MAX_LEN"
    --use_lora True
    --use_quantization True
    --gradient_checkpointing True
    --mode train
)

MODE="${1:-mixture}"

echo "================================================================"
echo "  GazeReward run_modes.sh"
echo "  Mode     : $MODE"
echo "  Model    : $MODEL"
echo "  Dataset  : $DATASET"
echo "================================================================"

case "$MODE" in

  # ── MIXTURE (default): per-response GMM summary token ─────────
  # All 5 ET features → mixture_module fits K-component GMM per
  # response → flatten (π, μ, Σ) → projector → (1 token, hidden).
  # That token is prepended to the input sequence so the RM can
  # attend to a response-level reading-mode summary as a prior.
  mixture)
    echo "  ET model : fmv=2  |  features: all 5 (1,1,1,1,1)"
    echo "  Mixture  : K=$MIX_K, cov_type=$MIX_COV (per-response GMM)"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --fixations_model_version 2 \
        --features_used "1,1,1,1,1" \
        --concat True \
        --use_softprompt True \
        --use_mixture_token True \
        --mixture_K "$MIX_K" \
        --mixture_cov_type "$MIX_COV" \
        --mixture_proj_hidden "$MIX_PROJ_HIDDEN" \
        --mixture_dropout "$MIX_DROPOUT" \
        --mixture_log_transform "$MIX_LOG_TRANSFORM"
    ;;

  # ── MIXTURE_FULL: same as mixture but with full covariance ────
  # Captures inter-feature correlations in each Gaussian component.
  # Higher capacity, more sensitive to short responses.
  mixture_full)
    echo "  ET model : fmv=2  |  features: all 5 (1,1,1,1,1)"
    echo "  Mixture  : K=$MIX_K, cov_type=full (per-response GMM)"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --fixations_model_version 2 \
        --features_used "1,1,1,1,1" \
        --concat True \
        --use_softprompt True \
        --use_mixture_token True \
        --mixture_K "$MIX_K" \
        --mixture_cov_type full \
        --mixture_proj_hidden "$MIX_PROJ_HIDDEN" \
        --mixture_dropout "$MIX_DROPOUT" \
        --mixture_log_transform "$MIX_LOG_TRANSFORM"
    ;;

  # ── LOPEZ-CARDONA fcomb2.5 ───────────────────────────────────
  # fmv=2, all 5 features, no mixture token.
  # Direct replication of Table 3/4 best f_comb2.5 (GazeConcat).
  # This is the paper baseline that mixture mode aims to beat.
  lopez)
    echo "  ET model : fmv=2  |  features: all 5 – fcomb2.5 (1,1,1,1,1)"
    echo "  Mixture  : disabled  |  concat=True, use_softprompt=True"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --fixations_model_version 2 \
        --features_used "1,1,1,1,1" \
        --concat True \
        --use_softprompt True \
        --use_mixture_token False
    ;;

  # ── LOPEZ-CARDONA fcomb2.2 (TRT + FFD only) ─────────────────
  lopez_22)
    echo "  ET model : fmv=2  |  features: TRT+FFD – fcomb2.2 (0,1,0,1,0)"
    echo "  Mixture  : disabled"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --fixations_model_version 2 \
        --features_used "0,1,0,1,0" \
        --concat True \
        --use_softprompt True \
        --use_mixture_token False
    ;;

  # ── LOPEZ-CARDONA fcomb1 (TRT via fmv=1) ────────────────────
  lopez_1)
    echo "  ET model : fmv=1  |  features: TRT only (fcomb1)"
    echo "  Mixture  : disabled"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --fixations_model_version 1 \
        --features_used "1,0,0,0,0" \
        --concat True \
        --use_softprompt True \
        --use_mixture_token False
    ;;

  # ── BASELINE (no ET) ────────────────────────────────────────
  baseline)
    echo "  No ET features – pure reward model baseline"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --concat False \
        --use_softprompt False \
        --use_mixture_token False
    ;;

  # ── EVAL ────────────────────────────────────────────────────
  eval)
    if [[ -z "$CKPT_DIR" ]]; then
        echo "[ERROR] Set CKPT_DIR to the saved model directory."
        exit 1
    fi
    echo "  Evaluating checkpoint: $CKPT_DIR"
    python3 "$MAIN_PY" "${COMMON_ARGS[@]}" \
        --mode evaluate \
        --fixations_model_version 2 \
        --features_used "1,1,1,1,1" \
        --concat True \
        --use_softprompt True \
        --use_mixture_token True \
        --mixture_K "$MIX_K" \
        --mixture_cov_type "$MIX_COV"
    ;;

  *)
    echo "[ERROR] Unknown mode: '$MODE'"
    echo "Available modes: mixture | mixture_full | lopez | lopez_22 | lopez_1 | baseline | eval"
    exit 1
    ;;
esac

echo "================================================================"
echo "  Run complete.  Mode: $MODE"
if [[ -f "$VRAM_LOG" ]]; then
    PEAK=$(python3 -c "
import json
with open('$VRAM_LOG') as f:
    lines = [json.loads(l) for l in f if l.strip()]
if lines:
    peak = max(l['used_gb'] for l in lines)
    print(f'{peak:.2f} GB')
" 2>/dev/null || echo "n/a")
    echo "  Peak VRAM: $PEAK  (full log: $VRAM_LOG)"
fi
echo "================================================================"

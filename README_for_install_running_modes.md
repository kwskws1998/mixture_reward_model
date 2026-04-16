# AGD — Installation & Usage Guide

This repo extends the Lopez-Cardona et al. (ICLR 2025) GazeReward framework with
a **per-response Gaussian Mixture Model (GMM) summary token** that models the
ET-feature distribution of each response as a mixture of reading modes (skim /
normal / deep / regression).

Two experiment families are supported:

| Family      | What it does                                                                       |
|-------------|------------------------------------------------------------------------------------|
| **mixture** | 5 ET features + per-response GMM token prepended to the RM input sequence.         |
| **lopez**   | Direct Lopez-Cardona 2025 replication (no mixture token), with fcomb1/2.2/2.5.     |

---

## Table of contents

1. [System requirements](#1-system-requirements)
2. [Install — step by step](#2-install--step-by-step)
3. [ET prediction models (ET1 / ET2)](#3-et-prediction-models-et1--et2)
4. [First run: H1 sanity check](#4-first-run-h1-sanity-check)
5. [Training](#5-training)
6. [Evaluation](#6-evaluation)
7. [Hyperparameter tuning (Optuna)](#7-hyperparameter-tuning-optuna)
8. [Ablation & hyperparameter sweeps](#8-ablation--hyperparameter-sweeps)
9. [Environment variables](#9-environment-variables)
10. [Hyperparameter reference](#10-hyperparameter-reference)
11. [Output artifacts](#11-output-artifacts)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System requirements

- **OS:** Linux (tested on Ubuntu 22.04). macOS will run CPU-only paths.
- **GPU:** NVIDIA with ≥40GB VRAM for 7B/8B base models with QLoRA.
  Tested on A100 80GB and H100 80GB.
- **CUDA:** 12.1 (matches `torch==2.2.2` in `requirements.txt`).
- **Python:** 3.10+ (3.10–3.12 tested).
- **Disk:** ~50GB for HF model caches + LMDB fixation cache + checkpoints.

---

## 2. Install — step by step

### 2-a. Clone and enter repo

```bash
git clone <your-fork-url> agd && cd agd
```

### 2-b. Create an environment

```bash
# Option A: conda
conda create -n agd python=3.11 -y
conda activate agd

# Option B: venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2-c. One-shot install

```bash
bash install.sh
```

This installs `requirements.txt`, the `eyetrackpy` package needed by ET model 1,
prints quick-start commands, and checks that torch sees a CUDA GPU.

If `install.sh` fails, the manual equivalent is:

```bash
pip install -r requirements.txt
pip install git+https://github.com/asieduofeibea/eyetrackpy.git
```

### 2-d. Hugging Face login (required for gated models)

```bash
huggingface-cli login
# Paste your token. Needed for meta-llama/Meta-Llama-3-8B[-Instruct].
```

### 2-e. (Optional) Set environment variables

See [section 9](#9-environment-variables). The only one you *usually* need is
`LMDB_CACHE_PATH` if you want the fixation cache outside the repo.

---

## 3. ET prediction models (ET1 / ET2)

The framework supports two ET feature predictors. Both are frozen during RM
training.

### ET1 — Huang & Hollenstein (2023), 1 feature (TRT)

- Source: `eyetrackpy` package (installed by `install.sh`).
- Tokenizer: T5.
- Output: 1 feature (TRT) per token.
- Activated with `--fixations_model_version 1`.
- **No setup needed beyond `install.sh`.**

### ET2 — Li & Rudzicz (2021), 5 features

- Source: a RoBERTa checkpoint you've trained (or the default on HF).
- Tokenizer: RoBERTa.
- Output: 5 features per token — `[nFix, FFD, GPT, TRT, fixProp]`.
- Activated with `--fixations_model_version 2`.
- **Requires a checkpoint.**

Run this once:

```bash
python setup_et_models.py
```

This will:

1. Try to download `skboy/et_prediction_2` from HF Hub (default). The file
   name is `et_predictor2_seed123.safetensors`.
2. Set `ET2_CHECKPOINT_PATH` in your current shell for this session.

**If you have your own ET2 checkpoint**, point to it explicitly:

```bash
export ET2_CHECKPOINT_PATH=/path/to/your_et2.safetensors
# or .pt / .bin — all accepted
```

The checkpoint must match the `_RobertaRegressionModel` layout in
`et2_wrapper.py` (RoBERTa base + `Linear(768, 5)` decoder head).

To verify ET2 loads:

```bash
python -c "
from et2_wrapper import FixationsPredictor_2
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
FixationsPredictor_2(modelTokenizer=tok, remap=False)
print('ET2 loaded OK')
"
```

---

## 4. First run: H1 sanity check

**Before training anything, verify the mixture hypothesis.** This script fits a
GMM with K=1..6 to each of a sample of OASST responses' 5-feature distributions
and reports how many responses prefer K≥2 by BIC.

```bash
python scripts/h1_bic_sanity_check.py \
    --n_responses 100 \
    --k_max 6 \
    --cov_type full \
    --out_dir bic_results
```

**Runtime:** ~10–30 min depending on GPU (one ET2 forward per sampled response).

**Outputs:**

- `bic_results/summary.json` — `{"pct_responses_with_K_ge_2": X, ...}`
- `bic_results/bic_curves.png` — BIC curves + best-K histogram
- `bic_results/bic_matrix.npy` — raw BIC matrix

**Decision rule:**

| `pct_responses_with_K_ge_2` | Verdict                                                        |
|-----------------------------|----------------------------------------------------------------|
| ≥ 70 %                      | Mixture hypothesis supported — proceed with method.            |
| 40–70 %                     | Weakly supported — frame as "often multimodal", K=2 default.   |
| < 40 %                      | Single Gaussian suffices — reconsider paper direction.         |

---

## 5. Training

All training goes through `rlhf_rw/main.py`. `run_modes.sh` is a convenience
wrapper with preset CLI flags.

### 5-a. Quickstart

```bash
# First: verify mixture hypothesis (section 4)
python scripts/h1_bic_sanity_check.py

# Default mixture run (K=3, diagonal covariance, all 5 ET features)
bash run_modes.sh mixture
```

### 5-b. All `run_modes.sh` presets

| Command                            | fmv | features       | mixture token | notes                                |
|------------------------------------|-----|----------------|---------------|--------------------------------------|
| `bash run_modes.sh mixture`        | 2   | `1,1,1,1,1`    | **yes**, K=3, diag | Default experimental mode.          |
| `bash run_modes.sh mixture_full`   | 2   | `1,1,1,1,1`    | **yes**, K=3, full | Full covariance variant.            |
| `bash run_modes.sh lopez`          | 2   | `1,1,1,1,1`    | no            | fcomb2.5 baseline (paper replication). |
| `bash run_modes.sh lopez_22`       | 2   | `0,1,0,1,0`    | no            | fcomb2.2 — TRT + FFD only.          |
| `bash run_modes.sh lopez_1`        | 1   | `1,0,0,0,0`    | no            | fcomb1 — TRT via ET1.                |
| `bash run_modes.sh baseline`       | 1   | `1,0,0,0,0`    | no            | No ET, pure RM baseline.            |
| `bash run_modes.sh eval`           | 2   | `1,1,1,1,1`    | yes           | Eval mode (set `CKPT_DIR` env var). |

### 5-c. Environment overrides for `run_modes.sh`

All env vars have defaults — override any by exporting or prefixing:

```bash
# Dataset & model
DATASET=nvidia/HelpSteer2 MODEL=meta-llama/Meta-Llama-3-8B \
    bash run_modes.sh mixture

# Training
BATCH=16 EPOCHS=3 LR=1e-5 bash run_modes.sh mixture

# Mixture-specific
MIX_K=4 MIX_COV=full MIX_DROPOUT=0.2 bash run_modes.sh mixture

# Multi-seed
SEED=123 bash run_modes.sh mixture
```

Full env-var list: see top of `run_modes.sh`.

### 5-d. Running `main.py` directly

If you need a flag combination `run_modes.sh` doesn't cover, call `main.py`
directly. Example — mixture + fcomb2.2 (TRT + FFD only):

```bash
python rlhf_rw/main.py \
    -d OpenAssistant/oasst1 \
    -m meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 8 \
    --train_epochs 2 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr_ratio 0.7 \
    --weight_decay 0.1 \
    --fp_dropout 0.1,0.3 \
    --use_lora True \
    --use_quantization True \
    --gradient_checkpointing True \
    --mode train \
    --fixations_model_version 2 \
    --features_used "0,1,0,1,0" \
    --concat True \
    --use_softprompt True \
    --use_mixture_token True \
    --mixture_K 3 \
    --mixture_cov_type diag \
    --mixture_proj_hidden 128 \
    --mixture_dropout 0.1 \
    --mixture_log_transform True \
    --seed 42
```

### 5-e. Feature-combo reference

The `features_used` vector is `[nFix, FFD, GPT, TRT, fixProp]` (5 binary flags).
Paper naming convention:

| Name     | fmv | `features_used`   | features                    |
|----------|-----|-------------------|-----------------------------|
| fcomb1   | 1   | `1,0,0,0,0`       | TRT (via ET1)              |
| fcomb2.1 | 2   | `0,0,0,1,0`       | TRT only (via ET2)         |
| fcomb2.2 | 2   | `0,1,0,1,0`       | FFD + TRT                  |
| fcomb2.5 | 2   | `1,1,1,1,1`       | all 5 (ET2 default)        |

(For `fmv=1`, `features_used` has only 1 meaningful slot — the first — since
ET1 only produces TRT. The rest are ignored.)

---

## 6. Evaluation

### 6-a. OASST / HelpSteer test split (automatic)

Evaluated automatically at the end of training. Result written to
`<ckpt_dir>/results_dataset_test.json` → `eval_accuracy`.

### 6-b. Eval from a checkpoint

```bash
CKPT_DIR=./models_save/<your-run-dir> bash run_modes.sh eval
```

### 6-c. RewardBench evaluation

Either turn it on at training time:

```bash
bash run_modes.sh mixture   # add: --run_rewardbench True
# (or pass directly to main.py)
```

Or run standalone from a checkpoint:

```bash
python run_rewardbench.py \
    --ckpt_dir ./models_save/<your-run-dir> \
    --batch_size 8 \
    --max_length 10000
```

Outputs RewardBench per-subset accuracy to stdout and to
`<ckpt_dir>/rewardbench_results.json`.

---

## 7. Hyperparameter tuning (Optuna)

`optuna_tune.py` runs Optuna + HyperbandPruner over 4 predefined modes.

### 7-a. Available modes

| `--mode`          | Preset                                                          |
|-------------------|-----------------------------------------------------------------|
| `mixture`         | fcomb2.5 + mixture token (diag cov). Also tunes `mixture_K`, `mixture_dropout`. |
| `mixture_full`    | fcomb2.5 + mixture token (full cov). Also tunes `mixture_K`, `mixture_dropout`. |
| `lopez`           | fcomb2.5 baseline, no mixture.                                  |
| `baseline`        | No ET.                                                          |

### 7-b. Basic usage

```bash
# Mixture mode, 20 trials, in-memory study
python optuna_tune.py --mode mixture --n_trials 20

# Persistent study so it can resume
python optuna_tune.py --mode mixture --n_trials 20 \
    --storage sqlite:///optuna_study.db \
    --study_name mixture_study

# Lopez-Cardona baseline tuning
python optuna_tune.py --mode lopez --n_trials 20
```

### 7-c. Hyperband settings (built-in)

- `min_resource = 1` epoch
- `reduction_factor = 3`
- Bracket 0 runs 9 epochs, bracket 1 runs 3, bracket 2 runs 1.

### 7-d. What's tuned per trial

Always:
- `learning_rate` ∈ [1e-6, 1e-4] (log)
- `batch_size` ∈ {8, 16, 32}
- `weight_decay` ∈ [1e-4, 0.3] (log)
- `lr_scheduler_type` ∈ {constant_with_warmup, cosine_with_min_lr, linear}
- `min_lr_ratio` ∈ [0.5, 0.9]
- `fp_dropout_1` ∈ [0.0, 0.3], `fp_dropout_2` ∈ [0.1, 0.5]

When mode in {`mixture`, `mixture_full`}:
- `mixture_K` ∈ [2, 6]
- `mixture_dropout` ∈ [0.0, 0.3]

---

## 8. Ablation & hyperparameter sweeps

`ablation_sweep.py` runs systematic sweeps by invoking `main.py` as subprocesses.
Use this for paper tables (feature ablation) and for ad-hoc hyperparameter grids.

### 8-a. Feature ablation — paper Table 3/4 replication

Run the same mode across all feature combos:

```bash
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode mixture \
    --train_epochs 2 \
    --out_dir ./sweep_results/feature_ablation_mixture

# Baseline comparison: same across Lopez-Cardona modes
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode lopez \
    --out_dir ./sweep_results/feature_ablation_lopez
```

Runs 4 combos: `fcomb1`, `fcomb2.1`, `fcomb2.2`, `fcomb2.5`.

Select subset:

```bash
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode mixture \
    --feature_combos fcomb1,fcomb2.5 \
    --out_dir ./sweep_results/ablation_subset
```

### 8-b. Mixture hyperparameter grid

```bash
python ablation_sweep.py \
    --sweep mixture_hparam \
    --K_grid 2,3,4,5 \
    --cov_grid diag,full \
    --dropout_grid 0.05,0.1,0.2 \
    --proj_hidden_grid 64,128,256 \
    --out_dir ./sweep_results/mixture_hparam
```

Default grid = 4 × 2 × 3 × 3 = **72 runs**. Trim aggressively for paper figures
(e.g. `--K_grid 2,3,4,5 --cov_grid diag --dropout_grid 0.1 --proj_hidden_grid 128`
= 4 runs for the K sweep).

### 8-c. Seed stability (for confidence intervals)

```bash
python ablation_sweep.py \
    --sweep seeds \
    --base_mode mixture \
    --feature_combo fcomb2.5 \
    --mixture_K 3 --mixture_cov_type diag \
    --seeds 42,123,2024 \
    --out_dir ./sweep_results/seed_stability
```

### 8-d. Useful sweep flags

- `--dry_run` — prints commands without executing (verify before spending GPU).
- `--continue_on_fail` — keeps going if one run crashes.
- `--run_rewardbench` — appends `--run_rewardbench True` to every sub-run.

### 8-e. Sweep outputs

```
sweep_results/<sweep_name>/
├── sweep_manifest.json          # all runs + their configs
└── <tag>/                       # one dir per run
    ├── config.json              # exact config for this run
    └── run.log                  # full stdout/stderr of main.py
```

The actual checkpoints land wherever `main.py` writes them (`./models_save/...`);
`config.json` + `run.log` in the sweep dir are the sweep-level index.

---

## 9. Environment variables

| Var                       | Default               | Purpose                                    |
|---------------------------|-----------------------|--------------------------------------------|
| `ET2_CHECKPOINT_PATH`     | auto-download from HF | Path to ET2 `.safetensors` / `.pt` / `.bin`. |
| `LMDB_CACHE_PATH`         | `./buffer_train.lmdb` | Where fixation predictions are cached.     |
| `HF_HUB_OFFLINE`          | unset                 | Set to `1` to prevent HF hub network calls.|
| `CUDA_VISIBLE_DEVICES`    | all                   | Pin training to specific GPUs.             |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` (set by optuna_tune) | Reduces fragmentation. |
| `VRAM_THRESHOLD_GB`       | 35                    | `run_modes.sh` VRAM monitor warning level. |
| `VRAM_LOG`                | `vram_log.jsonl`      | `run_modes.sh` VRAM log file.              |

---

## 10. Hyperparameter reference

### 10-a. RM training hyperparameters (common)

| Flag                       | Default                    | Notes                                              |
|----------------------------|----------------------------|----------------------------------------------------|
| `--learning_rate`          | `5e-5`                     | Paper used `5e-5`. Range in tuning: 1e-6..1e-4.   |
| `--batch_size`             | `8`                        | Paper grid: {8, 16, 32}.                          |
| `--train_epochs`           | `2`                        | Paper default.                                     |
| `--lr_scheduler_type`      | `constant_with_warmup`     | `cosine_with_min_lr` recommended.                  |
| `--min_lr_ratio`           | `0.7`                      | Cosine floor fraction.                             |
| `--weight_decay`           | `0.1`                      | AdamW weight decay.                                |
| `--fp_dropout`             | `0.1,0.3`                  | `(p_1, p_2)` in the ET-features projector MLP.    |
| `--gradient_acum_steps`    | `1`                        | Raise to ≥4 if OOM.                                |
| `--max_length`             | `10000`                    | Sequence cap (pre-tokenization).                   |
| `--noise_factor`           | `0.0`                      | Fixation-noise at eval (kept from paper code).     |
| `--seed`                   | `42`                       |                                                    |

### 10-b. Architecture flags

| Flag                            | Default | Notes                                              |
|---------------------------------|---------|----------------------------------------------------|
| `-m / --model_name`             | llama3  | Any `transformers` causal-seq-cls-capable model.  |
| `--fixations_model_version`     | `1`     | 1 = ET1 (T5 + BiLSTM), 2 = ET2 (RoBERTa).         |
| `--features_used`               | `1,1,1,1,1` | 5 binary flags: `[nFix, FFD, GPT, TRT, fixProp]`. |
| `--concat`                      | `False` | `True` → GazeConcat. `False` → GazeAdd.           |
| `--use_softprompt`              | `False` | Wraps ET embeds with `<eye/> ... </eye>` tokens.  |
| `--use_lora`                    | `True`  | QLoRA on the backbone.                             |
| `--use_quantization`            | `True`  | 4-bit NF4 quant.                                   |

### 10-c. Mixture-token flags

| Flag                       | Default      | Notes                                                           |
|----------------------------|--------------|-----------------------------------------------------------------|
| `--use_mixture_token`      | `False`      | Master switch.                                                  |
| `--mixture_K`              | `3`          | # GMM components. Tune in {2, 3, 4, 5, 6}.                      |
| `--mixture_cov_type`       | `diag`       | `diag` / `full` / `tied` / `spherical`. `diag` is safest.       |
| `--mixture_proj_hidden`    | `128`        | Hidden dim of the (π,μ,Σ) → hidden_size projector MLP.          |
| `--mixture_dropout`        | `0.1`        | Dropout in the projector MLP.                                   |
| `--mixture_log_transform`  | `True`       | `log1p(feature)` before GMM fit (ET features are right-skewed). |

**Covariance descriptor sizes** (so you can judge parameter count):

| cov_type     | descriptor size with K=3, F=5 |
|--------------|-------------------------------|
| `spherical`  | 21                            |
| `diag`       | 33                            |
| `tied`       | 33                            |
| `full`       | 63                            |

The projector MLP size is roughly `(desc + 1) * proj_hidden + proj_hidden * hidden_size`
parameters. For `diag, K=3, proj_hidden=128, hidden_size=4096` this is ~530K
parameters — negligible next to the backbone.

### 10-d. Sizing guidance (empirical)

- **Short responses (n < 40 real tokens)**: `mixture_cov_type=diag` is strongly
  preferred — full covariance often falls back to reduced K (see
  `mixture_module._fit_one`).
- **Many responses failing validity check**: if the `validity` flag is 0 for a
  big fraction, K is probably too large for the sample sizes you have — drop K.
- **Mode collapse suspicion**: always set `--mixture_log_transform True`.
  Without it the right-skewed raw features pull all components toward the
  origin.

---

## 11. Output artifacts

For a single training run:

```
models_save/<auto-named-run-dir>/
├── adapter_config.json                  # HF peft adapter
├── adapter_model.safetensors            # LoRA weights
├── fixations_projector_state_dict.bin   # ET-features MLP projector
├── layer_norm_state_dic.bin             # post-projector LayerNorm
├── mixture_module.bin                   # (if use_mixture_token) projector+descriptor
├── args.json                            # every CLI flag used
├── results_dataset_test.json            # final test accuracy on dataset split
└── rewardbench_results.json             # (if --run_rewardbench True)
```

For sweeps:

```
sweep_results/<sweep>/
├── sweep_manifest.json
└── <tag>/config.json + run.log
```

---

## 12. Troubleshooting

**`ET2 checkpoint not found`.** Set `ET2_CHECKPOINT_PATH` or run
`python setup_et_models.py` to auto-download. Verify with the 3-line snippet
in [section 3](#3-et-prediction-models-et1--et2).

**CUDA OOM during training.** Try in order:

1. `GRAD_ACCUM=4 bash run_modes.sh mixture` (effective batch 32 at physical 8).
2. Drop `BATCH` to 4.
3. Drop `MAX_LEN` to 4096 if your dataset allows.

**LMDB cache errors ("Environment mapsize reached").** Set
`LMDB_CACHE_PATH` to a fresh directory or delete the existing `.lmdb`.

**`mixture_module.bin not found` on eval from old checkpoint.** The run was
trained *before* you enabled mixture. Either disable `--use_mixture_token`
for eval, or retrain.

**BIC sanity check hangs on ET2 forward.** You're on CPU. Run with CUDA, or
reduce `--n_responses` for a CPU-only smoke test.

**`NaN` in mixture descriptor.** Indicates a response with all-zero features
(every token was a subword-non-first). The module already zero-pads and sets
the validity flag — check the log for `validity=0` counts. If most of your
batch is hitting this, your upstream tokenizer is doing something unusual.

**OASST filter returns too few responses.** Some OASST dumps tag messages
differently. Pass `--min_tokens 10` (instead of default 20) to the BIC sanity
check.

---

## Citation

This codebase extends:

- Lopez-Cardona, Segura, Karatzoglou, Abadal, Arapakis (ICLR 2025).
  *Seeing Eye to AI: Human Alignment via Gaze-Based Response Rewards for
  Large Language Models*. [arXiv:2410.01532](https://arxiv.org/abs/2410.01532)

ET prediction models:

- Li & Rudzicz (2021), *TorontoCL at CMCL 2021 Shared Task* — ET2.
- Huang & Hollenstein (2023), *Longer Fixations, More Computation* — ET1.

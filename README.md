# AGD — Mixture-Augmented Gaze Reward Modelling

Extension of Lopez-Cardona et al. (ICLR 2025) that models each response's
eye-tracking feature distribution as a **K-component Gaussian mixture** and
prepends its (π, μ, Σ) summary as a single token to the RM input sequence.

Thesis: reading is a mixture process (skim / normal / deep / regression),
so per-token ET features are samples from a K-modal distribution — not
points in flat ℝ⁵. Modelling that explicitly is the point of this repo.

<p align="center">
  <img src="assets/pipeline.png" alt="Overview" width="80%">
</p>

---

## Contents

1. [Install](#1-install)
2. [ET predictors (ET1 / ET2)](#2-et-predictors-et1--et2)
3. [H1 sanity check (run first)](#3-h1-sanity-check-run-first)
4. [Training](#4-training)
5. [Evaluation](#5-evaluation)
6. [Hyperparameter tuning (Optuna)](#6-hyperparameter-tuning-optuna)
7. [Ablation & sweeps](#7-ablation--sweeps)
8. [Hyperparameter reference](#8-hyperparameter-reference)
9. [Environment variables](#9-environment-variables)
10. [Output artifacts](#10-output-artifacts)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Install

**Requirements:** Linux, NVIDIA GPU with ≥40GB VRAM (A100-40GB minimum,
tested on A100-80GB/H100-80GB), CUDA 12.x, Python 3.10+.

```bash
git clone <your-fork> agd && cd agd

# Option A: fresh environment
conda create -n agd python=3.11 -y && conda activate agd
bash install.sh

# Option B: existing torch/CUDA environment (auto-detected, won't reinstall)
bash install.sh

# HuggingFace auth (required for meta-llama/* gated models)
huggingface-cli login
```

`install.sh` behaviour:
- **Auto-detects existing working torch+CUDA** and skips the pinned
  `torch==2.2.2` / `nvidia-*` stack (avoids multi-GB downgrade).
- Pass `FORCE_TORCH=1 bash install.sh` to force-reinstall the paper's pinned stack.
- Pass `SKIP_TORCH=1 bash install.sh` to explicitly skip torch.
- Pass `SKIP_ET2_DL=1 bash install.sh` if you want to bring your own ET2 checkpoint.

---

## 2. ET predictors (ET1 / ET2)

**ET1** (Huang & Hollenstein 2023, 1 feature = TRT, T5 tokenizer):
- Needs weights from the `SelectiveCacheForLM` repo. One-shot:
  ```bash
  python setup_et_models.py --skip-install
  ```
  (Pass `--skip-install` because `install.sh` already did the pip work.)

**ET2** (Li & Rudzicz 2021, 5 features, RoBERTa tokenizer):
- `install.sh` pre-downloads the default checkpoint from
  `skboy/et_prediction_2` into the HuggingFace cache.
- `et2_wrapper.py` will auto-find it at first use; **no extra setup needed.**
- If you want your own ET2 checkpoint, export before training:
  ```bash
  export ET2_CHECKPOINT_PATH=/path/to/your_et2.safetensors
  ```

Verify ET2 loads:
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2'); tok.pad_token = tok.eos_token
from et2_wrapper import FixationsPredictor_2
FixationsPredictor_2(modelTokenizer=tok, remap=False)
print('ET2 OK')
"
```

---

## 3. H1 sanity check (run first)

Before any training, verify the mixture hypothesis. Fits per-response GMMs
with K=1..6 on a sample of OASST responses and reports BIC-optimal K:

```bash
python scripts/h1_bic_sanity_check.py --n_responses 100 --k_max 6
```

Output → `bic_results/summary.json` + `bic_results/bic_curves.png`.

**Decision rule:**

| `pct_responses_with_K_ge_2` | Verdict                                                    |
|-----------------------------|------------------------------------------------------------|
| ≥ 70 %                      | H1 supported — proceed.                                    |
| 40–70 %                     | Weakly supported — frame as "often multimodal".            |
| < 40 %                      | Reconsider paper framing; single Gaussian may suffice.     |

---

## 4. Training

All training goes through `rlhf_rw/main.py` directly. No bash wrapper —
just copy the command template.

### 4-a. Mixture mode (my method, default)

```bash
python rlhf_rw/main.py \
    -d OpenAssistant/oasst1 \
    -m meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 4 \
    --gradient_acum_steps 2 \
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
    --features_used 1,1,1,1,1 \
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

On 40GB: `--batch_size 4 --gradient_acum_steps 2` is safe (effective batch 8,
matches paper). On 80GB you can use `--batch_size 8 --gradient_acum_steps 1`.

### 4-b. Lopez-Cardona baselines (paper replication)

Same feature-combo options as the paper (`features_used` vector is
`[nFix, FFD, GPT, TRT, fixProp]`):

| Combo     | `-fmv` | `features_used` | Features                    |
|-----------|--------|-----------------|-----------------------------|
| `fcomb1`  | 1      | `1,0,0,0,0`     | TRT via ET1                 |
| `fcomb2.1`| 2      | `0,0,0,1,0`     | TRT via ET2                 |
| `fcomb2.2`| 2      | `0,1,0,1,0`     | FFD + TRT                   |
| `fcomb2.5`| 2      | `1,1,1,1,1`     | all 5                       |

**fcomb2.5 (paper best):**
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
    --features_used 1,1,1,1,1 \
    --concat True \
    --use_softprompt True \
    --use_mixture_token False \
    --seed 42
```

For other Lopez-Cardona combos, just swap `--fixations_model_version` and
`--features_used` per the table above.

### 4-c. Pure baseline (no ET)

```bash
python rlhf_rw/main.py \
    -d OpenAssistant/oasst1 \
    -m meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 8 \
    --train_epochs 2 \
    --learning_rate 5e-5 \
    --use_lora True \
    --use_quantization True \
    --mode train \
    --fixations_model_version 1 \
    --features_used 1,0,0,0,0 \
    --concat False \
    --use_softprompt False \
    --use_mixture_token False \
    --seed 42
```

### 4-d. Disabling wandb

wandb interactive login is annoying and not needed for paper runs.
Disable it:

```bash
export WANDB_MODE=disabled
# then run main.py as above
```

Or one-shot:
```bash
WANDB_MODE=disabled python rlhf_rw/main.py ...
```

---

## 5. Evaluation

**Automatic at end of training** — writes `results_dataset_test.json` into the
checkpoint directory with `eval_accuracy`.

**Eval from a saved checkpoint:**
```bash
python rlhf_rw/main.py \
    -d OpenAssistant/oasst1 \
    -m meta-llama/Meta-Llama-3-8B-Instruct \
    --mode evaluate \
    --load_local_folder_name ./models_save/<your-run-dir> \
    --fixations_model_version 2 \
    --features_used 1,1,1,1,1 \
    --concat True --use_softprompt True \
    --use_mixture_token True --mixture_K 3 --mixture_cov_type diag
```

**RewardBench evaluation** (paper's external benchmark):
```bash
python run_rewardbench.py \
    --ckpt_dir ./models_save/<your-run-dir> \
    --batch_size 8 \
    --max_length 10000
```

Or add `--run_rewardbench True` to `main.py` to run it automatically after training.

---

## 6. Hyperparameter tuning (Optuna)

`optuna_tune.py` runs Optuna + HyperbandPruner over predefined modes.

```bash
# Mixture-token tuning (suggests mixture_K ∈ [2,6], mixture_dropout ∈ [0,0.3]
# on top of the usual LR/batch/scheduler)
python optuna_tune.py --mode mixture --n_trials 20

# Lopez-Cardona baseline tuning (no mixture)
python optuna_tune.py --mode lopez --n_trials 20

# Persistent study (resumable across restarts)
python optuna_tune.py --mode mixture --n_trials 20 \
    --storage sqlite:///optuna_study.db \
    --study_name mixture_study
```

Available `--mode` values: `mixture` / `mixture_full` / `lopez` / `baseline`.

Every trial also tunes: `learning_rate ∈ [1e-6, 1e-4]` (log),
`batch_size ∈ {8,16,32}`, `weight_decay ∈ [1e-4, 0.3]` (log),
`lr_scheduler_type`, `min_lr_ratio ∈ [0.5, 0.9]`, `fp_dropout_{1,2}`.

Hyperband pruner: `min_resource=1 epoch`, `reduction_factor=3`.

---

## 7. Ablation & sweeps

`ablation_sweep.py` runs systematic sweeps by launching `main.py` as
subprocesses. Use for paper tables.

### 7-a. Feature ablation (paper Table 3/4 replication)

```bash
# Run fcomb1, fcomb2.1, fcomb2.2, fcomb2.5 all under mixture mode
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode mixture \
    --out_dir ./sweep_results/mixture_feature_ablation

# Same for Lopez-Cardona baseline
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode lopez \
    --out_dir ./sweep_results/lopez_feature_ablation

# Subset:
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode mixture \
    --feature_combos fcomb1,fcomb2.5 \
    --out_dir ./sweep_results/ablation_subset
```

### 7-b. Mixture hyperparameter grid

```bash
python ablation_sweep.py \
    --sweep mixture_hparam \
    --K_grid 2,3,4,5 \
    --cov_grid diag,full \
    --dropout_grid 0.05,0.1,0.2 \
    --proj_hidden_grid 64,128,256 \
    --out_dir ./sweep_results/mixture_hparam
```

Trim for paper figures, e.g. just K sweep:
```bash
python ablation_sweep.py \
    --sweep mixture_hparam \
    --K_grid 2,3,4,5 \
    --cov_grid diag --dropout_grid 0.1 --proj_hidden_grid 128 \
    --out_dir ./sweep_results/K_sweep
```

### 7-c. Seed stability (confidence intervals)

```bash
python ablation_sweep.py \
    --sweep seeds \
    --base_mode mixture \
    --feature_combo fcomb2.5 \
    --mixture_K 3 --mixture_cov_type diag \
    --seeds 42,123,2024 \
    --out_dir ./sweep_results/seeds
```

### 7-d. Useful sweep flags

- `--dry_run` — print commands without executing (verify before spending GPU).
- `--continue_on_fail` — keep going if a run crashes.
- `--run_rewardbench` — add RewardBench eval to every sub-run.

Sweep output layout:
```
sweep_results/<sweep_name>/
├── sweep_manifest.json       # all planned runs + configs
└── <tag>/
    ├── config.json           # exact config for this run
    └── run.log               # full stdout/stderr
```

---

## 8. Hyperparameter reference

### 8-a. RM training

| Flag                    | Default                | Notes                                          |
|-------------------------|------------------------|------------------------------------------------|
| `--learning_rate`       | `5e-5`                 | Paper default. Tuning range: 1e-6..1e-4 log.  |
| `--batch_size`          | `8`                    | On 40GB: use 4 + `--gradient_acum_steps 2`.   |
| `--train_epochs`        | `2`                    | Paper default.                                 |
| `--lr_scheduler_type`   | `constant_with_warmup` | Paper best: `cosine_with_min_lr`.             |
| `--min_lr_ratio`        | `0.7`                  | Cosine floor fraction.                         |
| `--weight_decay`        | `0.1`                  | AdamW.                                         |
| `--fp_dropout`          | `0.1,0.3`              | `(p_1, p_2)` in features projector MLP.        |
| `--gradient_acum_steps` | `1`                    | Raise to 2–4 on 40GB or for OOM.              |
| `--max_length`          | `10000`                | Sequence length cap.                           |

### 8-b. Architecture / ET

| Flag                           | Default   | Notes                                           |
|--------------------------------|-----------|-------------------------------------------------|
| `-m / --model_name`            | llama3    | Any HF causal-seq-cls model.                    |
| `--fixations_model_version`    | `1`       | 1 = ET1 (TRT only), 2 = ET2 (5 features).       |
| `--features_used`              | `1,1,1,1,1` | `[nFix, FFD, GPT, TRT, fixProp]` binary flags. |
| `--concat`                     | `False`   | `True` = GazeConcat, `False` = GazeAdd.         |
| `--use_softprompt`             | `False`   | Wraps ET embeds with `<eye/>...</eye>`.         |
| `--use_lora`                   | `True`    | QLoRA on backbone (r=8, alpha=32, dropout=0.1). |
| `--use_quantization`           | `True`    | 4-bit NF4 + bf16 compute (paper setup).         |

### 8-c. Mixture-token flags

| Flag                      | Default | Notes                                                          |
|---------------------------|---------|----------------------------------------------------------------|
| `--use_mixture_token`     | `False` | Master switch.                                                 |
| `--mixture_K`             | `3`     | # GMM components. Tune in {2,3,4,5,6}.                         |
| `--mixture_cov_type`      | `diag`  | `diag` / `full` / `tied` / `spherical`. `diag` safest.          |
| `--mixture_proj_hidden`   | `128`   | Hidden dim of (π,μ,Σ) → hidden_size projector MLP.             |
| `--mixture_dropout`       | `0.1`   | Dropout in projector MLP.                                      |
| `--mixture_log_transform` | `True`  | `log1p` before GMM fit (right-skewed features).                |

Descriptor sizes (K=3, F=5):

| cov_type    | descriptor size |
|-------------|----------------|
| spherical   | 21             |
| diag        | 33             |
| tied        | 33             |
| full        | 63             |

### 8-d. Sizing guidance

- **Short responses (n<40 tokens):** prefer `diag` covariance. `full` often
  falls back to reduced K automatically.
- **Many responses failing validity flag:** K is too large for sample sizes;
  drop K.
- **Mode collapse suspicion:** always keep `--mixture_log_transform True`.

---

## 9. Environment variables

| Var                       | Default               | Purpose                                    |
|---------------------------|-----------------------|--------------------------------------------|
| `ET2_CHECKPOINT_PATH`     | auto (HF cache)       | Path to ET2 checkpoint.                    |
| `LMDB_CACHE_PATH`         | `./buffer_train.lmdb` | Fixation prediction cache.                 |
| `WANDB_MODE`              | online                | Set `disabled` to skip wandb entirely.     |
| `HF_HUB_OFFLINE`          | unset                 | `1` to prevent HF hub network calls.       |
| `CUDA_VISIBLE_DEVICES`    | all                   | Pin training to specific GPUs.             |

---

## 10. Output artifacts

```
models_save/<auto-named-run-dir>/
├── adapter_config.json                # HF peft
├── adapter_model.safetensors          # LoRA weights
├── fixations_projector_state_dict.bin # ET-features MLP
├── layer_norm_state_dic.bin           # post-projector LN
├── mixture_module.bin                 # (if mixture) projector state_dict
├── args.json                          # every CLI flag used
├── results_dataset_test.json          # final test accuracy
└── rewardbench_results.json           # (if --run_rewardbench True)
```

---

## 11. Troubleshooting

**CUDA OOM on 40GB.** Drop `--batch_size 4 --gradient_acum_steps 2`. Or try
`--concat False --use_softprompt False` (GazeAdd, smaller sequences).

**`IndexError: boolean index did not match indexed array`.** Fixed in current
version of `mixture_module.py` — length mismatches between `fixations` and
`fixations_attention_mask` are now cropped to their common prefix.

**wandb key rejected.** Your key is more than 40 chars = you pasted something
other than the hex API key. Easiest fix: `export WANDB_MODE=disabled`. For
real wandb, grab the exact 40-char key from https://wandb.ai/authorize.

**ET2 `FileNotFoundError`.** `install.sh` already downloaded ET2 to the HF
cache. If `setup_et_models.py` still fails, just run `--skip-install` or
ignore it — `et2_wrapper.py` auto-downloads at first use.

**LMDB "Environment mapsize reached".** Delete the existing `.lmdb` or point
`LMDB_CACHE_PATH` to a fresh directory.

**`mixture_module.bin not found` when evaluating an old checkpoint.** The run
was trained before mixture was enabled. Either disable mixture for eval or
retrain.

**BIC sanity check hangs.** You're on CPU. Use CUDA, or reduce `--n_responses`.

---

## Citation

Upstream paper:
```bibtex
@inproceedings{Lopez-Cardona2025Seeing,
  title  = {Seeing Eye to AI: Human Alignment via Gaze-Based Response Rewards for Large Language Models},
  author = {Lopez-Cardona, Angela and Segura, Carlos and Karatzoglou, Alexandros and Abadal, Sergi and Arapakis, Ioannis},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year   = {2025},
  url    = {https://openreview.net/forum?id=uZgK0tcPqd}
}
```

ET predictors: Li & Rudzicz (2021) (ET2) / Huang & Hollenstein (2023) (ET1).

## License

LGPLv3 — see [LICENSE](./LICENSE).

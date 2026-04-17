# AGD — Mixture-Augmented Gaze Reward Modelling

Extension of Lopez-Cardona et al. (ICLR 2025) that models each response's
eye-tracking feature distribution as a **K-component Gaussian mixture** and
prepends its (π, μ, Σ) summary as a single token to the RM input sequence.

<p align="center">
  <img src="assets/pipeline.png" alt="Overview" width="80%">
</p>

---

## 1. Install

```bash
git clone <your-fork> agd && cd agd
bash install.sh
huggingface-cli login
python setup_et_models.py --skip-install
```

`install.sh` auto-detects existing torch+CUDA and won't reinstall it.
Pass `FORCE_TORCH=1` to force the paper's pinned `torch==2.2.2` stack.

---

## 2. wandb setup

```bash
pip install -U wandb    # upgrade to latest (fixes 86-char API key issue)
wandb login             # paste your 40-char API key from https://wandb.ai/authorize
```

To disable wandb entirely (e.g. quick debugging):
```bash
export WANDB_MODE=disabled
```

---

## 3. ET predictors (ET1 / ET2)

**ET1** (Huang & Hollenstein 2023, 1 feature = TRT):
- Installed by `setup_et_models.py`. No extra setup.

**ET2** (Li & Rudzicz 2021, 5 features = nFix, FFD, GPT, TRT, fixProp):
- `install.sh` pre-downloads the default checkpoint from HuggingFace.
- `et2_wrapper.py` auto-finds it. No extra setup.
- Custom checkpoint: `export ET2_CHECKPOINT_PATH=/path/to/your_et2.safetensors`

Verify:
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

## 4. Key concepts: K and F

**F = 5**: the number of ET features (`[nFix, FFD, GPT, TRT, fixProp]`). Fixed
by the ET2 model — you don't change this.

**K**: the number of Gaussian mixture components. This is the main
hyperparameter you tune. K=3 means "this response's reading pattern is a
mixture of 3 modes (e.g. skim / normal / deep)."

From the H1 sanity check on OASST1 (100 responses): mean optimal K = 2.76,
so **K=3 is the default starting point**. But K=2,3,4,5 should all be
compared — see [section 9 (ablation)](#9-ablation--sweeps).

---

## 5. H1 sanity check (run first)

Verifies that per-response ET feature distributions are multimodal (K≥2
preferred over K=1 by BIC). **Run before any training.**

```bash
# Quick (100 responses, ~30 sec on GPU)
python scripts/h1_bic_sanity_check.py --n_responses 100 --k_max 6

# Full dataset (all ~20K eligible OASST1 responses, ~1-2 hours)
python scripts/h1_bic_sanity_check.py --n_responses 0 --k_max 6

# Custom settings
python scripts/h1_bic_sanity_check.py \
    --n_responses 500 \
    --k_max 8 \
    --cov_type diag \
    --min_tokens 20 \
    --out_dir bic_results
```

Output: `bic_results/summary.json` + `bic_results/bic_curves.png`.

| `pct_responses_with_K_ge_2` | Verdict                                  |
|-----------------------------|------------------------------------------|
| ≥ 70 %                      | H1 supported — proceed.                  |
| 40–70 %                     | Weakly supported — frame carefully.      |
| < 40 %                      | Reconsider mixture approach.             |

---

## 6. Training

All training goes through `rlhf_rw/main.py`. `--run_rewardbench True`
automatically runs RewardBench evaluation after training finishes.

### 6-a. Mixture mode (my method)

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
    --max_length 10000 \
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
    --run_rewardbench True \
    --seed 42 \
    2>&1 | tee train_mixture.log
```

**VRAM guidance:**
- 40GB (A100-40GB): `--batch_size 4 --gradient_acum_steps 2` (effective batch 8)
- 80GB (A100-80GB / H100): `--batch_size 8 --gradient_acum_steps 1`

### 6-b. Lopez-Cardona baselines (paper replication)

Feature combos (same naming as paper):

| Combo     | `-fmv` | `--features_used` | Features               |
|-----------|--------|--------------------|------------------------|
| fcomb1    | 1      | `1,0,0,0,0`        | TRT via ET1            |
| fcomb2.1  | 2      | `0,0,0,1,0`        | TRT via ET2            |
| fcomb2.2  | 2      | `0,1,0,1,0`        | FFD + TRT              |
| fcomb2.5  | 2      | `1,1,1,1,1`        | all 5 (paper best)     |

**fcomb2.5 (paper best, no mixture):**
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
    --max_length 10000 \
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
    --run_rewardbench True \
    --seed 42 \
    2>&1 | tee train_lopez_fcomb25.log
```

For other combos, swap `--fixations_model_version` and `--features_used`
per the table. For fcomb1, also set `--concat True --use_softprompt True`.

### 6-c. Mixture mode with different feature combos

You can combine mixture with any feature combo. Example — mixture + fcomb2.2:
```bash
python rlhf_rw/main.py \
    -d OpenAssistant/oasst1 \
    -m meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 4 --gradient_acum_steps 2 \
    --train_epochs 2 --learning_rate 5e-5 \
    --lr_scheduler_type cosine_with_min_lr --min_lr_ratio 0.7 \
    --weight_decay 0.1 --max_length 10000 --fp_dropout 0.1,0.3 \
    --use_lora True --use_quantization True --gradient_checkpointing True \
    --mode train \
    --fixations_model_version 2 \
    --features_used 0,1,0,1,0 \
    --concat True --use_softprompt True \
    --use_mixture_token True \
    --mixture_K 3 --mixture_cov_type diag \
    --run_rewardbench True \
    --seed 42 \
    2>&1 | tee train_mixture_fcomb22.log
```

Note: when `features_used` changes, F changes too (F = number of 1s in the
vector). The mixture module auto-adapts.

### 6-d. Pure baseline (no ET)

```bash
python rlhf_rw/main.py \
    -d OpenAssistant/oasst1 \
    -m meta-llama/Meta-Llama-3-8B-Instruct \
    --batch_size 8 --train_epochs 2 --learning_rate 5e-5 \
    --use_lora True --use_quantization True --mode train \
    --fixations_model_version 1 --features_used 1,0,0,0,0 \
    --concat False --use_softprompt False --use_mixture_token False \
    --run_rewardbench True --seed 42 \
    2>&1 | tee train_baseline.log
```

### 6-e. HelpSteer2 dataset

Replace `-d OpenAssistant/oasst1` with `-d nvidia/HelpSteer2`. Everything
else stays the same.

### 6-f. Different base models

Replace `-m` with any of:
- `meta-llama/Meta-Llama-3-8B` (pre-trained, no alignment)
- `meta-llama/Meta-Llama-3-8B-Instruct` (instruction-tuned)
- `mistralai/Mistral-7B-v0.3`

### 6-g. Background execution (survives SSH disconnect)

```bash
tmux new -s train
# run training command here
# Ctrl+B then D to detach
# tmux attach -t train to reconnect
```

---

## 7. Evaluation

**Automatic:** `--run_rewardbench True` in the training command runs
RewardBench after training finishes.

**Manual from checkpoint:**
```bash
python run_rewardbench.py \
    --ckpt_dir ./models_save/<your-run-dir>/checkpoint-XXX \
    --batch_size 4
```

Note: `args.json` must be in the checkpoint directory. If training crashed
before writing it, copy from the parent directory:
```bash
cp ./models_save/<run-dir>/args.json ./models_save/<run-dir>/checkpoint-XXX/
```

**Paper comparison (Table 5, OASST1, Llama-3-8B-Instruct):**

| Model        | RewardBench overall |
|--------------|---------------------|
| Baseline     | 46.9%               |
| Best ET (paper) | 58.4%            |
| **Mixture (ours)** | **TBD**        |

---

## 8. Hyperparameter tuning (Optuna)

```bash
# Mixture mode — tunes K, dropout, LR, batch, scheduler, etc.
python optuna_tune.py --mode mixture --n_trials 20

# Lopez baseline
python optuna_tune.py --mode lopez --n_trials 20

# Persistent study (resume across restarts)
python optuna_tune.py --mode mixture --n_trials 40 \
    --storage sqlite:///optuna_study.db \
    --study_name mixture_study
```

Available `--mode`: `mixture` / `mixture_full` / `lopez` / `baseline`.

**What's tuned per trial:**

Always: `learning_rate`, `batch_size`, `weight_decay`, `lr_scheduler_type`,
`min_lr_ratio`, `fp_dropout_1`, `fp_dropout_2`.

When mode is `mixture`/`mixture_full`: also `mixture_K ∈ [2,6]`,
`mixture_dropout ∈ [0, 0.3]`.

---

## 9. Ablation & sweeps

### 9-a. Feature ablation (paper Table 3/4 replication)

```bash
# All 4 feature combos under mixture mode
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode mixture \
    --out_dir ./sweep_results/mixture_feature_ablation

# Same under Lopez baseline
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode lopez \
    --out_dir ./sweep_results/lopez_feature_ablation

# Subset only
python ablation_sweep.py \
    --sweep feature_ablation \
    --base_mode mixture \
    --feature_combos fcomb1,fcomb2.5 \
    --out_dir ./sweep_results/ablation_subset
```

### 9-b. Mixture K sweep (most important ablation)

```bash
# K=2,3,4,5 with diag covariance (4 runs)
python ablation_sweep.py \
    --sweep mixture_hparam \
    --K_grid 2,3,4,5 \
    --cov_grid diag \
    --dropout_grid 0.1 \
    --proj_hidden_grid 128 \
    --out_dir ./sweep_results/K_sweep

# Full grid: K × cov × dropout (24 runs)
python ablation_sweep.py \
    --sweep mixture_hparam \
    --K_grid 2,3,4,5 \
    --cov_grid diag,full \
    --dropout_grid 0.05,0.1,0.2 \
    --proj_hidden_grid 128 \
    --out_dir ./sweep_results/mixture_full_grid
```

### 9-c. Seed stability (confidence intervals for paper)

```bash
python ablation_sweep.py \
    --sweep seeds \
    --base_mode mixture \
    --feature_combo fcomb2.5 \
    --mixture_K 3 --mixture_cov_type diag \
    --seeds 42,123,2024 \
    --out_dir ./sweep_results/seeds
```

### 9-d. Useful flags

- `--dry_run` — prints commands without executing.
- `--continue_on_fail` — keeps going if a run crashes.
- `--run_rewardbench` — adds RewardBench eval to every run.

---

## 10. Hyperparameter reference

### 10-a. RM training

| Flag                    | Default                | Notes                                          |
|-------------------------|------------------------|------------------------------------------------|
| `--learning_rate`       | `5e-5`                 | Tuning range: 1e-6..1e-4 log.                 |
| `--batch_size`          | `8`                    | On 40GB: use 4 + `--gradient_acum_steps 2`.   |
| `--train_epochs`        | `2`                    | Paper default.                                 |
| `--lr_scheduler_type`   | `constant_with_warmup` | Paper best: `cosine_with_min_lr`.             |
| `--min_lr_ratio`        | `0.7`                  | Cosine floor fraction.                         |
| `--weight_decay`        | `0.1`                  | AdamW.                                         |
| `--max_length`          | `10000`                | Pre-tokenization cap. fmv=2+concat auto-sets `max_tokens=1350`. |
| `--fp_dropout`          | `0.1,0.3`              | `(p_1, p_2)` in features projector MLP.        |
| `--gradient_acum_steps` | `1`                    | Raise to 2–4 on 40GB.                          |

### 10-b. Architecture / ET

| Flag                           | Default      | Notes                                           |
|--------------------------------|--------------|-------------------------------------------------|
| `-m / --model_name`            | —            | Any HF causal-seq-cls model.                    |
| `--fixations_model_version`    | `1`          | 1 = ET1 (TRT only), 2 = ET2 (5 features).       |
| `--features_used`              | `1,1,1,1,1`  | `[nFix, FFD, GPT, TRT, fixProp]` binary flags.  |
| `--concat`                     | `False`      | `True` = GazeConcat, `False` = GazeAdd.          |
| `--use_softprompt`             | `False`      | Wraps ET embeds with `<eye/>...</eye>`.          |
| `--use_lora`                   | `True`       | QLoRA (r=8, alpha=32, dropout=0.1).              |
| `--use_quantization`           | `True`       | 4-bit NF4 + bf16 compute.                        |

### 10-c. Mixture-token

| Flag                      | Default | Notes                                                          |
|---------------------------|---------|----------------------------------------------------------------|
| `--use_mixture_token`     | `False` | Master switch.                                                 |
| `--mixture_K`             | `3`     | # GMM components. **Main tuning target.** Try {2,3,4,5}.       |
| `--mixture_cov_type`      | `diag`  | `diag` / `full` / `tied` / `spherical`. `diag` safest.          |
| `--mixture_proj_hidden`   | `128`   | Hidden dim of (π,μ,Σ) → hidden_size projector.                  |
| `--mixture_dropout`       | `0.1`   | Dropout in projector MLP.                                       |
| `--mixture_log_transform` | `True`  | `log1p` before GMM fit.                                         |

**What K means in practice:**

| K | Assumption | Descriptor size (F=5, diag) |
|---|---|---|
| 2 | bimodal (skim vs read) | 23 |
| 3 | trimodal (skim / normal / deep) | 33 |
| 4 | 4 reading modes | 43 |
| 5 | 5 reading modes | 53 |

More K = more expressive but risks overfitting on short responses.

---

## 11. Environment variables

| Var                    | Default               | Purpose                              |
|------------------------|-----------------------|--------------------------------------|
| `ET2_CHECKPOINT_PATH`  | auto (HF cache)       | ET2 checkpoint path.                 |
| `LMDB_CACHE_PATH`      | `./buffer_train.lmdb` | Fixation prediction cache.           |
| `WANDB_MODE`           | online                | `disabled` to skip wandb.            |
| `CUDA_VISIBLE_DEVICES` | all                   | Pin to specific GPUs.                |

---

## 12. Output artifacts

```
models_save/<auto-named-run-dir>/
├── adapter_model.safetensors          # LoRA weights
├── fixations_projector_state_dict.bin # ET-features MLP
├── layer_norm_state_dic.bin           # post-projector LN
├── mixture_module.bin                 # (if mixture) projector weights
├── args.json                          # every CLI flag used
├── results_dataset_test.json          # test accuracy
└── rewardbench_results.json           # (if --run_rewardbench True)
```

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| CUDA OOM | `--batch_size 2 --gradient_acum_steps 4` |
| wandb login 86-char error | `pip install -U wandb` then `wandb login` |
| ET2 not found | `python setup_et_models.py --skip-install` or ignore (auto-downloads) |
| `args.json` not found on eval | `cp <run-dir>/args.json <run-dir>/checkpoint-XXX/` |
| LMDB mapsize error | Delete `buffer_train.lmdb` or set `LMDB_CACHE_PATH` to fresh path |

---

## Citation

```bibtex
@inproceedings{Lopez-Cardona2025Seeing,
  title  = {Seeing Eye to AI: Human Alignment via Gaze-Based Response Rewards for Large Language Models},
  author = {Lopez-Cardona, Angela and Segura, Carlos and Karatzoglou, Alexandros and Abadal, Sergi and Arapakis, Ioannis},
  booktitle = {ICLR},
  year   = {2025},
  url    = {https://openreview.net/forum?id=uZgK0tcPqd}
}
```

ET predictors: Li & Rudzicz (2021) / Huang & Hollenstein (2023).

## License

LGPLv3 — see [LICENSE](./LICENSE).

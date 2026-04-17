# AGD — Mixture-Augmented Gaze Reward Modelling

Extension of Lopez-Cardona et al. (ICLR 2025) that models each response's
eye-tracking feature distribution as a K-component Gaussian mixture and
prepends its (π, μ, Σ) summary as a single token to the reward model input.

<p align="center">
  <img src="assets/pipeline.png" alt="Overview" width="80%">
</p>

---

## 1. Install

```
git clone <your-fork> agd && cd agd
bash install.sh
huggingface-cli login
python setup_et_models.py --skip-install
```

`install.sh` auto-detects existing torch+CUDA. Pass `FORCE_TORCH=1` to pin
`torch==2.2.2`.

---

## 2. wandb

```
pip install -U wandb
wandb login
```

To disable wandb: `export WANDB_MODE=disabled`.

---

## 3. ET predictors

**ET1** (Huang & Hollenstein 2023, 1 feature = TRT): installed by `setup_et_models.py`.

**ET2** (Li & Rudzicz 2021, 5 features = nFix, FFD, GPT, TRT, fixProp):
pre-downloaded by `install.sh`. Custom checkpoint via `export ET2_CHECKPOINT_PATH=/path`.

---

## 4. Key concepts: K and F

**F = 5**: number of ET features `[nFix, FFD, GPT, TRT, fixProp]`. Fixed by ET2.

**K**: number of Gaussian mixture components. The main hyperparameter.

### H1 (multimodality) — result from the full OASST1 sweep (20,733 responses, diag cov, k_max=8)

The per-response ET feature distribution is clearly multimodal: 99.6% of
responses prefer K≥2 by BIC. But the evidence does not support large K:

| Jump | Mean ΔBIC | Median ΔBIC | % strong (ΔBIC > 10) |
|---|---|---|---|
| K=1 → K=2 | **+302** | +225 | **97.8 %** |
| K=2 → K=3 | +13 | +2.5 | 42.3 % |
| K=3 → K=4 | −20 | −24 | 15.9 % |
| K=4 → K=5 | −34 | −37 | 15.6 % |

Mean ΔBIC flips negative from K=3→K=4 — adding more components overfits
per-response. The raw K-argmin distribution looks like it favours K=2/3, but
conditioning on response length shows the picture is cleaner than it seems:

| Length quartile | K=2 | K=3 |
|---|---|---|
| Q1 (shortest) | 45.9 % | 15.1 % |
| Q2 | **70.7 %** | 25.0 % |
| Q3 | 39.2 % | **52.6 %** |
| Q4 (longest) | 7.7 % | **55.8 %** |

**Default K=2**, with **K=3 as the ablation**. K ≥ 4 is not supported by the data.

### Re-run the H1 sweep

Quick (100 responses):

```
python scripts/h1_bic_sanity_check.py --n_responses 100 --k_max 8 --cov_type diag
```

Full (all eligible OASST1 responses):

```
python scripts/h1_bic_sanity_check.py --n_responses 0 --k_max 8 --cov_type diag --out_dir bic_results
```

Defaults are now `cov_type=diag` (matches training) and `k_max=8` (avoids the
K=6 ceiling artifact seen in the earlier k_max=6 runs).

---

## 5. Training

All training goes through `rlhf_rw/main.py`. `--run_rewardbench True` runs
RewardBench eval after training finishes.

### 5-a. Mixture mode (recommended: K=2, diag)

```
python rlhf_rw/main.py -d OpenAssistant/oasst1 -m meta-llama/Meta-Llama-3-8B-Instruct --batch_size 4 --gradient_acum_steps 2 --train_epochs 2 --learning_rate 5e-5 --lr_scheduler_type cosine_with_min_lr --min_lr_ratio 0.7 --weight_decay 0.1 --max_length 10000 --fp_dropout 0.1,0.3 --use_lora True --use_quantization True --gradient_checkpointing True --mode train --fixations_model_version 2 --features_used 1,1,1,1,1 --concat True --use_softprompt True --use_mixture_token True --mixture_K 2 --mixture_cov_type diag --mixture_proj_hidden 128 --mixture_dropout 0.1 --mixture_log_transform True --run_rewardbench True --rb_char_filter 100000 --seed 42
```

K=3 ablation — replace `--mixture_K 2` with `--mixture_K 3`. Everything else stays the same.

VRAM:
- 40 GB (A100-40GB): `--batch_size 4 --gradient_acum_steps 2` (effective batch 8)
- 80 GB (A100-80GB / H100): `--batch_size 8 --gradient_acum_steps 1`

### 5-b. Lopez baselines (paper replication)

| Combo     | `-fmv` | `--features_used` | Features |
|-----------|--------|-------------------|----------|
| fcomb1    | 1      | `1,0,0,0,0`       | TRT (ET1) |
| fcomb2.1  | 2      | `0,0,0,1,0`       | TRT (ET2) |
| fcomb2.2  | 2      | `0,1,0,1,0`       | FFD + TRT |
| fcomb2.5  | 2      | `1,1,1,1,1`       | all 5 (paper best) |

fcomb2.5 (paper best, no mixture):

```
python rlhf_rw/main.py -d OpenAssistant/oasst1 -m meta-llama/Meta-Llama-3-8B-Instruct --batch_size 8 --train_epochs 2 --learning_rate 5e-5 --lr_scheduler_type cosine_with_min_lr --min_lr_ratio 0.7 --weight_decay 0.1 --max_length 10000 --fp_dropout 0.1,0.3 --use_lora True --use_quantization True --gradient_checkpointing True --mode train --fixations_model_version 2 --features_used 1,1,1,1,1 --concat True --use_softprompt True --use_mixture_token False --run_rewardbench True --rb_char_filter 100000 --seed 42
```

For other combos, swap `--fixations_model_version` and `--features_used` per the table above.

### 5-c. Pure no-ET baseline

```
python rlhf_rw/main.py -d OpenAssistant/oasst1 -m meta-llama/Meta-Llama-3-8B-Instruct --batch_size 8 --train_epochs 2 --learning_rate 5e-5 --use_lora True --use_quantization True --mode train --fixations_model_version 1 --features_used 1,0,0,0,0 --concat False --use_softprompt False --use_mixture_token False --run_rewardbench True --rb_char_filter 100000 --seed 42
```

### 5-d. HelpSteer2

Replace `-d OpenAssistant/oasst1` with `-d nvidia/HelpSteer2`. Everything else stays the same.

---

## 6. Evaluation

Automatic via `--run_rewardbench True` in the training command.

Manual from a checkpoint:

```
python run_rewardbench.py --ckpt_dir ./models_save/<your-run-dir>/checkpoint-XXX --batch_size 4 --rb_char_filter 100000
```

`args.json` must be in the checkpoint directory. If training crashed before
writing it:

```
cp ./models_save/<run-dir>/args.json ./models_save/<run-dir>/checkpoint-XXX/
```

### Important: `--rb_char_filter`

The earlier default filtered out RewardBench rows where
`len(chosen_chat) + len(rejected_chat) > 10000` characters. This silently
dropped every `alpacaeval-easy` and `alpacaeval-hard` row (responses are long),
so the Chat score became a weighted average of only the surviving subsets.
Default is now `100000`, which keeps all rows. Logs now list any subset that
was in the raw data but ended up with zero post-filter rows.

### Categories (current RewardBench)

- **Chat:** alpacaeval-easy, alpacaeval-length, alpacaeval-hard, mt-bench-easy, mt-bench-med
- **Chat Hard:** mt-bench-hard, llmbar-natural, llmbar-adver-{neighbor,GPTInst,GPTOut,manual}
- **Safety:** refusals-{dangerous,offensive}, xstest-should-{refuse,respond}, donotanswer
- **Reasoning:** math-prm, hep-cpp, hep-go, hep-java, hep-js, hep-python, hep-rust

`math-prm` was previously uncategorised — it is now aggregated into Reasoning
(447 rows, the largest single subset). The old `CODE_KEYS` name has been
replaced with `REASONING_KEYS` to match current RewardBench terminology.

### Paper comparison (Table 5, OASST1, Llama-3-8B-Instruct)

| Model              | RewardBench overall |
|--------------------|---------------------|
| Baseline           | 46.9 %              |
| Best ET (paper)    | 58.4 %              |
| **Mixture (ours)** | **TBD**             |

---

## 7. Hyperparameter tuning (Optuna)

```
python optuna_tune.py --mode mixture --n_trials 20
```

```
python optuna_tune.py --mode lopez --n_trials 20
```

Persistent study (resume across restarts):

```
python optuna_tune.py --mode mixture --n_trials 40 --storage sqlite:///optuna_study.db --study_name mixture_study
```

Available modes: `mixture`, `mixture_full`, `lopez`, `baseline`.

---

## 8. Ablations

### 8-a. Feature ablation

```
python ablation_sweep.py --sweep feature_ablation --base_mode mixture --out_dir ./sweep_results/mixture_feature_ablation
```

```
python ablation_sweep.py --sweep feature_ablation --base_mode lopez --out_dir ./sweep_results/lopez_feature_ablation
```

### 8-b. K sweep (the main mixture ablation)

Minimal (K=2 vs K=3, diag only):

```
python ablation_sweep.py --sweep mixture_hparam --K_grid 2,3 --cov_grid diag --dropout_grid 0.1 --proj_hidden_grid 128 --out_dir ./sweep_results/K_sweep
```

Full grid:

```
python ablation_sweep.py --sweep mixture_hparam --K_grid 2,3,4,5 --cov_grid diag,full --dropout_grid 0.05,0.1,0.2 --proj_hidden_grid 128 --out_dir ./sweep_results/mixture_full_grid
```

### 8-c. Seed stability

```
python ablation_sweep.py --sweep seeds --base_mode mixture --feature_combo fcomb2.5 --mixture_K 2 --mixture_cov_type diag --seeds 42,123,2024 --out_dir ./sweep_results/seeds
```

---

## 9. Hyperparameter reference

### 9-a. RM training

| Flag | Default | Notes |
|------|---------|-------|
| `--learning_rate` | `5e-5` | Tune in `[1e-6, 1e-4]` log. |
| `--batch_size` | `8` | 40 GB: `4` + `--gradient_acum_steps 2`. |
| `--train_epochs` | `2` | Paper default. |
| `--lr_scheduler_type` | `constant_with_warmup` | Paper best: `cosine_with_min_lr`. |
| `--min_lr_ratio` | `0.7` | Cosine floor. |
| `--weight_decay` | `0.1` | AdamW. |
| `--max_length` | `10000` | Char-length filter for training data. |
| `--fp_dropout` | `0.1,0.3` | `(p_1, p_2)` in features projector. |
| `--gradient_acum_steps` | `1` | Raise to 2–4 on 40 GB. |
| `--rb_char_filter` | `100000` | Char-length cap applied during RewardBench eval only. |

### 9-b. Architecture / ET

| Flag | Default | Notes |
|------|---------|-------|
| `-m / --model_name` | — | Any HF causal-seq-cls model. |
| `--fixations_model_version` | `1` | 1 = ET1 (TRT), 2 = ET2 (5 features). |
| `--features_used` | `1,1,1,1,1` | Binary flags. |
| `--concat` | `False` | `True` = GazeConcat, `False` = GazeAdd. |
| `--use_softprompt` | `False` | Wraps ET embeds with `<eye/>...</eye>`. |
| `--use_lora` | `True` | QLoRA r=8, α=32, dropout=0.1. |
| `--use_quantization` | `True` | 4-bit NF4 + bf16 compute. |

### 9-c. Mixture-token

| Flag | Default | Notes |
|------|---------|-------|
| `--use_mixture_token` | `False` | Master switch. |
| `--mixture_K` | `3` | Recommended: try `2` first (see H1 table). |
| `--mixture_cov_type` | `diag` | `diag` / `full` / `tied` / `spherical`. |
| `--mixture_proj_hidden` | `128` | Projector hidden dim. |
| `--mixture_dropout` | `0.1` | Projector MLP dropout. |
| `--mixture_log_transform` | `True` | `log1p` before GMM fit. |

Descriptor size for F=5, diag: K=2 → 23 dims, K=3 → 33 dims, K=4 → 43 dims.

---

## 10. Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `ET2_CHECKPOINT_PATH` | auto | ET2 checkpoint path. |
| `LMDB_CACHE_PATH` | `./buffer_train.lmdb` | Fixation + mixture descriptor cache. |
| `WANDB_MODE` | `online` | `disabled` to skip wandb. |
| `CUDA_VISIBLE_DEVICES` | all | Pin to specific GPUs. |

---

## 11. Output artifacts

```
models_save/<auto-named-run-dir>/
├── adapter_model.safetensors          LoRA weights
├── fixations_projector_state_dict.bin ET-features MLP
├── layer_norm_state_dic.bin           post-projector LN
├── mixture_module.bin                 (if mixture) projector weights
├── args.json                          every CLI flag used
├── results_dataset_test.json          test accuracy
└── results_rewardbench.json           (if --run_rewardbench True)
```

`results_rewardbench.json` now also includes `missing_subsets` and
`unknown_subsets` keys listing any RewardBench subsets present in the dataset
but absent from `accs`, or present in `accs` but not aggregated into any category.

---

## 12. Bug fixes relative to the previous revision

1. **RewardBench char-length filter was dropping AlpacaEval subsets.** The
   `filter_df_lenght_columns` step filtered rows by character count using the
   same `max_length` that controls tokenizer truncation, so `max_length=10000`
   (chars) dropped every `alpacaeval-easy`/`alpacaeval-hard` row. The fix
   decouples the two: RewardBench eval now uses `--rb_char_filter` (default
   `100000`), leaving tokenizer `max_length` untouched.
2. **`math-prm` was uncategorised.** The largest RewardBench subset (447 rows)
   was not in any of `CHAT_KEYS`/`CHATHARD_KEYS`/`SAFETY_KEYS`/`CODE_KEYS`, so
   it contributed to no aggregate score. Added a `REASONING_KEYS` category
   containing `math-prm + hep-*`, which matches current RewardBench taxonomy.
3. **Per-subset eval failures silently dropped subsets.** `eval_model(mode='all')`
   now wraps each subset's `trainer.evaluate()` in try/except and logs a
   summary of failed/skipped subsets.
4. **Zero-accuracy subsets were silently dropped** by `res.get('eval_accuracy') or res.get('accuracy')`
   (0.0 is falsy in Python). Replaced with explicit `is not None`.
5. **Noise injection ran at eval time, not train time.** `process_fixations`
   checked `self.training is False` when gating the ET-feature noise. Fixed
   to `self.training and self.noise_factor > 0`.
6. **GMM was refit from scratch on every forward pass.** The sklearn fit is
   serial, CPU-bound, and dominated step time. Descriptors are now hashed on
   `(input_ids_no_pad, K, F, cov_type, log_transform)` and cached in the same
   LMDB store that already caches fixations. First epoch pays the cost, later
   epochs are free.
7. **`_fit_one` only adjusted `effective_K` for `full` covariance.** `diag`
   covariance can also blow EM on short responses. The effective-K guard now
   uses a single `_params_per_component` helper for every covariance type.
8. **`torch.cuda.empty_cache()` in hot paths.** Removed from
   `process_fixations` and the soft-prompt forward — these sync the GPU on
   every forward and cost ~10–30 % of step time.
9. **Lambda late-binding in `load_dataset_rewardbench`.** The per-subset
   filter lambda captured `rb_subset` by reference. Defensive default-arg
   fix, plus sorted iteration for deterministic logs.
10. **H1 script defaults updated.** `cov_type` now defaults to `diag`
    (matches training), `k_max` raised to `8` to remove the K=6 ceiling
    artifact seen in earlier runs.

---

## 13. Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA OOM | `--batch_size 2 --gradient_acum_steps 4` |
| wandb 86-char error | `pip install -U wandb` then `wandb login` |
| ET2 not found | `python setup_et_models.py --skip-install` |
| `args.json` missing on eval | `cp <run-dir>/args.json <run-dir>/checkpoint-XXX/` |
| LMDB mapsize error | Delete `buffer_train.lmdb` or set `LMDB_CACHE_PATH` |
| RewardBench missing subsets | Raise `--rb_char_filter` (default 100000 should be enough) |

---

## Citation

```
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

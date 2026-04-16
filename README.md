# AGD — Mixture-Augmented Gaze Reward Modelling

Extension of the Lopez-Cardona et al. (ICLR 2025) GazeReward framework that
models each response's eye-tracking feature distribution as a **Gaussian
mixture** rather than a flat collection of per-token vectors.

<p align="center">
  <img src="assets/pipeline.png" alt="Overview" width="80%">
</p>

## What's new vs the upstream GazeReward paper

The original paper feeds per-token ET features (5 features × sequence length)
through a 2-layer MLP projector and concatenates them with text embeddings.
This treats features as points in ℝ⁵ with no generative structure.

This repo adds a **per-response GMM summary token**: for each response we fit
a K-component Gaussian mixture to its (n_tokens, 5) feature matrix, flatten
the mixture parameters (π, μ, Σ) into a fixed-size descriptor, and project
that descriptor to a single token that is prepended to the RM input
sequence. The thesis is that reading is a mixture process (skim / normal /
deep / regression) and modelling it explicitly provides a signal the flat
projector can't extract.

Key capabilities:

- Two experimental families: **mixture** (new) and **lopez** (paper baseline).
- Feature-combo replication: `fcomb1` / `fcomb2.1` / `fcomb2.2` / `fcomb2.5`.
- H1 BIC sanity-check script to verify the mixture hypothesis *before* training.
- Ablation + hyperparameter sweep runner (`ablation_sweep.py`).
- Optuna + Hyperband hyperparameter search (`optuna_tune.py`).

## Quick start

```bash
# 1. install
bash install.sh
huggingface-cli login

# 2. ET2 checkpoint (downloads from HF Hub by default)
python setup_et_models.py

# 3. verify mixture hypothesis (30 min)
python scripts/h1_bic_sanity_check.py

# 4. train
bash run_modes.sh mixture        # mixture token, K=3, diag covariance
bash run_modes.sh lopez          # Lopez-Cardona fcomb2.5 baseline
```

## Full documentation

See **[README_for_install_running_modes.md](./README_for_install_running_modes.md)**
for:

- System requirements
- Detailed install (including ET1 / ET2 setup)
- All `run_modes.sh` presets and `main.py` flags
- Hyperparameter reference tables
- Ablation & sweep usage
- Environment variables
- Output artifacts
- Troubleshooting

## Layout

```
.
├── rlhf_rw/
│   ├── main.py                              # training entry point
│   ├── models/
│   │   ├── mixture_module.py                # MixtureTokenModule (per-response GMM)
│   │   ├── reward_model_base.py             # base class + mixture hook
│   │   ├── reward_model_general_sp.py       # GazeConcat variant
│   │   ├── reward_model_general_add.py      # GazeAdd variant
│   │   └── reward_model_factory.py
│   ├── trainers/
│   │   ├── reward_trainer.py
│   │   └── reward_trainer_general.py
│   └── reward_utils/
├── scripts/
│   └── h1_bic_sanity_check.py               # paper go/no-go
├── et2_wrapper.py                           # ET2 (RoBERTa) wrapper
├── ablation_sweep.py                        # feature / hparam / seed sweeps
├── optuna_tune.py                           # Optuna Hyperband search
├── run_rewardbench.py                       # RewardBench eval
├── run_modes.sh                             # convenience wrapper
├── setup_et_models.py                       # ET2 checkpoint setup
└── install.sh
```

## Citation

Upstream paper:

```bibtex
@inproceedings{Lopez-Cardona2025Seeing,
  title     = {Seeing Eye to AI: Human Alignment via Gaze-Based Response Rewards for Large Language Models},
  author    = {Lopez-Cardona, Angela and Segura, Carlos and Karatzoglou, Alexandros and Abadal, Sergi and Arapakis, Ioannis},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=uZgK0tcPqd}
}
```

## License

© 2025 Telefónica Innovación Digital (upstream) + this extension.
Released under the GNU Lesser General Public License v3.0 (LGPLv3).
See [LICENSE](./LICENSE).

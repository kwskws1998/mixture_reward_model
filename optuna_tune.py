"""Optuna/Hyperband hyperparameter search for GazeReward. See README for usage."""

import argparse
import functools
import gc
import json
import math
import os
import sys
import threading
import time
import pathlib

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = str(pathlib.Path(__file__).parent.resolve())
sys.path.insert(0, ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import set_seed
from rlhf_rw.models.reward_model_factory import ModelFactory
from rlhf_rw.trainers.reward_trainer_general import (
    RewardTrainerConstructorGeneral,
    model_init_func,
)

# ─────────────────────────────────────────────────────────────────────────────
#  VRAM watcher (background thread)
# ─────────────────────────────────────────────────────────────────────────────

VRAM_THRESHOLD_GB = float(os.environ.get("VRAM_THRESHOLD_GB", 35))
_vram_stop = threading.Event()

def _vram_watcher(log_path: str, threshold_gb: float):
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"[vram_watcher] disabled ({e})")
        return

    import datetime
    threshold_bytes = threshold_gb * 1024 ** 3
    while not _vram_stop.is_set():
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024 ** 3
        record = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "used_gb": round(used_gb, 2),
            "total_gb": round(info.total / 1024 ** 3, 2),
        }
        with open(log_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if info.used > threshold_bytes:
            print(
                f"[VRAM WARNING] {used_gb:.1f} GB used "
                f"(>{threshold_gb:.0f} GB threshold)",
                flush=True,
            )
        _vram_stop.wait(60)  # sample every 60 s

def start_vram_watcher(log_path: str = "vram_log.jsonl") -> threading.Thread:
    t = threading.Thread(
        target=_vram_watcher,
        args=(log_path, VRAM_THRESHOLD_GB),
        daemon=True,
        name="vram_watcher",
    )
    t.start()
    return t

def peak_vram_gb(log_path: str) -> float:
    if not os.path.isfile(log_path):
        return 0.0
    try:
        with open(log_path) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        return max((r["used_gb"] for r in records), default=0.0)
    except Exception:
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
#  Mode definitions
# ─────────────────────────────────────────────────────────────────────────────

MODES = {
    # ── Mixture (default): all 5 features + per-response GMM summary token.
    #    The mixture token is prepended to the input sequence; the projector
    #    learns to encode (π, μ, Σ) into a hidden_size embedding.
    "mixture": dict(
        fixations_model_version=2,
        features_used=[1, 1, 1, 1, 1],
        concat=True,
        use_softprompt=True,
        use_mixture_token=True,
        mixture_K=3,
        mixture_cov_type="diag",
        mixture_proj_hidden=128,
        mixture_dropout=0.1,
        mixture_log_transform=True,
    ),
    # ── Mixture with full covariance — captures inter-feature correlations.
    "mixture_full": dict(
        fixations_model_version=2,
        features_used=[1, 1, 1, 1, 1],
        concat=True,
        use_softprompt=True,
        use_mixture_token=True,
        mixture_K=3,
        mixture_cov_type="full",
        mixture_proj_hidden=128,
        mixture_dropout=0.1,
        mixture_log_transform=True,
    ),
    # ── Lopez-Cardona fcomb2.5: all 5 features, no mixture token (paper baseline).
    "lopez": dict(
        fixations_model_version=2,
        features_used=[1, 1, 1, 1, 1],
        concat=True,
        use_softprompt=True,
        use_mixture_token=False,
        mixture_K=3,
        mixture_cov_type="diag",
        mixture_proj_hidden=128,
        mixture_dropout=0.1,
        mixture_log_transform=True,
    ),
    # ── Baseline: no ET
    "baseline": dict(
        fixations_model_version=1,
        features_used=[1, 0, 0, 0, 0],
        concat=False,
        use_softprompt=False,
        use_mixture_token=False,
        mixture_K=3,
        mixture_cov_type="diag",
        mixture_proj_hidden=128,
        mixture_dropout=0.1,
        mixture_log_transform=True,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
#  Optuna Pruning Callback
# ─────────────────────────────────────────────────────────────────────────────

class OptunaPruningCallback:
    """
    HuggingFace Trainer callback that reports validation accuracy to Optuna
    after every evaluation step, allowing HyperbandPruner to prune weak trials.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "eval_accuracy"):
        self.trial = trial
        self.monitor = monitor
        self._step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        value = metrics.get(self.monitor, None)
        if value is None:
            return
        self._step += 1
        self.trial.report(value, step=self._step)
        if self.trial.should_prune():
            print(
                f"[Optuna] Trial {self.trial.number} pruned at step {self._step} "
                f"(val_acc={value:.4f})"
            )
            raise optuna.TrialPruned()

# ─────────────────────────────────────────────────────────────────────────────
#  Objective
# ─────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial, cli_args: argparse.Namespace) -> float:
    """
    Single Optuna trial. Suggests hyperparameters, trains the model, and
    returns the best validation accuracy observed during training.
    """

    lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.3, log=True)
    lr_scheduler = trial.suggest_categorical(
        "lr_scheduler_type",
        ["constant_with_warmup", "cosine_with_min_lr", "linear"],
    )
    min_lr_ratio = trial.suggest_float("min_lr_ratio", 0.5, 0.9)
    fp1 = trial.suggest_float("fp_dropout_1", 0.0, 0.3)
    fp2 = trial.suggest_float("fp_dropout_2", 0.1, 0.5)
    fp_dropout = [fp1, fp2]

    mode_cfg = MODES[cli_args.mode].copy()
    if cli_args.mode in ("mixture", "mixture_full") and mode_cfg["use_mixture_token"]:
        mode_cfg["mixture_K"] = trial.suggest_int("mixture_K", 2, 6)
        mode_cfg["mixture_dropout"] = trial.suggest_float("mixture_dropout", 0.0, 0.3)

    set_seed(cli_args.seed)

    pruning_cb = OptunaPruningCallback(trial)

    gc.collect()
    torch.cuda.empty_cache()

    trainer = RewardTrainerConstructorGeneral(
        model_name=cli_args.model_name,
        dataset_name=cli_args.dataset_name,
        use_lora=True,
        use_quantization=True,
        batch_size=batch_size,
        train_epochs=cli_args.max_epochs,
        gradient_acum_steps=cli_args.gradient_acum_steps,
        logging_steps=cli_args.logging_steps,
        gradient_checkpointing=True,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,
        min_lr_ratio=min_lr_ratio,
        weight_decay=weight_decay,
        seed=cli_args.seed,
        fp_dropout=fp_dropout,
        load_fix_model=True,
        max_length=cli_args.max_length,
        **mode_cfg,
    )

    # Monkey-patch set_trainer to inject the Optuna pruning callback.
    _orig_set_trainer = trainer.set_trainer

    def _patched_set_trainer(save_folder="./reward_model"):
        _orig_set_trainer(save_folder=save_folder)
        # Append our callback AFTER the original trainer is created.
        from transformers import TrainerCallback

        class _CB(TrainerCallback):
            def on_evaluate(self_, args, state, control, metrics=None, **kw):
                pruning_cb.on_evaluate(args, state, control, metrics=metrics, **kw)

        trainer.trainer.add_callback(_CB())

    trainer.set_trainer = _patched_set_trainer

    save_folder = os.path.join(
        ROOT,
        "models_save",
        f"optuna_trial_{trial.number}",
    )
    try:
        trainer.train_model(save_folder=save_folder)
        results = trainer.eval_model_v2()
        accuracy = results.get("eval_accuracy", 0.0)
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f"[Optuna] Trial {trial.number} failed: {exc}")
        raise optuna.TrialPruned()
    finally:
        # Release GPU memory between trials
        try:
            del trainer
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    trial.set_user_attr("eval_accuracy", accuracy)
    return accuracy

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GazeReward Optuna / Hyperband tuning")
    p.add_argument("--mode", choices=list(MODES), default="mixture",
                   help="Run mode (mixture / mixture_full / lopez / baseline)")
    p.add_argument("-m", "--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("-d", "--dataset_name", default="OpenAssistant/oasst1")
    p.add_argument("--n_trials", type=int, default=20,
                   help="Total number of Optuna trials")
    p.add_argument("--max_epochs", type=int, default=2,
                   help="Maximum training epochs per trial (Hyperband max_resource)")
    p.add_argument("--min_resource", type=int, default=1,
                   help="Hyperband min_resource (eval checkpoints)")
    p.add_argument("--reduction_factor", type=int, default=3,
                   help="Hyperband reduction factor (default 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_acum_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--max_length", type=int, default=10000)
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (e.g. sqlite:///optuna.db). "
                        "Defaults to in-memory (results lost on exit).")
    p.add_argument("--study_name", default=None,
                   help="Optuna study name (useful with persistent storage)")
    p.add_argument("--vram_log", default="vram_log_optuna.jsonl",
                   help="Path for VRAM usage log")
    return p.parse_args()

def main():
    args = parse_args()

    # ── VRAM watcher ────────────────────────────────────────────────────
    vram_thread = start_vram_watcher(log_path=args.vram_log)
    print(f"[vram_watcher] VRAM log: {args.vram_log}  |  threshold: {VRAM_THRESHOLD_GB} GB")

    # ── Optuna study ────────────────────────────────────────────────────
    study_name = args.study_name or f"gazereward_{args.mode}"
    sampler = TPESampler(seed=args.seed)
    pruner = HyperbandPruner(
        min_resource=args.min_resource,
        max_resource=args.max_epochs,
        reduction_factor=args.reduction_factor,
    )
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=(args.storage is not None),
    )

    print(
        f"\n{'='*60}\n"
        f"  Optuna Hyperband search\n"
        f"  Study       : {study_name}\n"
        f"  Mode        : {args.mode}\n"
        f"  Model       : {args.model_name}\n"
        f"  Dataset     : {args.dataset_name}\n"
        f"  Trials      : {args.n_trials}\n"
        f"  Max epochs  : {args.max_epochs}\n"
        f"  HB factor   : {args.reduction_factor}\n"
        f"  Storage     : {args.storage or 'in-memory'}\n"
        f"{'='*60}\n"
    )

    obj = functools.partial(objective, cli_args=args)
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=True)

    # ── Results ─────────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "="*60)
    print("  Best trial results")
    print(f"  Accuracy  : {best.value:.4f}")
    print(f"  Params    :")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
    print("="*60)

    # Save results to JSON
    results_path = f"optuna_results_{study_name}.json"
    summary = {
        "study_name": study_name,
        "mode": args.mode,
        "best_value": best.value,
        "best_params": best.params,
        "n_trials": len(study.trials),
        "peak_vram_gb": peak_vram_gb(args.vram_log),
    }
    with open(results_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print(f"  Peak VRAM       : {summary['peak_vram_gb']:.2f} GB")

    _vram_stop.set()
    return summary

if __name__ == "__main__":
    main()

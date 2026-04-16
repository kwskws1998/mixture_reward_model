"""Run feature-ablation and mixture-hyperparameter sweeps. See README for usage."""

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent
MAIN_PY = ROOT / "rlhf_rw" / "main.py"


FEATURE_COMBOS = {
    "fcomb1":   {"fmv": 1, "features_used": "1,0,0,0,0"},
    "fcomb2.1": {"fmv": 2, "features_used": "0,0,0,1,0"},
    "fcomb2.2": {"fmv": 2, "features_used": "0,1,0,1,0"},
    "fcomb2.5": {"fmv": 2, "features_used": "1,1,1,1,1"},
}


MODE_PRESETS = {
    "lopez": {
        "concat": "True",
        "use_softprompt": "True",
        "use_mixture_token": "False",
    },
    "mixture": {
        "concat": "True",
        "use_softprompt": "True",
        "use_mixture_token": "True",
    },
    "baseline": {
        "concat": "False",
        "use_softprompt": "False",
        "use_mixture_token": "False",
        "fmv": 1,
        "features_used": "1,0,0,0,0",
    },
}


def slugify(parts):
    return "_".join(str(p).replace("/", "-").replace(",", "") for p in parts if p)


def build_cmd(args, run_cfg, run_dir):
    cmd = [
        sys.executable,
        str(MAIN_PY),
        "-d", args.dataset,
        "-m", args.model,
        "--batch_size", str(args.batch_size),
        "--train_epochs", str(args.train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler_type", args.lr_scheduler_type,
        "--min_lr_ratio", str(args.min_lr_ratio),
        "--weight_decay", str(args.weight_decay),
        "--seed", str(run_cfg.get("seed", args.seed)),
        "--fp_dropout", args.fp_dropout,
        "--gradient_acum_steps", str(args.gradient_acum_steps),
        "--logging_steps", str(args.logging_steps),
        "--max_length", str(args.max_length),
        "--use_lora", "True",
        "--use_quantization", "True",
        "--gradient_checkpointing", "True",
        "--mode", "train",
        "--fixations_model_version", str(run_cfg["fmv"]),
        "--features_used", run_cfg["features_used"],
        "--concat", run_cfg["concat"],
        "--use_softprompt", run_cfg["use_softprompt"],
        "--use_mixture_token", run_cfg["use_mixture_token"],
    ]
    if str(run_cfg.get("use_mixture_token", "False")).lower() == "true":
        cmd += [
            "--mixture_K", str(run_cfg["mixture_K"]),
            "--mixture_cov_type", run_cfg["mixture_cov_type"],
            "--mixture_proj_hidden", str(run_cfg.get("mixture_proj_hidden", 128)),
            "--mixture_dropout", str(run_cfg.get("mixture_dropout", 0.1)),
            "--mixture_log_transform", str(run_cfg.get("mixture_log_transform", True)),
        ]
    if args.run_rewardbench:
        cmd += ["--run_rewardbench", "True"]
    return cmd


def run_one(args, run_cfg, tag, out_dir):
    run_dir = out_dir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cfg_snapshot = dict(run_cfg)
    cfg_snapshot["tag"] = tag
    cfg_snapshot["started"] = datetime.utcnow().isoformat() + "Z"
    (run_dir / "config.json").write_text(json.dumps(cfg_snapshot, indent=2))

    cmd = build_cmd(args, run_cfg, run_dir)
    print(f"[sweep] ▶ {tag}")
    print(f"[sweep]   cmd: {' '.join(cmd)}")
    print(f"[sweep]   log: {log_path}")

    if args.dry_run:
        print("[sweep]   dry_run: not executing")
        return 0

    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"# {tag}\n# {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[sweep]   {status} in {elapsed/60:.1f} min")
    return proc.returncode


def plan_feature_ablation(args):
    """Run the same base-mode (lopez or mixture) across fcomb1/2.2/2.5."""
    base = MODE_PRESETS[args.base_mode]
    combos = args.feature_combos.split(",") if args.feature_combos else list(FEATURE_COMBOS)
    for combo_name in combos:
        if combo_name not in FEATURE_COMBOS:
            print(f"[sweep] WARN: unknown feature combo '{combo_name}', skipping")
            continue
        feat_cfg = FEATURE_COMBOS[combo_name]
        run_cfg = {**base, **feat_cfg}
        if str(run_cfg.get("use_mixture_token", "False")).lower() == "true":
            run_cfg["mixture_K"] = args.mixture_K
            run_cfg["mixture_cov_type"] = args.mixture_cov_type
        tag = slugify([args.base_mode, combo_name, f"seed{args.seed}"])
        yield tag, run_cfg


def plan_mixture_hparam(args):
    """Grid sweep over K × cov_type × dropout for mixture mode (fcomb2.5)."""
    base = dict(MODE_PRESETS["mixture"])
    feat_cfg = FEATURE_COMBOS["fcomb2.5"]
    base.update(feat_cfg)

    Ks         = [int(x) for x in args.K_grid.split(",")]
    covs       = [x.strip() for x in args.cov_grid.split(",")]
    dropouts   = [float(x) for x in args.dropout_grid.split(",")]
    proj_hids  = [int(x) for x in args.proj_hidden_grid.split(",")]

    for K, cov, dr, ph in itertools.product(Ks, covs, dropouts, proj_hids):
        run_cfg = {
            **base,
            "mixture_K": K,
            "mixture_cov_type": cov,
            "mixture_dropout": dr,
            "mixture_proj_hidden": ph,
        }
        tag = slugify(["mixture_hparam", f"K{K}", cov, f"do{dr}", f"ph{ph}",
                       f"seed{args.seed}"])
        yield tag, run_cfg


def plan_seeds(args):
    """Repeat a single (mode, feature combo, hparam) across multiple seeds."""
    base = dict(MODE_PRESETS[args.base_mode])
    feat_cfg = FEATURE_COMBOS[args.feature_combo]
    base.update(feat_cfg)
    if str(base.get("use_mixture_token", "False")).lower() == "true":
        base["mixture_K"] = args.mixture_K
        base["mixture_cov_type"] = args.mixture_cov_type
    seeds = [int(s) for s in args.seeds.split(",")]
    for s in seeds:
        run_cfg = {**base, "seed": s}
        tag = slugify([args.base_mode, args.feature_combo, f"seed{s}"])
        yield tag, run_cfg


def main():
    p = argparse.ArgumentParser(description="Ablation / hyperparameter sweep runner")

    p.add_argument("--sweep", required=True,
                   choices=["feature_ablation", "mixture_hparam", "seeds"],
                   help="Which sweep to run")

    p.add_argument("--dataset", default="OpenAssistant/oasst1")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--train_epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler_type", default="cosine_with_min_lr")
    p.add_argument("--min_lr_ratio", type=float, default=0.7)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--fp_dropout", default="0.1,0.3")
    p.add_argument("--gradient_acum_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--max_length", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_rewardbench", action="store_true")

    p.add_argument("--out_dir", default="./sweep_results")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--continue_on_fail", action="store_true",
                   help="Continue sweep even if a run fails")

    p.add_argument("--base_mode", default="mixture",
                   choices=list(MODE_PRESETS),
                   help="Base mode for feature_ablation / seeds sweep")
    p.add_argument("--feature_combos", default="",
                   help="Comma-separated subset of feature combos for "
                        "feature_ablation (e.g. 'fcomb1,fcomb2.2,fcomb2.5'). "
                        "Default: all four.")

    p.add_argument("--mixture_K", type=int, default=3,
                   help="K used in feature_ablation / seeds when mixture is on")
    p.add_argument("--mixture_cov_type", default="diag",
                   help="cov type used in feature_ablation / seeds when mixture is on")

    p.add_argument("--K_grid", default="2,3,4,5")
    p.add_argument("--cov_grid", default="diag,full")
    p.add_argument("--dropout_grid", default="0.05,0.1,0.2")
    p.add_argument("--proj_hidden_grid", default="64,128,256")

    p.add_argument("--feature_combo", default="fcomb2.5",
                   choices=list(FEATURE_COMBOS),
                   help="Feature combo for seeds sweep")
    p.add_argument("--seeds", default="42,123,2024",
                   help="Comma-separated seed list for seeds sweep")

    args = p.parse_args()

    if not MAIN_PY.exists():
        print(f"[sweep] FATAL: {MAIN_PY} not found", file=sys.stderr)
        sys.exit(2)

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep == "feature_ablation":
        runs = list(plan_feature_ablation(args))
    elif args.sweep == "mixture_hparam":
        runs = list(plan_mixture_hparam(args))
    else:
        runs = list(plan_seeds(args))

    print(f"[sweep] planned {len(runs)} runs → {out_dir}")
    manifest = {
        "sweep": args.sweep,
        "started": datetime.utcnow().isoformat() + "Z",
        "common": {
            "dataset": args.dataset,
            "model": args.model,
            "batch_size": args.batch_size,
            "train_epochs": args.train_epochs,
            "learning_rate": args.learning_rate,
        },
        "runs": [{"tag": t, "config": c} for t, c in runs],
    }
    (out_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2))

    failures = []
    for tag, run_cfg in runs:
        rc = run_one(args, run_cfg, tag, out_dir)
        if rc != 0:
            failures.append(tag)
            if not args.continue_on_fail:
                print(f"[sweep] aborting on first failure ({tag}). "
                      f"Use --continue_on_fail to keep going.")
                break

    print(f"[sweep] done. {len(runs) - len(failures)} OK / {len(failures)} FAIL")
    if failures:
        print(f"[sweep] failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()

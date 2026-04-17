import argparse
import gc
import json
import os
import pathlib
import sys

import torch

ROOT = str(pathlib.Path(__file__).parent.resolve())
sys.path.insert(0, ROOT)


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_mixture_meta(ckpt_dir):
    mp = os.path.join(ckpt_dir, "mixture_module.bin")
    if not os.path.isfile(mp):
        return None
    state = torch.load(mp, map_location="cpu")
    return {
        "num_params": sum(v.numel() for v in state.values()),
        "keys_sample": list(state.keys())[:3],
    }


CHAT_KEYS = [
    "alpacaeval-easy",
    "alpacaeval-length",
    "alpacaeval-hard",
    "mt-bench-easy",
    "mt-bench-med",
]
CHATHARD_KEYS = [
    "mt-bench-hard",
    "llmbar-natural",
    "llmbar-adver-neighbor",
    "llmbar-adver-GPTInst",
    "llmbar-adver-GPTOut",
    "llmbar-adver-manual",
]
SAFETY_KEYS = [
    "refusals-dangerous",
    "refusals-offensive",
    "xstest-should-refuse",
    "xstest-should-respond",
    "donotanswer",
]
REASONING_KEYS = [
    "math-prm",
    "hep-cpp",
    "hep-go",
    "hep-java",
    "hep-js",
    "hep-python",
    "hep-rust",
]
ALL_KNOWN_KEYS = CHAT_KEYS + CHATHARD_KEYS + SAFETY_KEYS + REASONING_KEYS


def wavg(accs, sizes, keys):
    s, w = 0.0, 0
    for k in keys:
        n = sizes.get(k, 0)
        if k in accs and n > 0:
            s += accs[k] * n
            w += n
    return (s / w) if w > 0 else 0.0


def _extract_acc(res):
    if not isinstance(res, dict):
        return None
    for k in ("eval_accuracy", "accuracy"):
        v = res.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def print_results(accs, subset_counts, label, missing=None):
    print("\n" + "=" * 60)
    print(f"  REWARDBENCH RESULTS — {label}")
    print("=" * 60)
    print(f"  Rows in dataset: {sum(subset_counts.values())}")
    print(f"  Subsets evaluated: {len(accs)}")
    print()
    for k in sorted(accs):
        print(f"  {k:<35s} {accs[k] * 100:>6.1f}%")
    if missing:
        print()
        print(f"  MISSING (in dataset but not evaluated): {sorted(missing)}")
    print()
    chat = wavg(accs, subset_counts, CHAT_KEYS)
    chat_hard = wavg(accs, subset_counts, CHATHARD_KEYS)
    safety = wavg(accs, subset_counts, SAFETY_KEYS)
    reasoning = wavg(accs, subset_counts, REASONING_KEYS)
    print(f"  {'Chat':<20s} {chat * 100:.1f}%")
    print(f"  {'Chat Hard':<20s} {chat_hard * 100:.1f}%")
    print(f"  {'Safety':<20s} {safety * 100:.1f}%")
    print(f"  {'Reasoning':<20s} {reasoning * 100:.1f}%")
    overall = (chat + chat_hard + safety + reasoning) / 4
    print(f"  {'Overall (4-cat)':<20s} {overall * 100:.1f}%")
    print("=" * 60)


def run_rewardbench(
    ckpt_dir,
    max_length=10000,
    batch_size=8,
    hf_token=None,
    rb_char_filter=100000,
):
    print(f"\n[run_rewardbench] ckpt: {ckpt_dir}")

    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

    args_path = os.path.join(ckpt_dir, "args.json")
    assert os.path.isfile(args_path), f"args.json not found in {ckpt_dir}"
    args = json.load(open(args_path))

    base_model = args["model_name"]
    concat = str(args["concat"]).lower() == "true"
    use_softprompt = str(args["use_softprompt"]).lower() == "true"
    fmv = int(args["fixations_model_version"])
    features_used = [int(x) for x in str(args["features_used"]).split(",")]
    seed = int(args["seed"])
    fp_dropout = [float(x) for x in str(args["fp_dropout"]).split(",")]
    max_tokens = 1350 if fmv == 2 else None
    use_mixture_token = str(args.get("use_mixture_token", "false")).lower() == "true"
    mixture_K = int(args.get("mixture_K", 3))
    mixture_cov_type = str(args.get("mixture_cov_type", "diag")).lower()
    mixture_proj_hidden = int(args.get("mixture_proj_hidden", 128))
    mixture_dropout = float(args.get("mixture_dropout", 0.1))
    mixture_log_transform = (
        str(args.get("mixture_log_transform", "true")).lower() == "true"
    )

    print(f"  model={base_model}  fmv={fmv}  features={features_used}")
    print(f"  use_mixture_token={use_mixture_token}")
    print(
        f"  rb_char_filter={rb_char_filter} (chars), "
        f"tokenizer max_length={max_length}"
    )

    if use_mixture_token:
        meta = load_mixture_meta(ckpt_dir)
        if meta:
            print(f"  mixture K={mixture_K}, cov_type={mixture_cov_type}")
            print(f"  mixture_module params: {meta['num_params']}")
        else:
            print("  mixture_module.bin not found — using freshly initialized projector")

    rt = os.path.join(ckpt_dir, "results_dataset_test.json")
    if os.path.isfile(rt):
        r = json.load(open(rt))
        if "eval_accuracy" in r:
            print(f"  OASST1 test acc (from training): {r['eval_accuracy'] * 100:.1f}%")

    from transformers import set_seed

    set_seed(seed)

    from rlhf_rw.trainers.reward_trainer_general import RewardTrainerConstructorGeneral

    trainer = RewardTrainerConstructorGeneral(
        model_name=base_model,
        dataset_name="allenai/reward-bench",
        use_lora=True,
        use_quantization=True,
        concat=concat,
        use_softprompt=use_softprompt,
        batch_size=batch_size,
        fp_dropout=fp_dropout,
        fixations_model_version=fmv,
        features_used=features_used,
        seed=seed,
        load_fix_model=True,
        max_tokens=max_tokens,
        max_length=rb_char_filter,
        use_mixture_token=use_mixture_token,
        mixture_K=mixture_K,
        mixture_cov_type=mixture_cov_type,
        mixture_proj_hidden=mixture_proj_hidden,
        mixture_dropout=mixture_dropout,
        mixture_log_transform=mixture_log_transform,
    )

    print("  running RewardBench eval...")
    results = trainer.eval_model(folder_name=ckpt_dir, mode="all")
    print("  eval done")

    subset_counts = {}
    test_data = trainer.dataset_procesor.data["test"]
    for s in test_data["subset"]:
        subset_counts[s] = subset_counts.get(s, 0) + 1

    accs = {}
    if isinstance(results, dict):
        for subset, res in results.items():
            acc = _extract_acc(res)
            if acc is not None:
                accs[subset] = acc

    missing = [s for s in subset_counts if s not in accs]
    unknown = [s for s in accs if s not in ALL_KNOWN_KEYS]

    if missing:
        print(
            f"\n[run_rewardbench] WARNING: "
            f"{len(missing)} subsets in dataset were not evaluated: {sorted(missing)}"
        )
        print(
            "  Likely causes: (1) char-length filter dropped all rows "
            "(raise --rb_char_filter), (2) eval crashed on that subset, "
            "(3) empty after preprocessing."
        )
    if unknown:
        print(
            f"\n[run_rewardbench] NOTE: unknown subsets not aggregated into "
            f"any category: {sorted(unknown)}"
        )

    label = os.path.basename(ckpt_dir.rstrip("/"))
    print_results(accs, subset_counts, label, missing=missing)

    out = {
        "accs": accs,
        "subset_counts": subset_counts,
        "missing_subsets": sorted(missing),
        "unknown_subsets": sorted(unknown),
        "chat": wavg(accs, subset_counts, CHAT_KEYS),
        "chat_hard": wavg(accs, subset_counts, CHATHARD_KEYS),
        "safety": wavg(accs, subset_counts, SAFETY_KEYS),
        "reasoning": wavg(accs, subset_counts, REASONING_KEYS),
        "overall": sum(
            [
                wavg(accs, subset_counts, k)
                for k in [CHAT_KEYS, CHATHARD_KEYS, SAFETY_KEYS, REASONING_KEYS]
            ]
        )
        / 4,
        "max_length": max_length,
        "rb_char_filter": rb_char_filter,
        "batch_size": batch_size,
    }
    out_path = os.path.join(ckpt_dir, "results_rewardbench.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved → {out_path}")

    del trainer, results, test_data
    free_gpu()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--max_length", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--rb_char_filter",
        type=int,
        default=100000,
        help="Char-length cap for RewardBench row filter. Set high to avoid "
        "dropping long AlpacaEval rows. Default 100000 keeps all rows.",
    )
    a = parser.parse_args()
    run_rewardbench(
        ckpt_dir=a.ckpt_dir,
        max_length=a.max_length,
        batch_size=a.batch_size,
        hf_token=a.hf_token,
        rb_char_filter=a.rb_char_filter,
    )

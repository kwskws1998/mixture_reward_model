"""H1 sanity check: is per-response ET feature distribution better fit by a
K-component mixture than a single Gaussian? Outputs summary.json + bic_curves.png."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from et2_wrapper import FixationsPredictor_2
from transformers import AutoTokenizer


def sample_oasst_responses(n_responses=100, min_tokens=20, seed=42):
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    rng = np.random.default_rng(seed)
    candidates = [
        r["text"] for r in ds
        if r.get("role") == "assistant"
        and r.get("lang") == "en"
        and len(r["text"].split()) >= min_tokens
    ]
    print(f"[h1] {len(candidates)} eligible responses", end="")
    if n_responses <= 0 or n_responses >= len(candidates):
        print(" — using ALL")
        return candidates
    print(f", sampling {n_responses}")
    idx = rng.choice(len(candidates), size=n_responses, replace=False)
    return [candidates[i] for i in idx]


def extract_features_per_response(responses, fp, tokenizer, device):
    feature_sets = []
    for txt in tqdm(responses, desc="ET predict"):
        enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=1024)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            fixations, _, _, _, _, _ = fp._compute_mapped_fixations(ids, mask)
        feats = fixations[0].cpu().numpy()
        norms = np.linalg.norm(feats, axis=1)
        feats = feats[norms > 1e-6]
        if len(feats) >= 10:
            feature_sets.append(feats)
    print(f"[h1] {len(feature_sets)} responses with sufficient non-zero tokens")
    return feature_sets


def fit_bic_curve(features, k_values, cov_type="diag", seed=0):
    bics = []
    for k in k_values:
        if k > len(features) // 2:
            bics.append(np.nan)
            continue
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=seed,
                reg_covar=1e-4,
                max_iter=200,
                n_init=2,
            )
            gmm.fit(features)
            bics.append(gmm.bic(features))
        except Exception:
            bics.append(np.nan)
    return np.array(bics)


def run_sweep(feature_sets, k_max=8, cov_type="diag"):
    k_values = list(range(1, k_max + 1))
    bic_matrix = []
    for feats in tqdm(feature_sets, desc=f"GMM BIC ({cov_type})"):
        bic_matrix.append(fit_bic_curve(feats, k_values, cov_type=cov_type))
    return np.array(bic_matrix), k_values


def analyze(bic_matrix, k_values, out_dir):
    best_ks = []
    for row in bic_matrix:
        if np.all(np.isnan(row)):
            continue
        best_ks.append(k_values[int(np.nanargmin(row))])
    best_ks = np.array(best_ks)

    counter = Counter(best_ks.tolist())
    n_total = len(best_ks)
    pct_k_ge_2 = 100 * np.mean(best_ks >= 2)

    summary = {
        "n_responses_analyzed": int(n_total),
        "best_k_distribution": {int(k): int(v) for k, v in sorted(counter.items())},
        "pct_responses_with_K_ge_2": float(pct_k_ge_2),
        "mean_best_k": float(np.mean(best_ks)),
        "median_best_k": float(np.median(best_ks)),
    }
    print("\n" + "=" * 60)
    print("H1 SANITY CHECK SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    if pct_k_ge_2 >= 70:
        print(f"  [+] {pct_k_ge_2:.1f}% of responses prefer K>=2 -> H1 SUPPORTED")
        print(f"      Mixture hypothesis is justified. Proceed with method.")
    elif pct_k_ge_2 >= 40:
        print(f"  [~] {pct_k_ge_2:.1f}% of responses prefer K>=2 -> H1 WEAKLY SUPPORTED")
        print(f"      Consider K=2 as default, frame as 'often multimodal'.")
    else:
        print(f"  [-] Only {pct_k_ge_2:.1f}% of responses prefer K>=2 -> H1 NOT SUPPORTED")
        print(f"      Single Gaussian suffices. Reframe paper or try different features.")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        for row in bic_matrix:
            if not np.all(np.isnan(row)):
                ax.plot(k_values, row, color="gray", alpha=0.15, linewidth=0.5)
        mean_bic = np.nanmean(bic_matrix, axis=0)
        ax.plot(k_values, mean_bic, color="C0", linewidth=2.5, marker="o", label="mean BIC")
        ax.set_xlabel("K (number of components)")
        ax.set_ylabel("BIC (lower = better)")
        ax.set_title(f"BIC curves over {n_total} responses")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.bar(sorted(counter.keys()), [counter[k] for k in sorted(counter.keys())],
               color="C1", edgecolor="black")
        ax.set_xlabel("BIC-optimal K")
        ax.set_ylabel("# responses")
        ax.set_title(f"Best K distribution ({pct_k_ge_2:.1f}% prefer K>=2)")
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        out_path = os.path.join(out_dir, "bic_curves.png")
        plt.savefig(out_path, dpi=120)
        print(f"[h1] saved plot: {out_path}")
    except Exception as e:
        print(f"[h1] plot skipped: {e}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.save(os.path.join(out_dir, "bic_matrix.npy"), bic_matrix)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_responses", type=int, default=100)
    p.add_argument("--k_max", type=int, default=8)
    p.add_argument("--cov_type", type=str, default="diag",
                   choices=["full", "diag", "tied", "spherical"])
    p.add_argument("--rm_tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--out_dir", type=str, default="bic_results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_tokens", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[h1] device={device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.rm_tokenizer, trust_remote_code=True)
    except Exception as e:
        print(f"[h1] failed to load {args.rm_tokenizer}: {e}, falling back to gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[h1] loading et2 predictor")
    fp = FixationsPredictor_2(modelTokenizer=tokenizer, remap=False)

    responses = sample_oasst_responses(args.n_responses, args.min_tokens, args.seed)
    feature_sets = extract_features_per_response(responses, fp, tokenizer, device)

    if len(feature_sets) < 10:
        print(f"[h1] WARNING: only {len(feature_sets)} usable responses")

    print(f"[h1] running BIC sweep K=1..{args.k_max}")
    bic_matrix, k_values = run_sweep(feature_sets, k_max=args.k_max, cov_type=args.cov_type)

    analyze(bic_matrix, k_values, args.out_dir)


if __name__ == "__main__":
    main()

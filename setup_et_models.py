"""Set up ET1 (Huang & Hollenstein 2023) + ET2 (Li & Rudzicz 2021) predictors.

Run once:
    python setup_et_models.py

Options:
    --et2-checkpoint PATH   Explicit ET2 checkpoint path (.pt / .safetensors).
                            If omitted, we try: HF cache → HF hub download → fail gracefully.
    --skip-install          Skip pip install (already done by install.sh).
    --clone-dir PATH        Where to clone SelectiveCacheForLM for ET1 weights.
"""

import argparse
import os
import shutil
import subprocess
import sys


ET2_HF_REPO = "skboy/et_prediction_2"
ET2_HF_FILE = "et_predictor2_seed123.safetensors"


def run(cmd, check=True):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=check)


def find_eyetrackpy_root():
    try:
        import eyetrackpy
        return os.path.dirname(eyetrackpy.__file__)
    except ImportError:
        return None


def install_packages():
    print("\n[1/4] Installing eyetrackpy, tokenizer_aligner...")
    run("pip install git+https://github.com/angelalopezcardona/tokenizer_aligner.git@v1.0.0 -q")
    run("pip install git+https://github.com/angelalopezcardona/eyetrackpy.git@v1.0.0 -q")
    run("pip install safetensors -q")


def setup_et_model1(clone_dir="./SelectiveCacheForLM"):
    print("\n[2/4] ET1 weights (Huang & Hollenstein 2023)...")

    et_root = find_eyetrackpy_root()
    if et_root is None:
        raise ImportError("eyetrackpy not installed. Run install.sh first.")

    dst_dir = os.path.join(et_root, "data_generator", "fixations_predictor_trained_1")
    dst = os.path.join(dst_dir, "T5-tokenizer-BiLSTM-TRT-12-concat-3")

    if os.path.isfile(dst):
        print(f"  already installed: {dst}")
        return

    if not os.path.isdir(clone_dir):
        run(f"git clone https://github.com/huangxt39/SelectiveCacheForLM.git {clone_dir}")
    else:
        print(f"  {clone_dir} already present")

    src = os.path.join(clone_dir, "FPmodels", "T5-tokenizer-BiLSTM-TRT-12-concat-3")
    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"ET1 weights not found at: {src}\n"
            "SelectiveCacheForLM repo structure may have changed."
        )

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied: {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")


def _try_hf_cache():
    """Return path to ET2 file in HF cache if already downloaded, else None."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo_id=ET2_HF_REPO, filename=ET2_HF_FILE)
        if p and os.path.isfile(p):
            return p
    except Exception:
        pass
    return None


def _try_hf_download():
    """Download ET2 from HF hub (cached for future use). Returns path or None."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=ET2_HF_REPO, filename=ET2_HF_FILE)
    except Exception as e:
        print(f"  HF download failed: {e}")
        return None


def _try_local_path(path):
    """Resolve user-supplied path with automatic extension search."""
    if not path:
        return None
    for ext in ["", ".safetensors", ".pt", ".bin"]:
        candidate = path + ext if (ext and not path.endswith(ext)) else path
        if os.path.isfile(candidate):
            return candidate
    return None


def setup_et_model2(checkpoint_path=None):
    """Locate ET2 checkpoint via: explicit path → HF cache → HF download.

    Never raises — if all fail, we just print guidance and let et2_wrapper
    handle auto-download at first use.
    """
    print("\n[3/4] ET2 checkpoint (Li & Rudzicz 2021)...")

    resolved = None
    source = None

    # 1. explicit path wins
    if checkpoint_path:
        resolved = _try_local_path(checkpoint_path)
        if resolved:
            source = "explicit path"

    # 2. HF cache (install.sh already populates this)
    if resolved is None:
        resolved = _try_hf_cache()
        if resolved:
            source = "HF cache"

    # 3. HF download
    if resolved is None:
        print(f"  not found locally or in HF cache — attempting download...")
        resolved = _try_hf_download()
        if resolved:
            source = "HF hub (just downloaded)"

    if resolved is None:
        print("  WARNING: ET2 checkpoint could not be set up.")
        print("  This is NOT fatal — et2_wrapper.py will auto-download at first use.")
        print(f"  To preempt, either:")
        print(f"    - Set ET2_CHECKPOINT_PATH to your own checkpoint, OR")
        print(f"    - Check internet access + 'huggingface-cli login'")
        return None

    print(f"  resolved via {source}: {resolved} ({os.path.getsize(resolved)/1e6:.1f} MB)")

    abs_path = os.path.abspath(resolved)
    env_line = f"ET2_CHECKPOINT_PATH={abs_path}"

    with open(".env_et", "w") as f:
        f.write(env_line + "\n")

    print(f"\n  To pin this checkpoint for all sessions, run one of:")
    print(f"    export {env_line}")
    print(f"    source .env_et")

    os.environ["ET2_CHECKPOINT_PATH"] = abs_path
    return abs_path


def verify_setup():
    print("\n[4/4] Verifying setup...")

    try:
        from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import FixationsPredictor_1
        print("  OK   FixationsPredictor_1 import")
    except Exception as e:
        print(f"  FAIL FixationsPredictor_1 import: {e}")

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from et2_wrapper import FixationsPredictor_2  # noqa: F401
        print("  OK   et2_wrapper import")
    except Exception as e:
        print(f"  FAIL et2_wrapper import: {e}")
        print(f"       Check et2_wrapper.py is in the repo root.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--et2-checkpoint",
        default=None,
        help="Explicit ET2 checkpoint path (extension auto-detected). "
             "If omitted, we use HF cache → HF download.",
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip pip install (install.sh already did this)",
    )
    parser.add_argument(
        "--clone-dir", default="./SelectiveCacheForLM",
        help="Where to clone SelectiveCacheForLM (for ET1 weights)",
    )
    args = parser.parse_args()

    if not args.skip_install:
        install_packages()

    setup_et_model1(args.clone_dir)
    setup_et_model2(args.et2_checkpoint)
    verify_setup()

    print("\nDone. Next steps:")
    print("  python scripts/h1_bic_sanity_check.py   # FIRST: verify H1")
    print("  bash run_modes.sh mixture               # mixture training")


if __name__ == "__main__":
    main()

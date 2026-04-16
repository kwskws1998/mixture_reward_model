"""
run_rewardbench.py
==================
RewardBench eval — runs automatically after training, or standalone.

Usage (standalone):
    python run_rewardbench.py \
        --ckpt_dir ./models_save/path/to/checkpoint \
        --hf_token hf_xxx \
        --max_length 10000 \
        --batch_size 8

Usage (auto, called from main.py after training):
    Just pass --run_rewardbench True to main.py  ← added below
"""

import os, sys, json, gc, math, argparse, pathlib
import torch
import torch.nn.functional as F

ROOT = str(pathlib.Path(__file__).parent.resolve())
sys.path.insert(0, ROOT)

# ── helpers ──────────────────────────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_mixture_meta(ckpt_dir):
    """Returns dict with mixture-token metadata if present, else None."""
    mp = os.path.join(ckpt_dir, 'mixture_module.bin')
    if not os.path.isfile(mp):
        return None
    state = torch.load(mp, map_location='cpu')
    return {
        'num_params': sum(v.numel() for v in state.values()),
        'keys_sample': list(state.keys())[:3],
    }

CHAT_KEYS     = ['alpacaeval-easy','alpacaeval-length','alpacaeval-hard','mt-bench-easy','mt-bench-med']
CHATHARD_KEYS = ['mt-bench-hard','llmbar-natural','llmbar-adver-neighbor',
                 'llmbar-adver-GPTInst','llmbar-adver-GPTOut','llmbar-adver-manual']
SAFETY_KEYS   = ['refusals-dangerous','refusals-offensive',
                 'xstest-should-refuse','xstest-should-respond','donotanswer']
CODE_KEYS     = ['hep-cpp','hep-go','hep-java','hep-js','hep-python','hep-rust']

def wavg(accs, sizes, keys):
    s, w = 0.0, 0
    for k in keys:
        n = sizes.get(k, 0)
        if k in accs and n > 0:
            s += accs[k] * n; w += n
    return (s / w) if w > 0 else 0.0

def print_results(accs, subset_counts, label):
    print('\n' + '='*60)
    print(f'  REWARDBENCH RESULTS — {label}')
    print('='*60)
    print(f'  Rows evaluated: {sum(subset_counts.values())}')
    print()

    # per-subset
    for k in sorted(accs):
        print(f'  {k:<35s} {accs[k]*100:>6.1f}%')

    print()
    print(f'  {"Chat":<20s} {wavg(accs, subset_counts, CHAT_KEYS)*100:.1f}%')
    print(f'  {"Chat Hard":<20s} {wavg(accs, subset_counts, CHATHARD_KEYS)*100:.1f}%')
    print(f'  {"Safety":<20s} {wavg(accs, subset_counts, SAFETY_KEYS)*100:.1f}%')
    print(f'  {"Code":<20s} {wavg(accs, subset_counts, CODE_KEYS)*100:.1f}%')

    cats = [wavg(accs, subset_counts, k) for k in
            [CHAT_KEYS, CHATHARD_KEYS, SAFETY_KEYS, CODE_KEYS]]
    print(f'  {"Overall (4-cat)":<20s} {sum(cats)/4*100:.1f}%')
    print('='*60)

# ── main eval function ────────────────────────────────────────────────────────

def run_rewardbench(ckpt_dir, max_length=10000, batch_size=8, hf_token=None):
    print(f'\n[run_rewardbench] ckpt: {ckpt_dir}')

    # set HF token if provided
    if hf_token:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        os.environ['HF_TOKEN'] = hf_token

    # load args.json
    args_path = os.path.join(ckpt_dir, 'args.json')
    assert os.path.isfile(args_path), f'args.json not found in {ckpt_dir}'
    args = json.load(open(args_path))

    base_model          = args['model_name']
    concat              = str(args['concat']).lower() == 'true'
    use_softprompt      = str(args['use_softprompt']).lower() == 'true'
    fmv                 = int(args['fixations_model_version'])
    features_used       = [int(x) for x in str(args['features_used']).split(',')]
    seed                = int(args['seed'])
    fp_dropout          = [float(x) for x in str(args['fp_dropout']).split(',')]
    max_tokens          = 1350 if fmv == 2 else None
    use_mixture_token   = str(args.get('use_mixture_token', 'false')).lower() == 'true'
    mixture_K           = int(args.get('mixture_K', 3))
    mixture_cov_type    = str(args.get('mixture_cov_type', 'diag')).lower()
    mixture_proj_hidden = int(args.get('mixture_proj_hidden', 128))
    mixture_dropout     = float(args.get('mixture_dropout', 0.1))
    mixture_log_transform = str(args.get('mixture_log_transform', 'true')).lower() == 'true'

    print(f'  model={base_model}  fmv={fmv}  features={features_used}')
    print(f'  use_mixture_token={use_mixture_token}')

    # print mixture-module metadata if saved
    if use_mixture_token:
        meta = load_mixture_meta(ckpt_dir)
        if meta:
            print(f'  mixture K={mixture_K}, cov_type={mixture_cov_type}')
            print(f'  mixture_module params: {meta["num_params"]}')
        else:
            print(f'  mixture_module.bin not found — using freshly initialized projector')

    # show OASST1 result from training if available
    rt = os.path.join(ckpt_dir, 'results_dataset_test.json')
    if os.path.isfile(rt):
        r = json.load(open(rt))
        if 'eval_accuracy' in r:
            print(f'  OASST1 test acc (from training): {r["eval_accuracy"]*100:.1f}%')

    from transformers import set_seed
    set_seed(seed)

    from rlhf_rw.trainers.reward_trainer_general import RewardTrainerConstructorGeneral

    trainer = RewardTrainerConstructorGeneral(
        model_name              = base_model,
        dataset_name            = 'allenai/reward-bench',
        use_lora                = True,
        use_quantization        = True,
        concat                  = concat,
        use_softprompt          = use_softprompt,
        batch_size              = batch_size,
        fp_dropout              = fp_dropout,
        fixations_model_version = fmv,
        features_used           = features_used,
        seed                    = seed,
        load_fix_model          = True,
        max_tokens              = max_tokens,
        max_length              = max_length,
        use_mixture_token       = use_mixture_token,
        mixture_K               = mixture_K,
        mixture_cov_type        = mixture_cov_type,
        mixture_proj_hidden     = mixture_proj_hidden,
        mixture_dropout         = mixture_dropout,
        mixture_log_transform   = mixture_log_transform,
    )

    print('  running RewardBench eval...')
    results = trainer.eval_model(folder_name=ckpt_dir, mode='all')
    print('  eval done')

    # collect subset counts and accuracies
    subset_counts = {}
    test_data = trainer.dataset_procesor.data['test']
    for s in test_data['subset']:
        subset_counts[s] = subset_counts.get(s, 0) + 1

    accs = {}
    if isinstance(results, dict):
        for subset, res in results.items():
            if isinstance(res, dict):
                acc = res.get('eval_accuracy') or res.get('accuracy')
                if acc is not None:
                    accs[subset] = float(acc)

    label = os.path.basename(ckpt_dir.rstrip('/'))
    print_results(accs, subset_counts, label)

    # save results next to checkpoint
    out = {
        'accs': accs,
        'subset_counts': subset_counts,
        'chat':      wavg(accs, subset_counts, CHAT_KEYS),
        'chat_hard': wavg(accs, subset_counts, CHATHARD_KEYS),
        'safety':    wavg(accs, subset_counts, SAFETY_KEYS),
        'code':      wavg(accs, subset_counts, CODE_KEYS),
        'overall':   sum([wavg(accs, subset_counts, k) for k in
                          [CHAT_KEYS, CHATHARD_KEYS, SAFETY_KEYS, CODE_KEYS]]) / 4,
        'max_length': max_length,
        'batch_size': batch_size,
    }
    out_path = os.path.join(ckpt_dir, 'results_rewardbench.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'  saved → {out_path}')

    del trainer, results, test_data
    free_gpu()

    return out

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',   required=True, help='path to saved checkpoint folder')
    parser.add_argument('--hf_token',   default=None,  help='HuggingFace token (if needed)')
    parser.add_argument('--max_length', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    run_rewardbench(
        ckpt_dir   = args.ckpt_dir,
        max_length = args.max_length,
        batch_size = args.batch_size,
        hf_token   = args.hf_token,
    )

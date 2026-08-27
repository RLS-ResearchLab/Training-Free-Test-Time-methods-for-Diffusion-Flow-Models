"""
Regenerates results/large_scale_eval.png from EXISTING .eval_checkpoints/*.pkl
files, without recomputing any generation. Use this after fixing a plotting
bug in make_figure() (label overlap, title, etc.) when the underlying
technique results are already checkpointed and correct.

Usage:
    python scripts/regenerate_figure.py \
        --real_dir data/imagenet_val \
        --n 1000 \
        --techniques cfg auto_guidance_latent_scale auto_guidance_weight_noise \
                     auto_guidance_fewer_timesteps fk_steering best_of_n \
        --n_fk_steering 100 --n_best_of_n 100 \
        --out results/large_scale_eval.png

--n / --n_fk_steering / --n_best_of_n must match the N each checkpoint was
saved with (same as your original eval_large_scale.py run), since checkpoint
filenames are keyed on technique+n+seed.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

from eval_large_scale import make_figure, TECHNIQUE_LABELS
from eval_fid_clip import _load_inception, _inception_features, sample_real_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--n", type=int, required=True, help="N used for cfg/auto_guidance_* checkpoints")
    ap.add_argument("--n_fk_steering", type=int, default=None)
    ap.add_argument("--n_best_of_n", type=int, default=None)
    ap.add_argument("--techniques", nargs="+", default=list(TECHNIQUE_LABELS.keys()))
    ap.add_argument("--n_grid", type=int, default=4)
    ap.add_argument("--img_cols", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint_dir", default=".eval_checkpoints")
    ap.add_argument("--feat_batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/large_scale_eval.png")
    args = ap.parse_args()

    def n_for(technique):
        if technique == "fk_steering" and args.n_fk_steering is not None:
            return args.n_fk_steering
        if technique == "best_of_n" and args.n_best_of_n is not None:
            return args.n_best_of_n
        return args.n

    results = []
    for technique in args.techniques:
        ckpt = Path(args.checkpoint_dir) / f"{technique}_n{n_for(technique)}_seed{args.seed}.pkl"
        if not ckpt.exists():
            print(f"[SKIP] {technique}: no checkpoint at {ckpt}")
            continue
        with open(ckpt, "rb") as f:
            results.append(pickle.load(f))
        print(f"[loaded] {technique} from {ckpt}")

    if not results:
        print("No checkpoints found — nothing to plot. Check --checkpoint_dir / --n values.")
        return

    print(f"[setup] Loading Inception and sampling real images from {args.real_dir} for FID reference")
    inception, use_pytorch_fid = _load_inception(args.device)
    real_images = sample_real_images(args.real_dir, args.n, seed=args.seed)
    real_feats = _inception_features(inception, use_pytorch_fid, real_images, args.device,
                                      batch_size=args.feat_batch_size)
    del real_images

    make_figure(results, real_feats, args.out, args.n_grid, img_cols=args.img_cols)


if __name__ == "__main__":
    main()
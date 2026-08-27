"""
Populates a local folder of real reference images for FID computation
(the --real_dir expected by scripts/eval_fid_clip.py and
scripts/eval_large_scale.py).

Streams images from the ImageNet-1k validation split (or any other HF
dataset with an "image" column) and writes them to --out_dir as JPEGs.
Streaming means the full dataset is never downloaded/cached at once —
only the --n images actually saved pass through.

One-time setup (only needed once per machine/account):
    1. Accept the ImageNet-1k terms on the dataset page:
       https://huggingface.co/datasets/ILSVRC/imagenet-1k
       (the old "imagenet-1k" URL/repo id was retired — it now lives
       under the ILSVRC org, which is why --dataset_id defaults there)
    2. huggingface-cli login
       (or pass --hf_token, or set the HF_TOKEN environment variable)

Usage:
    python scripts/prepare_imagenet_val.py --out_dir data/imagenet_val --n 10000

    # then:
    python scripts/eval_large_scale.py --real_dir data/imagenet_val --n 10000 ...
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="data/imagenet_val")
    ap.add_argument("--n", type=int, default=10000, help="number of real images to save")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--dataset_id", default="ILSVRC/imagenet-1k",
                     help="any HF dataset with an 'image' column works, not just ImageNet")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_buffer", type=int, default=10000,
                     help="streaming shuffle buffer size (memory vs. shuffle-quality trade-off)")
    ap.add_argument("--hf_token", default=None,
                     help="only needed if `huggingface-cli login` wasn't already run on this machine")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/2] Streaming '{args.dataset_id}' ({args.split} split) from the Hugging Face Hub")
    ds = load_dataset(
        args.dataset_id,
        split=args.split,
        streaming=True,
        token=args.hf_token if args.hf_token else True,  # True = use the cached CLI login
    )
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    print(f"[2/2] Saving {args.n} images to {out_dir}")
    saved = 0
    log_every = max(1, args.n // 20)
    for example in ds:
        if saved >= args.n:
            break
        image = example.get("image")
        if image is None:
            continue
        image = image.convert("RGB")
        image.save(out_dir / f"real_{saved:06d}.jpg", format="JPEG", quality=95)
        saved += 1
        if saved % log_every == 0:
            print(f"  saved {saved}/{args.n}")

    if saved < args.n:
        print(
            f"\n[WARNING] Only saved {saved}/{args.n} images before the split ran out. "
            f"Either lower --n on eval_fid_clip.py / eval_large_scale.py to match "
            f"({saved}), or pick a larger --dataset_id/--split here."
        )
    else:
        print(f"\n[Done] {saved} real images saved to {out_dir}")
        print(f"Use with: --real_dir {out_dir}")


if __name__ == "__main__":
    main()
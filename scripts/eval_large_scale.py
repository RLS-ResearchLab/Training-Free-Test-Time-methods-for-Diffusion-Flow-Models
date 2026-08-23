"""
Task — large-scale, multi-technique quantitative evaluation.

Runs every requested technique (cfg | auto_guidance | fk_steering) for N
samples each (10k-50k is the intended range), and reports:
  - mean +/- std CLIP score (prompt alignment) over the N samples
  - FID against a real-image reference set
  - average forward passes / wall-clock time per sample (compute proxy)

The ONLY output is a single combined figure (results/large_scale_eval.png
by default): a small grid of example generations per technique on top,
and CLIP / FID / quality-vs-compute bar & scatter panels underneath.
Nothing is written to results/*.json and no samples/<technique>/ folder
of thousands of PNGs is created — every sampler call below passes
save_to_disk=False, and only Inception/CLIP *features* (not the images
themselves) are kept in memory across the run.

Usage:
    python scripts/eval_large_scale.py \
        --model_id stabilityai/stable-diffusion-3.5-medium \
        --techniques cfg auto_guidance fk_steering \
        --real_dir /path/to/real/images \
        --prompts_file scripts/imagenet_prompts.txt \
        --n 10000 \
        --out results/large_scale_eval.png

Notes:
  - Generation is still one sample at a time per technique (the samplers
    in src/sampler.py and src/samplers/fk_steering.py are written around
    batch_size=1 latents), so N=50000 is a genuinely long run — this
    script does not change that, it only fixes what happens to the
    *output* of each run (features + a running mean, not a saved PNG).
  - Inception/CLIP features are accumulated as numpy arrays (2048-d /
    image), not the raw images, which is what keeps memory bounded at
    N=50k (~400MB per technique) instead of growing with saved PNGs.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))  # repo root -> src/
sys.path.append(str(Path(__file__).resolve().parent))         # scripts/  -> sibling imports

from src.model import SD35Wrapper
from src.sampler import ModularSampler
from src.samplers.fk_steering import FKSteeringSampler
from src.rewards.clip_reward import CLIPPromptReward
from src.utils import clip_align_score

from eval_fid_clip import (  # reuse the FID/Inception machinery already written for Task 2
    _load_inception,
    _inception_features,
    _fid_from_features,
    sample_real_images,
    load_prompts,
    _tensor_to_pil,
)

TECHNIQUE_LABELS = {
    "cfg": "CFG",
    "auto_guidance": "CFG + Autoguidance",
    "fk_steering": "FK Steering",
}


def build_sampler(technique, model_wrapper, reward_fn):
    if technique == "fk_steering":
        return FKSteeringSampler(model_wrapper, reward_fn)
    return ModularSampler(model_wrapper)


def sample_one(sampler, technique, prompt, config, run_idx):
    """Runs exactly one generation for `technique`. Never writes the image to disk."""
    kwargs = dict(
        prompt=prompt,
        config=config,
        exp_name=f"large_scale/{technique}",
        save_name=f"tmp_{run_idx}.png",
        save_to_disk=False,
    )
    if technique == "fk_steering":
        return sampler.sample(**kwargs)
    kwargs["methods_override"] = ["cfg"] if technique == "cfg" else ["cfg", "auto_guidance"]
    return sampler.sample(**kwargs)


def evaluate_technique(technique, model_wrapper, reward_fn, inception, use_pytorch_fid,
                        prompts, base_config, n, n_grid, device, seed, feat_batch_size=64):
    sampler = build_sampler(technique, model_wrapper, reward_fn)

    clip_scores = np.empty(n, dtype=np.float64)
    forward_passes = np.empty(n, dtype=np.float64)
    times = np.empty(n, dtype=np.float64)
    feats = np.empty((n, 2048), dtype=np.float32)
    grid_images = []

    buffer_imgs, buffer_start = [], 0

    def flush(end_idx):
        nonlocal buffer_imgs, buffer_start
        if not buffer_imgs:
            return
        batch_feats = _inception_features(
            inception, use_pytorch_fid, buffer_imgs, device, batch_size=len(buffer_imgs)
        )
        feats[buffer_start:end_idx] = batch_feats
        buffer_imgs = []
        buffer_start = end_idx

    t_start = time.time()
    log_every = max(1, n // 20)
    for i in range(n):
        prompt = prompts[i % len(prompts)]
        config = dict(base_config)
        config["prompt"] = prompt
        config["seed"] = seed + i

        image, metrics = sample_one(sampler, technique, prompt, config, i)

        clip_scores[i] = clip_align_score(reward_fn, image, prompt)
        forward_passes[i] = metrics.get("total_forward_passes", np.nan)
        times[i] = metrics.get("time_sec", np.nan)

        pil_img = _tensor_to_pil(image)
        buffer_imgs.append(pil_img)
        if len(grid_images) < n_grid:
            grid_images.append(pil_img)

        if len(buffer_imgs) >= feat_batch_size:
            flush(i + 1)

        if (i + 1) % log_every == 0 or (i + 1) == n:
            print(
                f"  [{technique}] {i + 1}/{n} samples  "
                f"mean CLIP so far={clip_scores[: i + 1].mean():.4f}  "
                f"({time.time() - t_start:.1f}s elapsed)"
            )

    flush(n)

    return {
        "technique": technique,
        "n": n,
        "clip_mean": float(clip_scores.mean()),
        "clip_std": float(clip_scores.std()),
        "forward_passes_mean": float(np.nanmean(forward_passes)),
        "time_mean": float(np.nanmean(times)),
        "feats": feats,
        "grid_images": grid_images,
    }


def make_figure(results, real_feats, out_path, n_grid):
    n_tech = len(results)
    ncols = max(n_grid, 3)
    fids = [_fid_from_features(real_feats, r["feats"]) for r in results]

    fig = plt.figure(figsize=(3.2 * ncols, 3.2 * n_tech + 4.5))
    gs = fig.add_gridspec(n_tech + 1, ncols, height_ratios=[1] * n_tech + [1.4])

    # --- one row of example generations per technique ---
    for row, r in enumerate(results):
        label = TECHNIQUE_LABELS.get(r["technique"], r["technique"])
        for col in range(n_grid):
            ax = fig.add_subplot(gs[row, col])
            if col < len(r["grid_images"]):
                ax.imshow(r["grid_images"][col])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(label, fontsize=12, fontweight="bold")

    # --- bottom row: CLIP bar / FID bar / quality-vs-compute scatter ---
    labels = [TECHNIQUE_LABELS.get(r["technique"], r["technique"]) for r in results]
    clip_means = [r["clip_mean"] for r in results]
    clip_sems = [r["clip_std"] / max(1, r["n"]) ** 0.5 for r in results]

    third = max(1, ncols // 3)
    ax_clip = fig.add_subplot(gs[n_tech, 0:third])
    ax_clip.bar(labels, clip_means, yerr=clip_sems, capsize=4, color="#4C72B0")
    ax_clip.set_title(f"Mean CLIP score (N={results[0]['n']}/technique)")
    ax_clip.tick_params(axis="x", rotation=20)

    ax_fid = fig.add_subplot(gs[n_tech, third:2 * third])
    ax_fid.bar(labels, fids, color="#DD8452")
    ax_fid.set_title("FID vs. real images (lower is better)")
    ax_fid.tick_params(axis="x", rotation=20)

    ax_qc = fig.add_subplot(gs[n_tech, 2 * third:ncols])
    for r in results:
        ax_qc.scatter(r["forward_passes_mean"], r["clip_mean"], s=90)
        ax_qc.annotate(
            TECHNIQUE_LABELS.get(r["technique"], r["technique"]),
            (r["forward_passes_mean"], r["clip_mean"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )
    ax_qc.set_xlabel("Avg. forward passes / sample (compute)")
    ax_qc.set_ylabel("Mean CLIP score (quality)")
    ax_qc.set_title("Quality vs. compute")

    fig.suptitle(
        f"Training-free / test-time methods — large-scale eval (N={results[0]['n']} samples/technique)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"\n[Done] Combined figure saved to '{out_path}' — no JSON, no samples/ dump.")

    print("\n=== SUMMARY ===")
    for r, fid_v in zip(results, fids):
        label = TECHNIQUE_LABELS.get(r["technique"], r["technique"])
        print(
            f"  {label:<22} CLIP={r['clip_mean']:.4f}±{r['clip_std']:.4f}  "
            f"FID={fid_v:.2f}  fwd_passes={r['forward_passes_mean']:.1f}  "
            f"time/sample={r['time_mean']:.2f}s"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--techniques", nargs="+", default=["cfg", "auto_guidance", "fk_steering"],
                     choices=list(TECHNIQUE_LABELS.keys()))
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--prompts_file", default="scripts/imagenet_prompts.txt")
    ap.add_argument("--n", type=int, default=10000, help="samples per technique (10000-50000 typical)")
    ap.add_argument("--n_grid", type=int, default=4, help="example images shown per technique in the figure")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--cfg_scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=1024, help="lower this (e.g. 512) if you hit CUDA OOM")
    ap.add_argument("--width", type=int, default=1024, help="lower this (e.g. 512) if you hit CUDA OOM")
    ap.add_argument("--auto_guidance_scale", type=float, default=3.5)
    ap.add_argument("--auto_guidance_variant", default="latent_scale",
                     choices=["latent_scale", "weight_noise", "fewer_timesteps"])
    ap.add_argument("--num_particles", type=int, default=4, help="FK steering particle count")
    ap.add_argument("--resample_interval", type=int, default=5)
    ap.add_argument("--feat_batch_size", type=int, default=64,
                     help="how many generated images to batch before an Inception forward pass")
    ap.add_argument("--out", default="results/large_scale_eval.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[setup] Loading model '{args.model_id}'")
    model_wrapper = SD35Wrapper(model_id=args.model_id)
    reward_fn = CLIPPromptReward(device=args.device)
    inception, use_pytorch_fid = _load_inception(args.device)

    print(f"[setup] Sampling {args.n} real images from {args.real_dir} for the FID reference set")
    real_images = sample_real_images(args.real_dir, args.n, seed=args.seed)
    real_feats = _inception_features(
        inception, use_pytorch_fid, real_images, args.device, batch_size=args.feat_batch_size
    )
    del real_images  # only the features are needed from here on

    prompts = load_prompts(args.prompts_file, args.n, seed=args.seed)
    base_config = {
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "height": args.height,
        "width": args.width,
        "auto_guidance_scale": args.auto_guidance_scale,
        "auto_guidance_variant": args.auto_guidance_variant,
        "num_particles": args.num_particles,
        "resample_interval": args.resample_interval,
        "negative_prompt": "blurry, low quality, distorted",
    }

    results = []
    for technique in args.techniques:
        print(f"\n[technique] {TECHNIQUE_LABELS.get(technique, technique)} — generating {args.n} samples")
        r = evaluate_technique(
            technique, model_wrapper, reward_fn, inception, use_pytorch_fid,
            prompts, base_config, args.n, args.n_grid, args.device, args.seed,
            feat_batch_size=args.feat_batch_size,
        )
        results.append(r)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_figure(results, real_feats, args.out, args.n_grid)


if __name__ == "__main__":
    main()
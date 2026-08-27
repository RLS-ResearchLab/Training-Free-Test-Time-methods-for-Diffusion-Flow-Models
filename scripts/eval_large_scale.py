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
import math
import pickle
import subprocess
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
from src.samplers.best_of_n import BestOfNSampler
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

# Every technique implemented anywhere in src/ gets its own entry here so it
# shows up as its own bar/scatter point in the comparison figure. The three
# Auto-Guidance "weak model" variants (src/sampler.py -> _get_weaker_prediction)
# are listed as separate techniques rather than folded into one "auto_guidance"
# bucket picked by a single --auto_guidance_variant flag, so all three are
# visible side-by-side in the same run.
def _gpu_util_string():
    """
    Lightweight GPU utilization/memory snapshot via `nvidia-smi`, appended
    to the periodic progress log. This is what makes GPU load visible
    without needing an interactive tool like nvtop running alongside —
    nvtop itself is a terminal UI, not something a script can query, so
    it's still worth running in a separate pane for a live view, but this
    gives you the same core numbers inline in the log you're already
    watching (or piping/logging to a file for later review).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return ""
        util, mem_used, mem_total = out.stdout.strip().split(", ")
        return f"  [GPU {util}% util, {mem_used}/{mem_total} MiB]"
    except Exception:
        return ""  # nvidia-smi not found / no permission / timeout — don't break logging over this


TECHNIQUE_LABELS = {
    "cfg": "CFG",
    "auto_guidance_latent_scale": "CFG + AutoGuidance (latent scale)",
    "auto_guidance_weight_noise": "CFG + AutoGuidance (weight noise)",
    "auto_guidance_fewer_timesteps": "CFG + AutoGuidance (fewer timesteps)",
    "fk_steering": "FK Steering",
    "best_of_n": "Best-of-N",
}

# techniques whose config needs auto_guidance_variant forced, keyed by technique name
_AUTO_GUIDANCE_VARIANT_OF = {
    "auto_guidance_latent_scale": "latent_scale",
    "auto_guidance_weight_noise": "weight_noise",
    "auto_guidance_fewer_timesteps": "fewer_timesteps",
}


def build_sampler(technique, model_wrapper, reward_fn, best_of_n_n):
    if technique == "fk_steering":
        return FKSteeringSampler(model_wrapper, reward_fn)
    if technique == "best_of_n":
        # best-of-N composes on top of plain CFG generations here (the point
        # of this run is to compare best_of_n against the other *test-time*
        # techniques on equal footing, not to stack it on top of them too)
        return BestOfNSampler(model_wrapper, base_methods=["cfg"], n=best_of_n_n, reward_fn=reward_fn)
    return ModularSampler(model_wrapper)


_BATCHABLE_TECHNIQUES = {"cfg", "auto_guidance_latent_scale", "auto_guidance_weight_noise", "auto_guidance_fewer_timesteps"}


def sample_one(sampler, technique, prompt, config, run_idx, steps_override=None):
    """Runs one generation for `technique`. `prompt` may be a str (batch_size=1)
    or a list of str (batched — only cfg/auto_guidance_* support this; fk_steering
    and best_of_n always run one prompt at a time, they already batch internally
    via particles/candidates). Never writes images to disk.

    steps_override: if set, overrides config["num_inference_steps"] for just this
    call — lets fk_steering/best_of_n run at a different step count than
    cfg/auto_guidance_* without touching the shared base_config (and therefore
    without invalidating cfg/auto_guidance_* checkpoints, which are keyed only
    on technique+n+seed, not on steps — so keep the steps you originally
    checkpointed those with if you want them to keep resolving as a hit)."""
    config = dict(config)
    if technique in _AUTO_GUIDANCE_VARIANT_OF:
        config["auto_guidance_variant"] = _AUTO_GUIDANCE_VARIANT_OF[technique]
    if steps_override is not None:
        config["num_inference_steps"] = steps_override

    kwargs = dict(
        prompt=prompt,
        config=config,
        exp_name=f"large_scale/{technique}",
        save_name=f"tmp_{run_idx}.png",
        save_to_disk=False,
    )
    if technique in ("fk_steering", "best_of_n"):
        assert not isinstance(prompt, list), f"{technique} does not support external batching"
        return sampler.sample(**kwargs)
    if technique == "cfg":
        kwargs["methods_override"] = ["cfg"]
    else:  # any auto_guidance_* variant
        kwargs["methods_override"] = ["cfg", "auto_guidance"]
    return sampler.sample(**kwargs)


def evaluate_technique(technique, model_wrapper, reward_fn, inception, use_pytorch_fid,
                        prompts, base_config, n, n_grid, device, seed, feat_batch_size=64,
                        best_of_n_n=5, batch_size=1, steps_override=None):
    sampler = build_sampler(technique, model_wrapper, reward_fn, best_of_n_n)
    # fk_steering / best_of_n don't support external batching (see sample_one) —
    # silently fall back to batch_size=1 for them rather than erroring, since a
    # single --batch_size flag is applied across all --techniques in one run.
    effective_batch_size = batch_size if technique in _BATCHABLE_TECHNIQUES else 1

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
    i = 0
    while i < n:
        chunk = min(effective_batch_size, n - i)
        batch_prompts = [prompts[(i + j) % len(prompts)] for j in range(chunk)]
        config = dict(base_config)
        config["seed"] = seed + i

        prompt_arg = batch_prompts if chunk > 1 else batch_prompts[0]
        images, metrics = sample_one(sampler, technique, prompt_arg, config, i, steps_override=steps_override)
        # images: [chunk, C, H, W] if batched, else a single-item tensor — normalize to a list
        image_list = [images[j:j + 1] for j in range(chunk)] if chunk > 1 else [images]

        # total_forward_passes/time_sec from `metrics` cover the whole chunk;
        # divide evenly across the chunk's samples for the per-sample arrays.
        fwd_per_sample = metrics.get("total_forward_passes", np.nan) / chunk
        time_per_sample = metrics.get("time_sec", np.nan) / chunk

        for j in range(chunk):
            idx = i + j
            clip_scores[idx] = clip_align_score(reward_fn, image_list[j], batch_prompts[j])
            forward_passes[idx] = fwd_per_sample
            times[idx] = time_per_sample

            pil_img = _tensor_to_pil(image_list[j])
            buffer_imgs.append(pil_img)
            if len(grid_images) < n_grid:
                grid_images.append(pil_img)

        if len(buffer_imgs) >= feat_batch_size:
            flush(i + chunk)

        i += chunk
        if i % log_every < effective_batch_size or i == n:
            gpu_util_str = _gpu_util_string()
            print(
                f"  [{technique}] {i}/{n} samples  "
                f"mean CLIP so far={clip_scores[: i].mean():.4f}  "
                f"({time.time() - t_start:.1f}s elapsed)"
                f"{gpu_util_str}"
            )
            # Periodic cache clear — long runs (hundreds-thousands of samples)
            # otherwise let PyTorch's caching allocator fragment over time,
            # which is what caused OOMs partway through a 1000-sample run
            # even though peak per-sample usage never actually grew.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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


def make_figure(results, real_feats, out_path, n_grid, img_cols=8):
    """
    n_grid: how many example images per technique to lay out (pass the same
        value as --n to show every generated image — practical for small
        smoke-test runs like N=20; NOT recommended above a few hundred,
        since matplotlib can't reasonably render thousands of subplots in
        one figure — metrics are always computed over the FULL N regardless
        of how many images are drawn here).
    img_cols: images wrap onto a new row after this many columns, so the
        image block stays a readable grid instead of one endless row.
    """
    n_tech = len(results)
    fids = [_fid_from_features(real_feats, r["feats"]) for r in results]

    if n_grid > 200:
        print(
            f"\n[WARNING] --n_grid={n_grid} will render {n_grid * n_tech} subplots — "
            f"this may be slow and hard to read. Consider a smaller --n_grid "
            f"(e.g. 16-50) for large-N runs; metrics still use the full N either way."
        )

    img_cols = max(1, min(img_cols, n_grid)) if n_grid > 0 else 1
    img_rows_per_tech = math.ceil(n_grid / img_cols) if n_grid > 0 else 0
    total_img_rows = n_tech * img_rows_per_tech
    ncols = max(img_cols, 3)
    total_rows = total_img_rows + 1  # +1 for the bottom metrics row

    fig = plt.figure(figsize=(3.0 * ncols, 2.6 * max(total_img_rows, 1) + 4.5))
    gs = fig.add_gridspec(total_rows, ncols, height_ratios=[1] * total_img_rows + [1.4])

    # --- image grid: every technique gets img_rows_per_tech rows, wrapping at img_cols ---
    row_cursor = 0
    for r in results:
        label = TECHNIQUE_LABELS.get(r["technique"], r["technique"])
        images = r["grid_images"]
        for ridx in range(img_rows_per_tech):
            for cidx in range(img_cols):
                idx = ridx * img_cols + cidx
                ax = fig.add_subplot(gs[row_cursor, cidx])
                if idx < len(images):
                    ax.imshow(images[idx])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if cidx == 0 and ridx == 0:
                    ax.set_ylabel(label, fontsize=12, fontweight="bold")
            row_cursor += 1

    # --- bottom row: CLIP bar / FID bar / quality-vs-compute scatter ---
    # append n= to each label when techniques don't all share the same N,
    # so the figure doesn't silently imply a uniform sample count
    n_values = {r["n"] for r in results}
    if len(n_values) > 1:
        labels = [f'{TECHNIQUE_LABELS.get(r["technique"], r["technique"])} (N={r["n"]})' for r in results]
    else:
        labels = [TECHNIQUE_LABELS.get(r["technique"], r["technique"]) for r in results]
    clip_means = [r["clip_mean"] for r in results]
    clip_sems = [r["clip_std"] / max(1, r["n"]) ** 0.5 for r in results]

    third = max(1, ncols // 3)
    ax_clip = fig.add_subplot(gs[total_img_rows, 0:third])
    ax_clip.bar(labels, clip_means, yerr=clip_sems, capsize=4, color="#4C72B0")
    ax_clip.set_title(f"Mean CLIP score (N varies per technique — see labels)")
    ax_clip.tick_params(axis="x", rotation=20)

    ax_fid = fig.add_subplot(gs[total_img_rows, third:2 * third])
    ax_fid.bar(labels, fids, color="#DD8452")
    ax_fid.set_title("FID vs. real images (lower is better)")
    ax_fid.tick_params(axis="x", rotation=20)

    ax_qc = fig.add_subplot(gs[total_img_rows, 2 * third:ncols])
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

    n_summary = ", ".join(f"{r['technique']}={r['n']}" for r in results) if len(n_values) > 1 else str(results[0]['n'])
    fig.suptitle(
        f"Training-free / test-time methods — large-scale eval (N: {n_summary})",
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
    ap.add_argument("--techniques", nargs="+", default=list(TECHNIQUE_LABELS.keys()),
                     choices=list(TECHNIQUE_LABELS.keys()),
                     help="which techniques to run; defaults to ALL techniques in the repo "
                          "(cfg, all 3 auto_guidance variants, fk_steering, best_of_n)")
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--prompts_file", default="scripts/imagenet_prompts.txt")
    ap.add_argument("--n", type=int, default=10000, help="samples per technique for cfg/auto_guidance_* "
                     "(10000-50000 typical). fk_steering/best_of_n use --n_fk_steering/--n_best_of_n "
                     "instead if set, since those two are far more expensive per sample and often need "
                     "a smaller N to stay tractable.")
    ap.add_argument("--n_fk_steering", type=int, default=None,
                     help="samples for fk_steering specifically; falls back to --n if not set")
    ap.add_argument("--n_best_of_n", type=int, default=None,
                     help="samples for best_of_n specifically; falls back to --n if not set")
    ap.add_argument("--n_grid", type=int, default=4,
                     help="example images shown per technique (pass the same value as --n "
                          "to display every generated image — only practical for small N)")
    ap.add_argument("--img_cols", type=int, default=8, help="image grid wraps to a new row after this many columns")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--num_inference_steps_fk_steering", type=int, default=None,
                     help="denoising steps for fk_steering specifically; falls back to "
                          "--num_inference_steps if not set. Does NOT affect cfg/auto_guidance_* "
                          "or their checkpoints.")
    ap.add_argument("--num_inference_steps_best_of_n", type=int, default=None,
                     help="denoising steps for best_of_n specifically; falls back to "
                          "--num_inference_steps if not set. Does NOT affect cfg/auto_guidance_* "
                          "or their checkpoints.")
    ap.add_argument("--cfg_scale", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=1024, help="lower this (e.g. 512) if you hit CUDA OOM")
    ap.add_argument("--width", type=int, default=1024, help="lower this (e.g. 512) if you hit CUDA OOM")
    ap.add_argument("--auto_guidance_scale", type=float, default=3.5)
    # NOTE: --auto_guidance_variant is no longer used to pick a single variant to run —
    # all 3 variants now run as separate techniques when "auto_guidance_*" is in
    # --techniques (or by default). These flags tune each variant's own knob:
    ap.add_argument("--auto_guidance_latent_scale", type=float, default=0.95,
                     help="[latent_scale variant] shrink factor applied to latents in the weak branch")
    ap.add_argument("--auto_guidance_weight_noise_std", type=float, default=0.05,
                     help="[weight_noise variant] relative Gaussian noise std added to perturbed weights")
    ap.add_argument("--auto_guidance_num_weights_perturbed", type=int, default=1,
                     help="[weight_noise variant] how many weight tensors to perturb")
    ap.add_argument("--auto_guidance_timestep_stride", type=int, default=4,
                     help="[fewer_timesteps variant] round weak-branch timestep down to this stride")
    ap.add_argument("--num_particles", type=int, default=4, help="FK steering particle count")
    ap.add_argument("--resample_interval", type=int, default=5)
    ap.add_argument("--best_of_n_n", type=int, default=5,
                     help="number of candidates generated per sample for the best_of_n technique")
    ap.add_argument("--checkpoint_dir", default=".eval_checkpoints",
                     help="per-technique results (scores/feats/grid images) are pickled here as soon "
                          "as each technique finishes, so a crash/disconnect only loses the technique "
                          "in progress, not previously completed ones. On restart, any technique with "
                          "a matching checkpoint (same technique+n+seed) is loaded instead of "
                          "recomputed. Does not change the final output: still exactly one figure, "
                          "these are working checkpoints, not a deliverable. Delete this dir for a "
                          "clean re-run, or pass --no_checkpoint to disable entirely.")
    ap.add_argument("--no_checkpoint", action="store_true", help="disable checkpointing entirely")
    ap.add_argument("--batch_size", default="auto",
                     help="prompts denoised together per forward pass, for cfg/auto_guidance_* only "
                          "(fk_steering/best_of_n always run 1 prompt at a time externally — they "
                          "already batch internally via particles/candidates). Pass an integer to "
                          "fix it manually, or 'auto' (default) to probe the actual free VRAM at "
                          "startup and pick the largest batch_size that fits (up to 256), backing "
                          "off one step for headroom. There is no safe universal value: on a shared "
                          "~24GB GPU at 512-1024px, auto will typically land on 1-8, not 128-256 — "
                          "that's a real hardware ceiling, not a bug.")
    ap.add_argument("--no_amp", action="store_true",
                     help="disable torch.autocast around transformer forward passes. AMP is ON by "
                          "default now. Note: the model is already loaded in bf16 (src/model.py), "
                          "so autocast mainly buys numerical stability for sensitive ops, not a big "
                          "extra speed/memory win on top of that.")
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
        "auto_guidance_latent_scale": args.auto_guidance_latent_scale,
        "auto_guidance_weight_noise_std": args.auto_guidance_weight_noise_std,
        "auto_guidance_num_weights_perturbed": args.auto_guidance_num_weights_perturbed,
        "auto_guidance_timestep_stride": args.auto_guidance_timestep_stride,
        "num_particles": args.num_particles,
        "resample_interval": args.resample_interval,
        "use_amp": not args.no_amp,
        "negative_prompt": "blurry, low quality, distorted",
    }

    # --batch_size auto: probe real free VRAM with a throwaway ModularSampler
    # call before the real run starts, rather than guessing a fixed number.
    if str(args.batch_size).lower() == "auto":
        from src.sampler import find_max_batch_size
        probe_sampler = ModularSampler(model_wrapper)
        probe_config = dict(base_config)
        probe_config["seed"] = args.seed
        args.batch_size = find_max_batch_size(
            probe_sampler, prompts[0], probe_config, ceiling=256,
        )
    else:
        args.batch_size = int(args.batch_size)
    print(f"[setup] Using batch_size={args.batch_size} for cfg/auto_guidance_* techniques")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def n_for(technique):
        if technique == "fk_steering" and args.n_fk_steering is not None:
            return args.n_fk_steering
        if technique == "best_of_n" and args.n_best_of_n is not None:
            return args.n_best_of_n
        return args.n

    def steps_for(technique):
        if technique == "fk_steering" and args.num_inference_steps_fk_steering is not None:
            return args.num_inference_steps_fk_steering
        if technique == "best_of_n" and args.num_inference_steps_best_of_n is not None:
            return args.num_inference_steps_best_of_n
        return None  # None = use whatever's already in base_config, unchanged

    def checkpoint_path(technique):
        # keyed on technique+n+seed so a checkpoint is only reused when it
        # actually matches the run being requested
        return Path(args.checkpoint_dir) / f"{technique}_n{n_for(technique)}_seed{args.seed}.pkl"

    results = []
    for technique in args.techniques:
        n_tech = n_for(technique)
        ckpt = checkpoint_path(technique)
        if not args.no_checkpoint and ckpt.exists():
            print(f"\n[technique] {TECHNIQUE_LABELS.get(technique, technique)} — "
                  f"found checkpoint at {ckpt}, loading instead of recomputing")
            with open(ckpt, "rb") as f:
                r = pickle.load(f)
            results.append(r)
            continue

        print(f"\n[technique] {TECHNIQUE_LABELS.get(technique, technique)} — generating {n_tech} samples")
        r = evaluate_technique(
            technique, model_wrapper, reward_fn, inception, use_pytorch_fid,
            prompts, base_config, n_tech, args.n_grid, args.device, args.seed,
            feat_batch_size=args.feat_batch_size, best_of_n_n=args.best_of_n_n,
            batch_size=args.batch_size, steps_override=steps_for(technique),
        )
        results.append(r)

        if not args.no_checkpoint:
            with open(ckpt, "wb") as f:
                pickle.dump(r, f)
            print(f"  [checkpoint] saved {ckpt} — safe to resume from here if the run is interrupted later")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_figure(results, real_feats, args.out, args.n_grid, img_cols=args.img_cols)


if __name__ == "__main__":
    main()
"""
Task 2 — FID + CLIP score: model-generated images vs. real ImageNet images.

Usage:
    python scripts/eval_fid_clip.py \
        --model_id stabilityai/stable-diffusion-3.5-medium \
        --sampler cfg \
        --real_dir /path/to/imagenet/val \
        --prompts_file scripts/imagenet_prompts.txt \
        --n 5000 \
        --out results/fid_clip_report.json

Notes:
  - Real images: point --real_dir at a local ImageNet-style folder
    (class subfolders of images). We do NOT auto-download ImageNet —
    no image-host network access here, and ImageNet requires a license
    agreement anyway. Put --n (default 5000) real images in there.
  - Generated images: one prompt per image, sampled from
    --prompts_file (one prompt per line, e.g. "a photo of a {class}"
    for all 1000 ImageNet classes — you supply this list once).
  - FID: standard Heusel et al. recipe — InceptionV3 pool (2048-d)
    features, Frechet distance between the two Gaussian fits. Uses
    `pytorch-fid`'s Inception weights if that package is installed
    (closest match to the original TF FID); otherwise falls back to
    torchvision's inception_v3 so the script still runs.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torchvision import transforms as T
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.model import SD35Wrapper
from src.sampler import ModularSampler
from src.rewards.clip_reward import CLIPPromptReward
from src.utils import clip_align_score


def _load_inception(device):
    try:
        from pytorch_fid.inception import InceptionV3
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device).eval()
        return model, True
    except ImportError:
        from torchvision.models import inception_v3, Inception_V3_Weights
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        return model, False


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def _inception_features(model, use_pytorch_fid, images, device, batch_size=50):
    tfm = T.Compose([T.Resize((299, 299)), T.ToTensor()])
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([tfm(img.convert("RGB")) for img in images[i:i + batch_size]]).to(device)
        if use_pytorch_fid:
            # pytorch-fid's InceptionV3 wrapper expects [0, 1] input and
            # normalizes internally — do NOT normalize here.
            out = model(batch)[0].squeeze(-1).squeeze(-1)
        else:
            # torchvision's IMAGENET1K_V1 weights expect standard ImageNet
            # mean/std normalization, not [-1, 1].
            mean = _IMAGENET_MEAN.to(device)
            std = _IMAGENET_STD.to(device)
            out = model((batch - mean) / std)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)


def _fid_from_features(feats_real, feats_gen):
    mu1, sigma1 = feats_real.mean(axis=0), np.cov(feats_real, rowvar=False)
    mu2, sigma2 = feats_gen.mean(axis=0), np.cov(feats_gen, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def sample_real_images(real_dir, n, seed=0):
    random.seed(seed)
    paths = [p for p in Path(real_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if len(paths) < n:
        raise ValueError(
            f"Only found {len(paths)} real images in {real_dir}, need {n}.\n"
            f"This script does NOT download ImageNet itself. Run "
            f"scripts/prepare_imagenet_val.py first to populate {real_dir} "
            f"with real images (see that script for the one-time HF login/"
            f"license-accept step), or point --real_dir at a folder you "
            f"already have >= {n} real images in, or lower --n for a quick test."
        )
    return [Image.open(p) for p in random.sample(paths, n)]


def load_prompts(prompts_file, n, seed=0):
    random.seed(seed)
    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) >= n:
        return random.sample(prompts, n)
    return [random.choice(prompts) for _ in range(n)]  # sample with replacement


def _tensor_to_pil(img_tensor):
    arr = img_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return Image.fromarray((arr * 255).astype("uint8"))


def generate_images(model_wrapper, sampler_name, prompts, base_config):
    sampler = ModularSampler(model_wrapper)
    reward_fn = CLIPPromptReward()
    images, clip_scores = [], []
    for i, prompt in enumerate(prompts):
        config = dict(base_config)
        config["prompt"] = prompt
        config["seed"] = base_config.get("seed", 0) + i
        img, _ = sampler.sample(
            prompt=prompt, config=config, exp_name=f"fid_eval/{sampler_name}",
            methods_override=[sampler_name], save_name=f"gen_{i:05d}.png",
        )
        images.append(img)
        clip_scores.append(clip_align_score(reward_fn, img, prompt))
        if (i + 1) % 200 == 0:
            print(f"  generated {i + 1}/{len(prompts)}")
    return images, clip_scores

print("\n>>> RUNNING UPDATED EVAL_FID_CLIP SCRIPT <<<\n")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-large")
    ap.add_argument("--sampler", default="cfg", help="method key")
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--prompts_file", default="scripts/imagenet_prompts.json")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--cfg_scale", type=float, default=7.5)
    ap.add_argument("--out", default="results/fid_clip_report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[1/4] Sampling {args.n} real images from {args.real_dir}")
    real_images = sample_real_images(args.real_dir, args.n, seed=args.seed)

    print(f"[2/4] Generating {args.n} images with sampler='{args.sampler}'")
    model_wrapper = SD35Wrapper(model_id=args.model_id)
    prompts = load_prompts(args.prompts_file, args.n, seed=args.seed)
    base_config = {
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "negative_prompt": "blurry, low quality, distorted",
    }
    gen_tensors, clip_scores = generate_images(model_wrapper, args.sampler, prompts, base_config)
    gen_images = [_tensor_to_pil(t) for t in gen_tensors]

    print("[3/4] Extracting Inception features + computing FID")
    inception, use_pytorch_fid = _load_inception(args.device)
    feats_real = _inception_features(inception, use_pytorch_fid, real_images, args.device)
    feats_gen = _inception_features(inception, use_pytorch_fid, gen_images, args.device)
    fid = _fid_from_features(feats_real, feats_gen)

    report = {
        "model_id": args.model_id,
        "sampler": args.sampler,
        "n_real": len(real_images),
        "n_generated": len(gen_images),
        "fid": round(fid, 3),
        "clip_score_mean": round(float(np.mean(clip_scores)), 4),
        "clip_score_std": round(float(np.std(clip_scores)), 4),
        "inception_backend": "pytorch_fid" if use_pytorch_fid else "torchvision_fallback",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n[4/4] === RESULTS ===")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
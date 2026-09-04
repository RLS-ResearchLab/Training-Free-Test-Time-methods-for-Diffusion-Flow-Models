"""
Task 4 — multi-prompt evaluation.

The existing configs all reuse one short prompt ("A high-tech
cybernetic garden, cinematic lighting, 8k"). This runs a technique
across several long, compositionally demanding prompts (multiple
objects, attributes, spatial relationships) and generates several
samples per prompt, so results reflect prompt complexity rather than
one lucky/unlucky prompt.

Usage:
    python scripts/run_multi_prompt.py --sampler best_of_n --k 4 \
        --out results/multi_prompt_report.json
"""
import argparse
import json
from pathlib import Path

from src.model import load_model
from src.sampler import ModularSampler
from src.samplers.best_of_n import BestOfNSampler
from src.rewards.clip_reward import CLIPPromptReward
from src.utils import clip_align_score


LARGE_PROMPTS = [
    "A cluttered Victorian-era study at dusk: a brass telescope pointed "
    "out a rain-streaked window, stacks of leather-bound books on the "
    "floor, a globe tilted on its axis, warm candlelight, oil painting "
    "style, extremely detailed",

    "An overhead shot of a bustling night market street in Southeast "
    "Asia: dozens of red paper lanterns strung overhead, steam rising "
    "from food stalls, a wet reflective road surface, crowds of people "
    "with umbrellas, neon signage in the background, cinematic color "
    "grading",

    "A cutaway technical illustration of a deep-sea research submarine, "
    "showing the pilot cabin, ballast tanks, robotic manipulator arms, "
    "and pressure hull in cross-section, labeled diagram style, blue "
    "and orange color scheme, engineering blueprint aesthetic",

    "A close-up macro photograph of a dew-covered spiderweb strung "
    "between two dry wheat stalks at golden hour, extremely shallow "
    "depth of field, individual water droplets in sharp focus, soft "
    "bokeh background",

    "A fantasy marketplace built into the trunk of an enormous ancient "
    "tree: spiral wooden staircases, glowing lanterns made of "
    "fireflies in jars, merchant stalls selling potions and scrolls, "
    "small winged creatures flying between branches, painterly "
    "concept-art style",

    "A minimalist product shot of a matte black ceramic coffee mug on "
    "a polished concrete surface, single soft studio light from the "
    "upper left, subtle shadow, shallow depth of field, commercial "
    "photography style",
]


def build_sampler(name, model_wrapper, k):
    if name == "best_of_n":
        return BestOfNSampler(model_wrapper, base_methods=["cfg"], n=k), "best_of_n"
    return ModularSampler(model_wrapper), name


def run_one_prompt(sampler, method_key, prompt, base_config, reward_fn, p_idx, k):
    results = []
    for j in range(k):
        run_config = dict(base_config)
        run_config["prompt"] = prompt
        run_config["seed"] = base_config["seed"] + p_idx * 100 + j
        kwargs = dict(
            prompt=prompt, config=run_config,
            exp_name=f"multi_prompt/{method_key}/{p_idx:02d}", save_name=f"sample_{j}.png",
        )
        if method_key != "best_of_n":
            kwargs["methods_override"] = [method_key]
        img, metrics = sampler.sample(**kwargs)
        score = clip_align_score(reward_fn, img, prompt)
        results.append({
            "sample_idx": j,
            "clip_score": round(score, 4),
            "forward_passes": metrics.get("total_forward_passes"),
            "time_sec": round(metrics.get("time_sec", 0.0), 3),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--sampler", default="cfg", help="cfg | auto_guidance | fk_steering | best_of_n")
    ap.add_argument("--k", type=int, default=4, help="samples returned per prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--cfg_scale", type=float, default=7.5)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    model_wrapper = load_model(args.model_id)
    sampler, method_key = build_sampler(args.sampler, model_wrapper, args.k)
    reward_fn = CLIPPromptReward()
    base_config = {
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "negative_prompt": "blurry, low quality, distorted",
    }

    all_results = []
    for p_idx, prompt in enumerate(LARGE_PROMPTS):
        print(f"\n[Prompt {p_idx + 1}/{len(LARGE_PROMPTS)}] {prompt[:70]}...")
        sample_results = run_one_prompt(sampler, method_key, prompt, base_config, reward_fn, p_idx, args.k)
        best = max(sample_results, key=lambda r: r["clip_score"])
        for r in sample_results:
            r["is_best_for_prompt"] = (r["sample_idx"] == best["sample_idx"])

        all_results.append({
            "prompt_idx": p_idx,
            "prompt": prompt,
            "sampler": method_key,
            "samples": sample_results,
            "mean_clip_score": round(sum(r["clip_score"] for r in sample_results) / len(sample_results), 4),
            "best_clip_score": best["clip_score"],
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if Path(args.out).exists():
        existing = json.loads(Path(args.out).read_text())
    existing.setdefault("ablation_experiments", [])
    existing.setdefault("multi_prompt", {})
    existing["multi_prompt"][method_key] = all_results
    existing.setdefault("notes", {
        "fid": (
            "Aucun score FID n'est present : scripts/eval_fid_clip.py exige --real_dir "
            "(un dossier local d'images reelles type ImageNet-val) qui n'est jamais telecharge "
            "automatiquement (licence requise + pas d'acces reseau a un hebergeur d'images dans "
            "cet environnement). dino_lpips_vs_without sert de proxy de similarite en attendant."
        ),
    })
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    for r in all_results:
        print(f"[{r['prompt_idx']}] mean_clip={r['mean_clip_score']:.4f}  best_clip={r['best_clip_score']:.4f}  {r['prompt'][:60]}...")

    print(f"\n[OK] Résultats multi-prompt ({method_key}) fusionnés dans {args.out}")


if __name__ == "__main__":
    main()
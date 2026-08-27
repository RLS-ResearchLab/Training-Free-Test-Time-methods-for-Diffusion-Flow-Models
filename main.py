"""
Patched version of main.py — only change from the original: LPIPS is
now computed on DINO features instead of raw pixels/AlexNet (Task 1).
Everything else (experiment discovery, figure, results table) is
unchanged. Diff-relevant lines are marked with `# >>> DINO`.
"""
import glob
import json
import yaml
import torch

from src.model import SD35Wrapper
from src.sampler import ModularSampler
from src.samplers.fk_steering import FKSteeringSampler
from src.rewards.clip_reward import CLIPPromptReward
from src.utils import clip_align_score
from src.dino_lpips import DinoLPIPS, dino_lpips_distance  # >>> DINO
from build_full_comparison import build as build_comparison_grid


def _make_row(exp_name, method, state, metrics, prompt, image, reward_fn, variant=None, dino_lpips_vs_without=None):
    return {
        "experiment": exp_name,
        "method": method,
        "variant": variant or "",
        "state": state,
        "forward_passes": metrics.get("total_forward_passes"),
        "time_sec": round(metrics.get("time_sec", 0.0), 3) if metrics.get("time_sec") is not None else None,
        "peak_memory_mb": round(metrics.get("peak_memory_mb", 0.0), 1) if metrics.get("peak_memory_mb") is not None else None,
        "clip_score": round(clip_align_score(reward_fn, image, prompt), 4),
        # >>> DINO: renamed field + DINO-based value instead of pixel/AlexNet LPIPS
        "dino_lpips_vs_without": round(dino_lpips_vs_without, 4) if dino_lpips_vs_without is not None else None,
    }


def run_all_experiments():
    config_paths = sorted(glob.glob("configs/*.yaml"))
    print(f"Found {len(config_paths)} experiment config(s) to run.")
    for p in config_paths:
        print(f"  - {p}")

    model_wrapper = SD35Wrapper(model_id="stabilityai/stable-diffusion-3.5-medium")
    modular_sampler = ModularSampler(model_wrapper)
    reward_fn = CLIPPromptReward()
    fk_sampler = FKSteeringSampler(model_wrapper, reward_fn)
    dino_model = DinoLPIPS(device="cuda" if torch.cuda.is_available() else "cpu")  # >>> DINO: build once

    rows_for_figure = []
    results = []

    for path in config_paths:
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        exp_name = config.get("experiment_name")
        if not exp_name:
            print(f"  [Skipping] {path}: missing 'experiment_name' key.")
            continue
        prompt = config.get("prompt")
        if not prompt:
            print(f"  [Skipping] {exp_name}: missing 'prompt' key.")
            continue
        methods = config.get("methods", ["cfg"])
        print(f"\n[Running] {exp_name} | Methods: {methods}")

        try:
            if "fk_steering" in methods:
                img_without, met_without = modular_sampler.sample(
                    prompt=prompt, config=config, exp_name=f"{exp_name}/fk_steering",
                    methods_override=["cfg"], save_name="without_fk_steering.png",
                )
                img_with, met_with = fk_sampler.sample(
                    prompt=prompt, config=config, exp_name=f"{exp_name}/fk_steering",
                    save_name="with_fk_steering.png",
                )
                dist = dino_lpips_distance(img_with, img_without, dino_model=dino_model)  # >>> DINO

                rows_for_figure.append((f"{exp_name} — fk_steering", img_without, met_without, img_with, met_with))
                results.append(_make_row(exp_name, "fk_steering", "without", met_without, prompt, img_without, reward_fn))
                results.append(_make_row(exp_name, "fk_steering", "with", met_with, prompt, img_with, reward_fn, dino_lpips_vs_without=dist))

                torch.cuda.empty_cache()
                continue

            for method in methods:
                methods_without = [m for m in methods if m != method]
                variant = config.get("auto_guidance_variant") if method == "auto_guidance" else None

                img_without, met_without = modular_sampler.sample(
                    prompt=prompt, config=config, exp_name=f"{exp_name}/{method}",
                    methods_override=methods_without, save_name=f"without_{method}.png",
                )
                img_with, met_with = modular_sampler.sample(
                    prompt=prompt, config=config, exp_name=f"{exp_name}/{method}",
                    methods_override=methods, save_name=f"with_{method}.png",
                )
                dist = dino_lpips_distance(img_with, img_without, dino_model=dino_model)  # >>> DINO

                row_title = f"{exp_name} — {method}" + (f" ({variant})" if variant else "")
                rows_for_figure.append((row_title, img_without, met_without, img_with, met_with))
                results.append(_make_row(exp_name, method, "without", met_without, prompt, img_without, reward_fn, variant=variant))
                results.append(_make_row(exp_name, method, "with", met_with, prompt, img_with, reward_fn, variant=variant, dino_lpips_vs_without=dist))

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] {exp_name} failed: {type(e).__name__}: {e}")
            continue

    if not rows_for_figure:
        print("\nNo experiments produced any images — check the [ERROR]/[Skipping] lines above.")
        return

    _write_results_json(results)
    build_comparison_grid()


def _write_results_json(results, path="results.json"):
    """Fusionne les nouvelles lignes d'ablation dans l'unique fichier de
    résultats results.json, en conservant les données multi_prompt et les
    notes déjà présentes plutôt que d'écrire results_table.csv/.md séparés."""
    from pathlib import Path

    existing = {}
    if Path(path).exists():
        existing = json.loads(Path(path).read_text())

    existing["ablation_experiments"] = results
    existing.setdefault("multi_prompt", {})
    existing.setdefault("notes", {
        "fid": (
            "Aucun score FID n'est present : scripts/eval_fid_clip.py exige --real_dir "
            "(un dossier local d'images reelles type ImageNet-val) qui n'est jamais telecharge "
            "automatiquement (licence requise + pas d'acces reseau a un hebergeur d'images dans "
            "cet environnement). dino_lpips_vs_without sert de proxy de similarite en attendant."
        ),
    })

    Path(path).write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"[Done] Results written to '{path}'")


if __name__ == "__main__":
    run_all_experiments()
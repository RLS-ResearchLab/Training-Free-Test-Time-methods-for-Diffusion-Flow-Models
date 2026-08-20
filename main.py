"""
Patched version of main.py — only change from the original: LPIPS is
now computed on DINO features instead of raw pixels/AlexNet (Task 1).
Everything else (experiment discovery, figure, results table) is
unchanged. Diff-relevant lines are marked with `# >>> DINO`.
"""
import csv
import glob
import yaml
import matplotlib.pyplot as plt
import torch

from src.model import SD35Wrapper
from src.sampler import ModularSampler
from src.samplers.fk_steering import FKSteeringSampler
from src.rewards.clip_reward import CLIPPromptReward
from src.utils import clip_align_score
from src.dino_lpips import DinoLPIPS, dino_lpips_distance  # >>> DINO


def _to_display(image: torch.Tensor):
    img = image[0].detach().cpu().permute(1, 2, 0).numpy()
    return (img - img.min()) / (img.max() - img.min())


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

    n = len(rows_for_figure)
    fig, axes = plt.subplots(n, 2, figsize=(12, 6 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row_idx, (title, img_without, met_without, img_with, met_with) in enumerate(rows_for_figure):
        ax_without, ax_with = axes[row_idx, 0], axes[row_idx, 1]
        ax_without.imshow(_to_display(img_without))
        ax_without.set_title(f"{title}\nWithout — {met_without.get('total_forward_passes', 'N/A')} fwd passes", fontsize=10)
        ax_without.axis("off")
        ax_with.imshow(_to_display(img_with))
        ax_with.set_title(f"{title}\nWith — {met_with.get('total_forward_passes', 'N/A')} fwd passes", fontsize=10)
        ax_with.axis("off")

    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=200)
    print("\n[Done] Comparison figure saved to 'experiment_comparison.png'")

    _write_results_table(results)


def _write_results_table(results):
    fieldnames = [
        "experiment", "method", "variant", "state",
        "forward_passes", "time_sec", "peak_memory_mb",
        "clip_score", "dino_lpips_vs_without",  # >>> DINO: renamed column
    ]

    with open("results_table.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("[Done] Results table saved to 'results_table.csv'")

    display_rows = [[r[k] if r[k] is not None else "—" for k in fieldnames] for r in results]
    try:
        from tabulate import tabulate
        table_str = tabulate(display_rows, headers=fieldnames, tablefmt="github")
    except ImportError:
        col_widths = [max(len(str(fieldnames[i])), *(len(str(r[i])) for r in display_rows)) for i in range(len(fieldnames))]
        header = " | ".join(str(fieldnames[i]).ljust(col_widths[i]) for i in range(len(fieldnames)))
        sep = "-+-".join("-" * w for w in col_widths)
        body = "\n".join(
            " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(fieldnames)))
            for r in display_rows
        )
        table_str = f"{header}\n{sep}\n{body}"

    print("\n" + table_str)
    with open("results_table.md", "w") as f:
        f.write(table_str + "\n")
    print("\n[Done] Formatted table also saved to 'results_table.md'")


if __name__ == "__main__":
    run_all_experiments()
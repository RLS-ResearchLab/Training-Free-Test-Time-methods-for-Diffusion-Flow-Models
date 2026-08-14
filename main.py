import glob
import yaml
import matplotlib.pyplot as plt
import torch

from src.model import SD35Wrapper
from src.sampler import ModularSampler
from src.samplers.fk_steering import FKSteeringSampler
from src.rewards.clip_reward import CLIPPromptReward


def run_all_experiments():
    # 1. Discover all configuration files automatically
    config_paths = glob.glob("configs/*.yaml")
    print(f"Found {len(config_paths)} experiment config(s) to run.")

    # 2. Load model into memory ONCE, shared across all samplers
    model_wrapper = SD35Wrapper(model_id="stabilityai/stable-diffusion-3.5-medium")

    # 3. Instantiate both samplers up front — reward model loaded once too,
    #    so FK steering doesn't reload CLIP on every experiment.
    modular_sampler = ModularSampler(model_wrapper)
    reward_fn = CLIPPromptReward()
    fk_sampler = FKSteeringSampler(model_wrapper, reward_fn)

    fig, axes = plt.subplots(1, len(config_paths), figsize=(6 * len(config_paths), 6))
    if len(config_paths) == 1:
        axes = [axes]  # normalize so axes[idx] always works, even with 1 config

    # 4. Loop over each experiment
    for idx, path in enumerate(config_paths):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        methods = config.get("methods", ["cfg"])
        print(f"\n[Running] {config['experiment_name']} | Methods: {methods}")

        # Route to the right sampler based on the method declared in the config.
        # fk_steering needs a reward_fn and returns only its best particle,
        # so it can't share ModularSampler's call signature/behavior.
        if "fk_steering" in methods:
            image, metrics = fk_sampler.sample(
                prompt=config["prompt"], config=config, exp_name=config["experiment_name"]
            )
        else:
            image, metrics = modular_sampler.sample(
                prompt=config["prompt"], config=config, exp_name=config["experiment_name"]
            )

        # 5. Display result in comparison plot
        img_display = image[0].detach().cpu().permute(1, 2, 0).numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        axes[idx].imshow(img_display)
        axes[idx].set_title(
            f"{config['experiment_name']}\nForward Passes: {metrics.get('total_forward_passes', 'N/A')}",
            fontsize=12
        )
        axes[idx].axis("off")
        torch.cuda.empty_cache()

    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=300)
    print("\n[Done] Results saved to 'experiment_comparison.png'")


if __name__ == "__main__":
    run_all_experiments()
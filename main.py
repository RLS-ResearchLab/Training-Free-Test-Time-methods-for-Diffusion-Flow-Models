import glob
import yaml
import matplotlib.pyplot as plt

from src.model import PretrainedT2IModel
from src.sampler import TestTimeSampler

def run_all_experiments():
    # 1. Discover all configuration files automatically
    config_paths = glob.glob("configs/*.yaml")
    print(f"Found {len(config_paths)} experiment config(s) to run.")

    # 2. Load model into memory ONCE
    model_wrapper = PretrainedT2IModel(model_id="runwayml/stable-diffusion-v1-5")
    sampler = TestTimeSampler(model_wrapper)

    # Prepare matplotlib figure for side-by-side comparison
    fig, axes = plt.subplots(1, len(config_paths), figsize=(6 * len(config_paths), 6))
    if len(config_paths) == 1:
        axes = [axes]  # Ensure indexing works if there's only 1 config

    # 3. Loop over each experiment
    for idx, path in enumerate(config_paths):
        # Read parameters from YAML file
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        print(f"\n[Running] {config['experiment_name']} | Methods: {config['methods']}")

        # Run custom sampling loop
        image, metrics = sampler.sample(prompt=config["prompt"], config=config)

        # 4. Display result in comparison plot
        axes[idx].imshow(image[0])
        axes[idx].set_title(
            f"{config['experiment_name']}\nForward Passes: {metrics['total_forward_passes']}",
            fontsize=12
        )
        axes[idx].axis("off")

    # Save final plot grid
    plt.tight_layout()
    plt.savefig("experiment_comparison.png", dpi=300)
    print("\n[Done] Results saved to 'experiment_comparison.png'")

if __name__ == "__main__":
    run_all_experiments()
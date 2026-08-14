import os
import torch
from PIL import Image

class ModularSampler:
    def __init__(self, model_wrapper, output_dir="samples"):
        self.model = model_wrapper
        self.output_dir = output_dir

    def _get_weaker_prediction(self, latents, t, cond_embeds, pooled):
        """Helper for Auto-Guidance (e.g., perturbed latent pass or secondary model)."""
        perturbed_latents = latents * 0.95  # Placeholder perturbation
        return self.model.transformer(
            hidden_states=perturbed_latents,
            timestep=t.unsqueeze(0) if t.dim() == 0 else t,
            encoder_hidden_states=cond_embeds,
            pooled_projections=pooled,
            return_dict=False
        )[0]

    @torch.no_grad()
    def sample(self, prompt: str, config: dict, exp_name: str = "exp_run"):
        save_folder = os.path.join(self.output_dir, exp_name)
        active_methods = config.get("methods", ["cfg"])
        
        cfg_scale = config.get("cfg_scale", 4.5)
        ag_scale = config.get("auto_guidance_scale", 1.0)

        # 1. Encode prompt (conditioned & unconditioned)
        embeds = self.model.encode_prompts(prompt)
        cond_embeds, cond_pooled = embeds["cond"]["prompt_embeds"], embeds["cond"]["pooled_prompt_embeds"]
        uncond_embeds, uncond_pooled = embeds["uncond"]["prompt_embeds"], embeds["uncond"]["pooled_prompt_embeds"]

        # 2. Init Latents & Scheduler
        latents = self.model.init_latents(batch_size=1, seed=config.get("seed", 42))
        self.model.scheduler.set_timesteps(
        config.get("num_inference_steps", 28),
        device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # 3. Modular Denoising Loop
        for t in self.model.scheduler.timesteps:
            # Base Forward Pass (Conditional)
            noise_cond = self.model.transformer(
                hidden_states=latents,
                timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                encoder_hidden_states=cond_embeds,
                pooled_projections=cond_pooled,
                return_dict=False
            )[0]

            guided_noise = noise_cond.clone()

            # --- Method A: Classifier-Free Guidance (CFG) ---
            if "cfg" in active_methods and cfg_scale > 1.0:
                noise_uncond = self.model.transformer(
                    hidden_states=latents,
                    timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                    encoder_hidden_states=uncond_embeds,
                    pooled_projections=uncond_pooled,
                    return_dict=False
                )[0]
                guided_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # --- Method B: Auto-Guidance ---
            if "auto_guidance" in active_methods and ag_scale > 0.0:
                noise_weaker = self._get_weaker_prediction(latents, t, cond_embeds, cond_pooled)
                guided_noise = guided_noise + ag_scale * (noise_cond - noise_weaker)

            # Update latent step (t -> t-1)
            latents = self.model.scheduler.step(guided_noise, t, latents, return_dict=False)[0]

        # 4. Decode & Save Result
        image = self.model.decode(latents)
        self._save_tensor_as_image(image, os.path.join(save_folder, "result.png"))
        return image, {}

    def _save_tensor_as_image(self, tensor: torch.Tensor, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tensor = (tensor.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
        tensor = (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(tensor).save(save_path)
        print(f"Saved result: {save_path}")
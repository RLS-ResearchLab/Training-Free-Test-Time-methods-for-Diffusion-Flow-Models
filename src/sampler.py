import torch
from src.model import PretrainedT2IModel

class TestTimeSampler:
    """Executes denoising loops with combined test-time techniques."""

    def __init__(self, model_wrapper: PretrainedT2IModel):
        self.model = model_wrapper

    def sample(self, prompt: str, config: dict):
        num_inference_steps = config.get("num_inference_steps", 50)
        cfg_scale = config.get("cfg_scale", 7.5)
        methods = config.get("methods", [])  # e.g. ["cfg", "sag"]

        text_embeds = self.model.encode_prompt(prompt)
        generator = torch.Generator(device=self.model.device).manual_seed(config.get("seed", 42))
        
        # Initialize noise
        latents = torch.randn((1, 4, 64, 64), generator=generator, device=self.model.device)
        self.model.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.model.scheduler.init_noise_sigma

        # Count forward passes to estimate FLOPs / compute cost
        forward_pass_count = 0 

        for t in self.model.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            latent_input = self.model.scheduler.scale_model_input(latent_input, t)

            # Predict Noise
            with torch.no_grad():
                noise_pred = self.model.unet(latent_input, t, encoder_hidden_states=text_embeds).sample
                forward_pass_count += 2  # uncond + cond pass

            noise_uncond, noise_cond = noise_pred.chunk(2)

            # 1. Base Classifier-Free Guidance (CFG)
            guided_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # 2. Add additional test-time methods dynamically (0 to N combined)
            if "sag" in methods:
                # Example placeholder logic for Self-Attention Guidance (SAG)
                sag_scale = config.get("sag_scale", 0.5)
                guided_noise += sag_scale * (noise_cond - noise_uncond)

            # Step latent update
            latents = self.model.scheduler.step(guided_noise, t, latents).prev_sample

        decoded_image = self.model.decode_latents(latents)
        
        metrics = {
            "total_forward_passes": forward_pass_count,
            "approx_gflops": forward_pass_count * 1.2  # Placeholder FLOP factor
        }
        
        return decoded_image, metrics
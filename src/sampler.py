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


class ModularSampler:
    """Configurable and modular sampling loop supporting CFG and Auto-Guidance."""

    def __init__(self, model_wrapper: PretrainedT2IModel, weaker_model_wrapper: PretrainedT2IModel = None):
        self.model = model_wrapper
        self.weaker_model = weaker_model_wrapper  # Optional secondary model for Auto-Guidance

    def _get_weaker_prediction(self, latents, t, encoder_hidden_states):
        """Helper to get weaker model prediction for Auto-Guidance."""
        if self.weaker_model is not None:
            # Option A: Use an actual weaker checkpoint if available
            return self.weaker_model.unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
        else:
            # Option B: Fallback / Placeholder (e.g., perturbed input or degraded prediction)
            # This keeps the code ready without breaking if no weaker model is provided yet
            perturbed_latents = latents * 0.95  # Simple placeholder degradation
            return self.model.unet(perturbed_latents, t, encoder_hidden_states=encoder_hidden_states).sample

    @torch.no_grad()
    def sample(self, prompt: str, config: dict):
        # Extract configuration flags
        num_steps = config.get("num_inference_steps", 30)
        cfg_scale = config.get("cfg_scale", 7.5)
        auto_guidance_scale = config.get("auto_guidance_scale", 1.0)
        
        active_methods = config.get("methods", ["cfg"])  # e.g., ["cfg"], ["auto_guidance"], or ["cfg", "auto_guidance"]
        
        # 1. Prepare Text Embeddings
        text_embeds = self.model.encode_prompt(
            prompt=prompt, 
            negative_prompt=config.get("negative_prompt", "")
        )  # Returns concatenated [uncond_embeds, cond_embeds]
        
        uncond_embeds, cond_embeds = text_embeds.chunk(2)

        # 2. Prepare Latents & Timesteps
        generator = torch.Generator(device=self.model.device).manual_seed(config.get("seed", 42))
        latents = torch.randn((1, 4, 64, 64), generator=generator, device=self.model.device)
        
        self.model.scheduler.set_timesteps(num_steps)
        latents = latents * self.model.scheduler.init_noise_sigma

        # 3. Modular Denoising Loop
        for t in self.model.scheduler.timesteps:
            latent_model_input = self.model.scheduler.scale_model_input(latents, t)

            # --- Base Forward Pass (Conditional) ---
            noise_cond = self.model.unet(latent_model_input, t, encoder_hidden_states=cond_embeds).sample
            guided_noise = noise_cond.clone()

            # --- Method 1: Classifier-Free Guidance (CFG) ---
            if "cfg" in active_methods and cfg_scale > 1.0:
                noise_uncond = self.model.unet(latent_model_input, t, encoder_hidden_states=uncond_embeds).sample
                guided_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # --- Method 2: Auto-Guidance ---
            if "auto_guidance" in active_methods and auto_guidance_scale > 0.0:
                noise_weaker = self._get_weaker_prediction(latent_model_input, t, cond_embeds)
                # Combine auto-guidance delta
                guided_noise = guided_noise + auto_guidance_scale * (noise_cond - noise_weaker)

            # Step latent update (t -> t-1)
            latents = self.model.scheduler.step(guided_noise, t, latents).prev_sample

        # 4. Decode to Image
        decoded_image = self.model.decode_latents(latents)
        return decoded_image
import os
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils import ComputeTracker


class FKSteeringSampler:
    """
    Feynman-Kac steering: maintient K particules (trajectoires de denoising
    partielles) en parallèle, et les rééchantillonne périodiquement selon
    un signal de récompense, au lieu de générer N échantillons complets
    indépendamment (best-of-N classique).
    """

    def __init__(self, model_wrapper, reward_fn, output_dir="samples"):
        self.model = model_wrapper
        self.reward_fn = reward_fn  # callable(model_wrapper, latents, prompt) -> tensor [K]
        self.output_dir = output_dir

    @torch.no_grad()
    def sample(self, prompt: str, config: dict, exp_name: str = "fk_run", save_name: str = "result.png",
               save_to_disk: bool = True):
        num_particles = config.get("num_particles", 4)
        resample_interval = config.get("resample_interval", 5)
        temperature = config.get("fk_temperature", 1.0)
        cfg_scale = config.get("cfg_scale", 7.5)
        num_steps = config.get("num_inference_steps", 30)
        seed = config.get("seed", 42)

        forward_pass_count = 0

        with ComputeTracker() as tracker:
            # 1. Encode le prompt une seule fois, dupliqué sur la dimension batch (= particules)
            embeds = self.model.encode_prompts(prompt)
            cond_embeds = embeds["cond"]["prompt_embeds"].repeat(num_particles, 1, 1)
            cond_pooled = embeds["cond"]["pooled_prompt_embeds"].repeat(num_particles, 1)
            uncond_embeds = embeds["uncond"]["prompt_embeds"].repeat(num_particles, 1, 1)
            uncond_pooled = embeds["uncond"]["pooled_prompt_embeds"].repeat(num_particles, 1)

            # 2. Init K particules (bruits différents dans le même batch)
            latents = self.model.init_latents(batch_size=num_particles, seed=seed)
            self.model.scheduler.set_timesteps(
                num_steps, device="cuda" if torch.cuda.is_available() else "cpu"
            )

            for step_idx, t in enumerate(self.model.scheduler.timesteps):
                t_batch = t.repeat(num_particles) if t.dim() == 0 else t

                noise_cond = self.model.transformer(
                    hidden_states=latents, timestep=t_batch,
                    encoder_hidden_states=cond_embeds, pooled_projections=cond_pooled,
                    return_dict=False,
                )[0]
                forward_pass_count += num_particles

                noise_uncond = self.model.transformer(
                    hidden_states=latents, timestep=t_batch,
                    encoder_hidden_states=uncond_embeds, pooled_projections=uncond_pooled,
                    return_dict=False,
                )[0]
                forward_pass_count += num_particles

                guided_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                step_out = self.model.scheduler.step(guided_noise, t, latents, return_dict=True)
                latents = step_out.prev_sample

                # --- Checkpoint de rééchantillonnage FK ---
                is_last = step_idx == len(self.model.scheduler.timesteps) - 1
                if (step_idx + 1) % resample_interval == 0 and not is_last:
                    # x0 estimé si le scheduler l'expose, sinon fallback sur les latents courants
                    x0_pred = getattr(step_out, "pred_original_sample", latents)

                    rewards = self.reward_fn(self.model, x0_pred, prompt)  # [K]
                    weights = F.softmax(rewards / temperature, dim=0)
                    idx = self._systematic_resample(weights)
                    latents = latents[idx]

            # 3. Décodage final, on garde la meilleure particule
            images = self.model.decode(latents)
            final_rewards = self.reward_fn(self.model, latents, prompt)
            best_idx = int(torch.argmax(final_rewards).item())
            best_image = images[best_idx:best_idx + 1]

        if save_to_disk:
            self._save_tensor_as_image(best_image, os.path.join(self.output_dir, exp_name, save_name))

        metrics = {
            "total_forward_passes": forward_pass_count,
            "num_particles": num_particles,
            "resample_interval": resample_interval,
            "final_rewards": final_rewards.tolist(),
            "time_sec": tracker.elapsed_sec,
            "peak_memory_mb": tracker.peak_memory_mb,
        }
        return best_image, metrics

    @staticmethod
    def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
        """Rééchantillonnage systématique — variance plus faible qu'un tirage multinomial naïf."""
        K = weights.shape[0]
        positions = (torch.arange(K, device=weights.device) + torch.rand(1, device=weights.device)) / K
        cumsum = torch.cumsum(weights, dim=0)
        cumsum[-1] = 1.0  # garde-fou contre les erreurs d'arrondi flottant
        return torch.searchsorted(cumsum, positions)

    def _save_tensor_as_image(self, tensor, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t = (tensor.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0
        t = (t.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(t).save(save_path)
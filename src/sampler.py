import os
import torch
from src.utils import save_image_tensor

class ModularSampler:
    def __init__(self, model, output_dir="samples"):
        self.model = model
        self.output_dir = output_dir

    @torch.no_grad()
    def sample(self, prompt: str, config: dict, exp_name: str = "run_01"):
        save_folder = os.path.join(self.output_dir, exp_name)
        
        # 1. Encodage & Latents
        embeds = self.model.encode_prompts([prompt])
        latents = self.model.init_latents(batch_size=1, seed=config.get("seed", 42))

        self.model.scheduler.set_timesteps(config.get("num_inference_steps", 28))
        timesteps = self.model.scheduler.timesteps

        # 2. Loop de Dénoyautage
        for idx, t in enumerate(timesteps):
            # Prédiction du Transformer / DiT
            noise_pred = self.model.transformer(
                hidden_states=latents,
                timestep=t / 1000.0,
                encoder_hidden_states=embeds["cond"]["prompt_embeds"],
                pooled_projections=embeds["cond"]["pooled_prompt_embeds"],
                return_dict=False
            )[0]

            latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Sauvegarde des étapes/points intermédiaires (tous les 5 steps)
            if config.get("save_points", False) and (idx % 5 == 0 or idx == len(timesteps) - 1):
                interm_img = self.model.decode(latents)
                save_image_tensor(interm_img, f"{save_folder}/points/step_{idx:02d}_t{int(t)}.png")

        # 3. Sauvegarde de l'image finale
        final_img = self.model.decode(latents)
        save_image_tensor(final_img, f"{save_folder}/final_output.png")

        return final_img
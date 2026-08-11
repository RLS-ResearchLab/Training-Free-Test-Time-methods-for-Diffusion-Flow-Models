import os
import torch
from PIL import Image

class ModularSampler:
    def __init__(self, model_wrapper):
        self.model = model_wrapper

    @torch.no_grad()
    def sample(self, prompt: str, num_steps: int = 28, cfg_scale: float = 4.5, save_path: str = "samples/baseline.png"):
        # 1. Encodings & Latents
        prompt_embeds, neg_embeds, pooled, neg_pooled = self.model.encode_prompts(prompt)
        
        
        torch.manual_seed(42)
        latents = torch.randn(1, 16, 64, 64, device="cuda", dtype=self.model.dtype)

        
        self.model.scheduler.set_timesteps(num_steps)
        timesteps = self.model.scheduler.timesteps

       
        for t in timesteps:
            # Predict noise / velocity
            noise_pred = self.model.transformer(
                hidden_states=latents,
                timestep=t / 1000.0,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=False
            )[0]

            
            latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 4. Decode & Save Image
        image_tensor = self.model.decode(latents)
        self._save_tensor_as_image(image_tensor, save_path)
        return image_tensor

    def _save_tensor_as_image(self, tensor: torch.Tensor, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tensor = (tensor.squeeze(0).clamp(-1, 1) + 1) / 2.0
        tensor = (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        img = Image.fromarray(tensor)
        img.save(save_path)
        print(f"Saved test image to: {save_path}")
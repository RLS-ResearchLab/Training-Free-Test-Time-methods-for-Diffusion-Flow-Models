# src/rewards/clip_reward.py
import torch
import open_clip


class CLIPPromptReward:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, model_wrapper, latents, prompt: str) -> torch.Tensor:
        images = model_wrapper.decode(latents)
        images = (images.clamp(-1, 1) + 1) / 2  # -> [0, 1]
        clip_images = torch.nn.functional.interpolate(images, size=224, mode="bicubic")

        text_tokens = self.tokenizer([prompt]).to(self.device)
        image_features = self.model.encode_image(clip_images.to(self.device))
        text_features = self.model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T).squeeze(-1)  # [K]
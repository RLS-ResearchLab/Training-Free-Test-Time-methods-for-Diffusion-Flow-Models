import os
import time
import torch
from PIL import Image


class ComputeTracker:
    """
    Context manager that measures wall-clock time and peak GPU memory for a
    block of code (a full sampling run, typically). Use it as:

        with ComputeTracker() as tracker:
            ...do the denoising loop...
        tracker.elapsed_sec, tracker.peak_memory_mb
    """

    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.elapsed_sec = None
        self.peak_memory_mb = None

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.elapsed_sec = time.time() - self._start
        return False  # never swallow exceptions


_lpips_model = None  # lazy-loaded, shared across calls so we only pay the load cost once


def lpips_distance(image_a: torch.Tensor, image_b: torch.Tensor) -> float:
    """
    Perceptual distance between two decoded images (each shaped [1, 3, H, W]).
    Lower = more visually similar. Loads an AlexNet-backed LPIPS net lazily.

    Why LPIPS and not FID: FID compares the *distributions* of two large sets
    of images via Inception feature statistics (mean/covariance) — it isn't
    defined for a single pair of images. LPIPS is a pairwise perceptual metric,
    which is what's needed to quantify how much one sampling method changes
    the output relative to another, image-for-image.
    """
    global _lpips_model
    import lpips as lpips_lib  # local import: optional heavy dependency

    if _lpips_model is None:
        _lpips_model = lpips_lib.LPIPS(net="alex")
        if torch.cuda.is_available():
            _lpips_model = _lpips_model.cuda()
        _lpips_model.eval()

    device = next(_lpips_model.parameters()).device

    def _prep(img):
        img = img.detach().to(device)
        if img.min() >= 0.0:  # already in [0, 1] -> LPIPS wants roughly [-1, 1]
            img = img * 2.0 - 1.0
        return img.clamp(-1.0, 1.0)

    with torch.no_grad():
        dist = _lpips_model(_prep(image_a), _prep(image_b))
    return float(dist.mean().item())


@torch.no_grad()
def clip_align_score(reward_fn, image: torch.Tensor, prompt: str) -> float:
    """
    CLIP image-text alignment score for an already-decoded image (skips the
    re-decode-from-latents step that CLIPPromptReward.__call__ does, since by
    the time we're building the results table we already have the final
    image). Reuses the CLIP model/tokenizer already loaded inside reward_fn,
    so this doesn't load anything new.
    """
    images = (image.clamp(-1, 1) + 1) / 2
    clip_images = torch.nn.functional.interpolate(images, size=224, mode="bicubic")
    text_tokens = reward_fn.tokenizer([prompt]).to(reward_fn.device)
    image_features = reward_fn.model.encode_image(clip_images.to(reward_fn.device))
    text_features = reward_fn.model.encode_text(text_tokens)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return float((image_features @ text_features.T).squeeze().item())


def freeze(module: torch.nn.Module):
    """Gèle les paramètres d'un module."""
    for param in module.parameters():
        param.requires_grad = False

def save_image_tensor(tensor: torch.Tensor, path: str):
    """Convertit un tensor [-1, 1] ou [0, 1] et le sauvegarde en PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Normalisation [0, 1]
    tensor = tensor.detach().cpu().squeeze(0)
    if tensor.min() < 0:
        tensor = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
    
    tensor = (tensor * 255).permute(1, 2, 0).to(torch.uint8).numpy()
    img = Image.fromarray(tensor)
    img.save(path)
    print(f"Image enregistrée : {path}")
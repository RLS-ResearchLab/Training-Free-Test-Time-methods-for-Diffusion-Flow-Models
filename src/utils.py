import os
import torch
from PIL import Image

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
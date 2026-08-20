
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class DinoLPIPS(nn.Module):
    def __init__(self, model_name="dino_vits16", layers=(2, 5, 8, 11), device="cuda"):
        """
        model_name: torch.hub facebookresearch/dino checkpoint name.
        layers: transformer block indices to read features from
                (vits16 has 12 blocks, 0-indexed). Spread across depth,
                same idea as LPIPS sampling multiple conv stages.
        """
        super().__init__()
        self.device = device
        self.model = torch.hub.load("facebookresearch/dino:main", model_name)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layers = set(layers)
        self._feats = {}
        for i, blk in enumerate(self.model.blocks):
            if i in self.layers:
                blk.register_forward_hook(self._make_hook(i))

        self.resize = T.Resize((224, 224), antialias=True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device) 
    def _make_hook(self, idx):
        def hook(module, inp, out):
            self._feats[idx] = out  # (B, 1 + n_patches, C), token 0 = CLS
        return hook

    def _extract(self, img):
        x = self.resize(img)
        x = (x - self.mean) / self.std
        self._feats.clear()
        with torch.no_grad():
            self.model(x)
        return {idx: F.normalize(tok[:, 1:, :], dim=-1) for idx, tok in self._feats.items()}

    def forward(self, img1, img2):
        f1 = self._extract(img1.to(self.device))
        f2 = self._extract(img2.to(self.device))
        dist = 0.0
        for idx in self.layers:
            diff2 = (f1[idx] - f2[idx]) ** 2               # (B, n_patches, C)
            dist = dist + diff2.sum(dim=-1).mean(dim=1)     # mean over patch tokens
        return dist  # (B,)


def dino_lpips_distance(img1, img2, dino_model=None, device="cuda"):
    """Drop-in replacement for the existing `lpips_distance(img1, img2)`
    calls in main.py. img1/img2: (1, 3, H, W) tensors in [0, 1].

    Pass a pre-built `dino_model` (build once outside the experiment
    loop) to avoid reloading the ViT for every with/without pair.
    """
    if dino_model is None:
        dino_model = DinoLPIPS(device=device)
    with torch.no_grad():
        d = dino_model(img1, img2)
    return d.item()
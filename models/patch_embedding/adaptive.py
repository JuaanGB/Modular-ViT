import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import List
from models.patch_embedding.base import BasePatchEmbedding, PatchOutput
from models.ExecutionState import ExecutionState

class APTPatchEmbedding(BasePatchEmbedding):

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState, scales: int = 3, entropy_thresholds: List[float] = None):
        super().__init__(img_size, patch_size, in_channels, embed_dim, execution_state)
        self.scales = scales
        self.grid_size = img_size // patch_size
        self.max_possible_tokens = self.grid_size ** 2

        if entropy_thresholds is None:
            entropy_thresholds = [4.0, 5.75]
        if len(entropy_thresholds) != scales - 1:
            print(f"[-] ERROR: thresholds deben tener longitud {scales - 1}"); sys.exit(1)

        self.register_buffer("thresholds", torch.tensor(entropy_thresholds, dtype=torch.float32))
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.execution_state.grid_size = self.grid_size
        self.execution_state.embed_dim = embed_dim
        self.execution_state.max_tokens = self.max_possible_tokens


    @staticmethod
    def compute_entropy_map(x: torch.Tensor, patch_size: int, num_bins: int = 16):
        B, C, H, W = x.shape
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]) if C == 3 else x.mean(dim=1)
        gray_q = torch.clamp((gray * (num_bins - 1)).long(), 0, num_bins - 1)
        
        patches = gray_q.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        Bp, Hb, Wb, _, _ = patches.shape
        patches = patches.reshape(Bp, Hb, Wb, -1)

        counts = torch.zeros(Bp, Hb, Wb, num_bins, device=x.device)
        for b in range(num_bins):
            counts[..., b] = (patches == b).sum(dim=-1)

        probs = torch.clamp(counts / patches.shape[-1], min=1e-8)
        return -(probs * torch.log2(probs)).sum(dim=-1)


    def _process_region(self, img: torch.Tensor, x0: int, y0: int, scale: int, b: int, entropy_maps: dict[int, torch.Tensor], img_tokens: List[torch.Tensor], img_coords_raw: List[tuple[int, int]]):
        p = self.patch_size

        if scale == 0:
            patch = img[:, :, y0:y0 + p, x0:x0 + p]
            token = self.proj(patch).flatten(2).transpose(1, 2)
            img_tokens.append(token.squeeze(0))
            img_coords_raw.append((y0 // p, x0 // p))
            return

        current_size = p * (2 ** scale)
        block_i, block_j = y0 // current_size, x0 // current_size
        entropy = entropy_maps[scale][b, block_i, block_j]

        if entropy < self.thresholds[scale - 1]:
            patch = img[:, :, y0:y0 + current_size, x0:x0 + current_size]
            patch_resized = F.interpolate(patch, size=(p, p), mode="bilinear", align_corners=False)
            token = self.proj(patch_resized).flatten(2).transpose(1, 2)
            img_tokens.append(token.squeeze(0))
            img_coords_raw.append((y0 // p, x0 // p))
            return

        child_size = current_size // 2
        self._process_region(img, x0, y0, scale - 1, b, entropy_maps, img_tokens, img_coords_raw)
        self._process_region(img, x0 + child_size, y0, scale - 1, b, entropy_maps, img_tokens, img_coords_raw)
        self._process_region(img, x0, y0 + child_size, scale - 1, b, entropy_maps, img_tokens, img_coords_raw)
        self._process_region(img, x0 + child_size, y0 + child_size, scale - 1, b, entropy_maps, img_tokens, img_coords_raw)


    def forward(self, x: torch.Tensor) -> PatchOutput:
        B, C, H, W = x.shape
        p = self.patch_size
        
        entropy_maps = {s: self.compute_entropy_map(x, p * (2 ** s)) for s in range(1, self.scales)}
        batch_features, batch_coords = [], []

        for b in range(B):
            img = x[b:b + 1]
            img_tokens, img_coords_raw = [], []

            max_scale = self.scales - 1
            root_size = p * (2 ** max_scale)

            for y in range(0, H, root_size):
                for x_ in range(0, W, root_size):
                    self._process_region(img, x_, y, max_scale, b, entropy_maps, img_tokens, img_coords_raw)

            batch_features.append(torch.cat(img_tokens, dim=0))
            batch_coords.append(torch.tensor(img_coords_raw, device=x.device, dtype=torch.float32))

        max_tokens = max(t.shape[0] for t in batch_features)
        padded_features, padded_coords = [], []

        for feat, coord in zip(batch_features, batch_coords):
            diff = max_tokens - feat.shape[0]
            if diff > 0:
                feat = F.pad(feat, (0, 0, 0, diff), mode='constant', value=0.0)
                coord = F.pad(coord, (0, 0, 0, diff), mode='constant', value=-1.0)
            padded_features.append(feat)
            padded_coords.append(coord)

        features = torch.stack(padded_features, dim=0)
        coords = torch.stack(padded_coords, dim=0)
        self.execution_state.max_tokens = max_tokens

        return PatchOutput(features=features, coords=coords)


    @staticmethod
    def create_from_config(config: dict, img_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState) -> BasePatchEmbedding:
        size = config.get("size", 0)
        if size <= 0:
            print(f"[-] ERROR: El tamaño del patch debe ser superior a 0"); sys.exit(1)

        return APTPatchEmbedding(
            img_size=img_size, 
            patch_size=size, 
            in_channels=in_channels, 
            embed_dim=embed_dim, 
            execution_state=execution_state, 
            scales=config.get("apt_scales", 3), 
            entropy_thresholds=config.get("apt_thresholds", [4.0, 5.75])
            
        )
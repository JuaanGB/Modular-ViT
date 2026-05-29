import torch
import torch.nn as nn
from models.patch_embedding.base import BasePatchEmbedding

class VanillaPatchEmbedding(BasePatchEmbedding):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__(img_size, patch_size, in_channels, embed_dim)
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolución que hace el troceado y la proyección lineal a la vez
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrada x: [B, C, H, W]
        x = self.projection(x)  # Salida: [B, Embed_Dim, H/Patch, W/Patch]
        x = x.flatten(2)        # Salida: [B, Embed_Dim, Num_Patches]
        x = x.transpose(1, 2)   # Salida: [B, Num_Patches, Embed_Dim]
        return x
import torch
import torch.nn as nn

class BasePatchEmbedding(nn.Module):
    """
    Clase abstracta para transformar imágenes en parches/tokens secuenciales.
    Mecanismos futuros: Vanilla (Fixed), APT (Dinámico), MViT (Fractal).
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de imagen con forma [Batch, Channels, Height, Width]
        Returns:
            Tensor de tokens con forma [Batch, Num_Patches, Embed_Dim]
        """
        raise NotImplementedError
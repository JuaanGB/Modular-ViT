import torch
import torch.nn as nn
from dataclasses import dataclass
from models.ExecutionState import ExecutionState

@dataclass
# B: Tamaño del lote (número de imágenes)
# N: Número de parches por imagen
# D: Tamaño del parche aplanado (longitud del vector)
class PatchOutput:
    features: torch.Tensor  # Forma: [B, N, D] (Los parches aplanados)
    coords: torch.Tensor    # Forma: [B, N, D] (Las coordenadas i,j de cada parche)

class BasePatchEmbedding(nn.Module):
    """
    Clase abstracta para transformar imágenes en parches/tokens secuenciales.
    Mecanismos futuros: Vanilla (Fixed), APT (Dinámico), MViT (Fractal).
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.execution_state = execution_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de imagen con forma [Batch, Channels, Height, Width]
        Returns:
            Tensor de tokens con forma [Batch, Num_Patches, Embed_Dim]
        """
        raise NotImplementedError
    
    @staticmethod
    def create_from_config(config: dict, img_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState) -> "BasePatchEmbedding":
        """
        Args: 
            dict: Diccionario conteniendo toda la información del mecanismo de Patch Embedding
            img_size: Tamaño de la imagen
            in_channels: Número de canales de la imagen
            embed_dim: 
        Returns:
            Mecanismo de Patch Embedding parametrizado
        """
        raise NotImplementedError
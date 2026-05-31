import torch
import torch.nn as nn
import sys
from models.patch_embedding.base import BasePatchEmbedding, PatchOutput
from models.ExecutionState import ExecutionState

class VanillaPatchEmbedding(BasePatchEmbedding):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState):
        super().__init__(img_size, patch_size, in_channels, embed_dim, execution_state)
        
        self.grid_size = img_size // patch_size  # Ej: 224 // 16 = 14 parches por lado
        self.num_patches = self.grid_size ** 2   # 14 * 14 = 196 parches en total

        # Capa estándar para proyectar los parches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # --- AQUÍ CREAMOS LA RELACIÓN (I, J) ---
        # 1. Generamos rangos de 0 a grid_size-1 (ej: de 0 a 13)
        ax_i = torch.arange(self.grid_size)
        ax_j = torch.arange(self.grid_size)
        
        # 2. Creamos la malla/grid 2D. 
        # grid_i tendrá las filas y grid_j las columnas
        grid_i, grid_j = torch.meshgrid(ax_i, ax_j, indexing='ij')
        
        # 3. Las aplanamos y las juntamos en un tensor de forma [196, 2]
        # Cada fila de este tensor es un par [coordenada_i, coordenada_j]
        coords_base = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=-1) # [N, 2]

        # Lo registramos como buffer (así se mueve a la GPU con el modelo pero no se entrena)
        self.register_buffer('coords_base', coords_base)
        
        # Agrupamos la información global en esa estructura execution state
        self.execution_state.grid_size = self.grid_size
        self.execution_state.embed_dim = embed_dim
        self.execution_state.max_tokens = self.num_patches
       

    def forward(self, x: torch.Tensor) -> PatchOutput:
        # Recibe: [Nº imágenes, Nº canales, Altura, Anchura]
        B, C, H, W = x.shape
        
        # 1. Pasar por la convolución: [B, 3, 224, 224] -> [B, 768, 14, 14]
        x_proj = self.proj(x)
        
        # 2. Aplanar las dimensiones espaciales (14x14) y trasponer para el Transformer
        # [B, 768, 14, 14] -> [B, 768, 196] -> [B, 196, 768]
        features = x_proj.flatten(2).transpose(1, 2)
        
        # 3. Expandir la matriz de coordenadas para que coincida con el tamaño del Batch (B)
        # Pasa de [196, 2] a [B, 196, 2]
        coords = self.coords_base.unsqueeze(0).expand(B, -1, -1)
        
        # Devolvemos nuestro contenedor limpio
        return PatchOutput(features=features, coords=coords)
    
    @staticmethod
    def create_from_config(config: dict, img_size: int, in_channels: int, embed_dim: int) -> BasePatchEmbedding:
        """
        Args: 
            dict: Diccionario conteniendo toda la información del mecanismo de Patch Embedding
        Returns:
            Mecanismo de Patch Embedding estándar
        """

        # Obtenemos el tamaño del patch
        size = config.get("size", 0)
        if size <= 0:
            print(f"[-] ERROR: El tamaño del patch debe ser superior a 0")
            sys.exit(1)

        # Devolvemos el mecanismo de patch embedding con los parámetros
        return VanillaPatchEmbedding(
            img_size=img_size,
            patch_size=size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
import torch
import torch.nn as nn
import sys
from models.patch_embedding.base import BasePatchEmbedding, PatchOutput
from models.ExecutionState import ExecutionState

class OverlappingPatchEmbedding(BasePatchEmbedding):
    def __init__(self, img_size: int, patch_size: int, overlap_pixels: int, in_channels: int, embed_dim: int, execution_state: ExecutionState):
        """
        Args:
            img_size: Tamaño de la imagen de entrada (H y W idénticos).
            patch_size: Tamaño base del parche (determina el paso/stride).
            overlap_pixels: Cuántos píxeles se extiende el parche hacia los lados para solapar.
            in_channels: Canales de entrada (ej: 3 para RGB).
            embed_dim: Dimensión de la proyección (ej: 768).
            execution_state: Contenedor de estado global compartido.
        """
        super().__init__(img_size, patch_size, in_channels, embed_dim, execution_state)
        
        self.overlap_pixels = overlap_pixels
        
        # El stride coincide con el patch_size para mantener el avance de la rejilla estándar
        stride = patch_size
        
        # El tamaño del kernel aumenta hacia ambos lados debido al solapamiento
        kernel_size = patch_size + (2 * overlap_pixels)
        
        # El padding evita que perdamos resolución espacial en las esquinas y bordes de la imagen
        padding = overlap_pixels

        # Calculamos las dimensiones resultantes de la rejilla de tokens
        self.grid_size = ((img_size - kernel_size + (2 * padding)) // stride) + 1
        self.num_patches = self.grid_size ** 2

        # Capa de proyección convolucional con solapamiento y padding explícito
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        # --- REJILLA DE COORDENADAS (I, J) ---
        # Se hereda exactamente igual para garantizar la compatibilidad con WePE o AxialRoPE
        ax_i = torch.arange(self.grid_size)
        ax_j = torch.arange(self.grid_size)
        
        grid_i, grid_j = torch.meshgrid(ax_i, ax_j, indexing='ij')
        coords_base = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=-1) # [N, 2]

        self.register_buffer('coords_base', coords_base)
        
        # Sincronizamos las propiedades con el estado de ejecución global
        self.execution_state.grid_size = self.grid_size
        self.execution_state.embed_dim = embed_dim
        self.execution_state.max_tokens = self.num_patches

    def forward(self, x: torch.Tensor) -> PatchOutput:
        B, C, H, W = x.shape
        
        # 1. Convolución con solapamiento: [B, C, 224, 224] -> [B, embed_dim, grid_size, grid_size]
        x_proj = self.proj(x)
        
        # 2. Aplanamos y trasponemos: [B, embed_dim, grid_size, grid_size] -> [B, N, embed_dim]
        features = x_proj.flatten(2).transpose(1, 2)
        
        # 3. Expandimos coordenadas para el batch
        coords = self.coords_base.unsqueeze(0).expand(B, -1, -1)
        
        return PatchOutput(features=features, coords=coords)
    
    @staticmethod
    def create_from_config(config: dict, img_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState) -> BasePatchEmbedding:
        """
        Factoría para instanciar el mecanismo desde el fichero de configuración YAML.
        Esperará una estructura en el YAML como:
          patch_embedding:
            type: "overlapping"
            size: 16
            overlap_pixels: 3
        """
        size = config.get("size", 0)
        overlap_pixels = config.get("overlap_pixels", 0)

        if size <= 0:
            print(f"[-] ERROR: El tamaño del patch debe ser superior a 0")
            sys.exit(1)
            
        if overlap_pixels < 0:
            print(f"[-] ERROR: El solapamiento (overlap_pixels) no puede ser negativo")
            sys.exit(1)

        return OverlappingPatchEmbedding(
            img_size=img_size,
            patch_size=size,
            overlap_pixels=overlap_pixels,
            in_channels=in_channels,
            embed_dim=embed_dim,
            execution_state=execution_state
        )
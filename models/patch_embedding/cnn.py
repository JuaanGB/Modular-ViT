import torch
import torch.nn as nn
import sys
from models.patch_embedding.base import BasePatchEmbedding, PatchOutput
from models.ExecutionState import ExecutionState

class CNNPatchEmbedding(BasePatchEmbedding):
    def __init__(
        self, 
        img_size: int, 
        patch_size: int, 
        in_channels: int, 
        embed_dim: int, 
        kernel_size: int,  # Ahora se recibe desde el archivo de configuración
        execution_state: ExecutionState
    ):
        super().__init__(img_size, patch_size, in_channels, embed_dim, execution_state)
        
        # 1. Configuración de las Convoluciones Tempranas Aprendibles
        hidden_channels = 64 
        
        # Calculamos el padding dinámicamente para que la imagen no encoja (mismo ancho y alto)
        if kernel_size % 2 == 0:
            print(f"[-] WARNING: Se recomienda un kernel_size impar. Un tamaño de {kernel_size} puede alterar las dimensiones espaciales.")
        padding = (kernel_size - 1) // 2
        
        # Primera convolución aprendible
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.GELU()
        
        # Segunda convolución aprendible
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=hidden_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.act2 = nn.GELU()

        # 2. Configuración del Grid y Proyección de Parches
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels=hidden_channels, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

        # --- GENERACIÓN DE COORDENADAS (I, J) ---
        ax_i = torch.arange(self.grid_size)
        ax_j = torch.arange(self.grid_size)
        grid_i, grid_j = torch.meshgrid(ax_i, ax_j, indexing='ij')
        coords_base = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=-1) # [N, 2]

        self.register_buffer('coords_base', coords_base)
        
        # Actualizar el estado de ejecución global
        self.execution_state.grid_size = self.grid_size
        self.execution_state.embed_dim = embed_dim
        self.execution_state.max_tokens = self.num_patches

    def forward(self, x: torch.Tensor) -> PatchOutput:
        B, C, H, W = x.shape
        
        # Extraemos características locales con sesgo inductivo (parámetros aprendibles)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        
        # Proyección en parches lineales aplanados
        x_proj = self.proj(out)
        features = x_proj.flatten(2).transpose(1, 2)
        
        # Coordenadas espaciales
        coords = self.coords_base.unsqueeze(0).expand(B, -1, -1)
        
        return PatchOutput(features=features, coords=coords)
    
    @staticmethod
    def create_from_config(config: dict, img_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState) -> BasePatchEmbedding:
        """
        Instancia la capa leyendo los parámetros desde el diccionario de configuración.
        """
        # Tamaño del patch (Obligatorio)
        size = config.get("size", 0)
        if size <= 0:
            print(f"[-] ERROR: El tamaño del patch debe ser superior a 0")
            sys.exit(1)

        # Tamaño del kernel (Por defecto 3 si no se define en el fichero)
        kernel_size = config.get("kernel_size", 3)

        return CNNPatchEmbedding(
            img_size=img_size,
            patch_size=size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            execution_state=execution_state
        )
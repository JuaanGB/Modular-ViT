from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

@dataclass
class ExecutionState:
    """Clase central que viaja a través del forward con la información dinámica."""
    grid_size: Optional[int] = None       # Ej: 14 (se calcula en el forward)
    embed_dim: Optional[int] = None       # Ej: 768
    max_tokens: Optional[int] = None      # Nº máximo de tokens. Digo "máximo" porque hay mecanismo con número variable de tokens

    # --- NUEVOS ATRIBUTOS PARA EL SOPORTE DINÁMICO DE GCT (ICLR 2026) ---
    raw_images: Optional[torch.Tensor] = None   # Guarda el lote original [B, C, H, W]
    patch_layer: Optional[nn.Module] = None     # Guarda la referencia a la capa de proyección (Conv2d)
    patch_size: Optional[int] = None            # Guarda el tamaño PxP del parche (ej: 9 o 16)
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ExecutionState:
    """Clase central que viaja a través del forward con la información dinámica."""
    grid_size: Optional[int] = None       # Ej: 14 (se calcula en el forward)
    embed_dim: Optional[int] = None       # Ej: 768
    max_tokens: Optional[int] = None      # Nº máximo de tokens. Digo "máximo" porque hay mecanismo con número variable de tokens

    # Para Axial RoPE
    token_coords: Optional[torch.Tensor] = None
    head_dim: Optional[int] = None
    num_heads: Optional[int] = None

    # Para APT
    attn_mask: Optional[torch.Tensor] = None
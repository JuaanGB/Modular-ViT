import torch
import torch.nn as nn

class BaseTokenInjection(nn.Module):
    """
    Clase abstracta para gestionar tokens especiales al inicio de la secuencia.
    Mecanismos futuros: CLS Injection (Vanilla), None (para GAP/GMP).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de tokens [Batch, Num_Patches, Embed_Dim]
        Returns:
            Tensor modificado [Batch, Num_Patches (+ 1 si CLS), Embed_Dim]
        """
        raise NotImplementedError
        
    @property
    def has_cls_token(self) -> bool:
        """Indica a los módulos siguientes si el token CLS está presente."""
        raise NotImplementedError
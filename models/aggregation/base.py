import torch
import torch.nn as nn

class BaseAggregation(nn.Module):
    """
    Clase abstracta para colapsar la secuencia de tokens en un único vector de características.
    Mecanismos futuros: CLS extraction, GAP (Mean), GAP+GMP (Fusión).
    """
    def __init__(self, has_cls_token: bool):
        super().__init__()
        self.has_cls_token = has_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Secuencia de salida del Encoder [Batch, Num_Tokens, Embed_Dim]
        Returns:
            Vector global consolidado [Batch, Embed_Dim (o Embed_Dim * 2 si hay fusión)]
        """
        raise NotImplementedError
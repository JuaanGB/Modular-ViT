import torch
import torch.nn as nn
from models.aggregation.base import BaseAggregation

class CLSAggregation(BaseAggregation):
    def __init__(self, has_cls_token: bool = True):
        super().__init__(has_cls_token)
        assert has_cls_token, "CLSAggregation requiere obligatoriamente que exista un token CLS."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Num_Tokens, Embed_Dim]
        # Extraemos el primer token de la secuencia para todas las muestras del batch
        return x[:, 0, :] # Salida: [Batch, Embed_Dim]
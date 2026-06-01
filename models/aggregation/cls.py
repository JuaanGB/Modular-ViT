import torch
import torch.nn as nn
from models.aggregation.base import BaseAggregation
from models.ExecutionState import ExecutionState

class CLSAggregation(BaseAggregation):
    def __init__(self, has_cls_token: bool = True, execution_state: ExecutionState = None):
        super().__init__(has_cls_token, execution_state)
        assert has_cls_token, "CLSAggregation requiere obligatoriamente que exista un token CLS."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Num_Tokens, Embed_Dim]
        # Extraemos el primer token de la secuencia para todas las muestras del batch
        return x[:, 0, :] # Salida: [Batch, Embed_Dim]
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseAggregation:
        return CLSAggregation(True, execution_state)
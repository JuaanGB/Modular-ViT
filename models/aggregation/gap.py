import torch
import torch.nn as nn
from models.aggregation.base import BaseAggregation
from models.ExecutionState import ExecutionState

class GAPAggregation(BaseAggregation):
    def __init__(self, has_cls_token: bool = False, execution_state: ExecutionState = None):
        # Para GAP puro, asumimos que has_cls_token es False
        super().__init__(has_cls_token, execution_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Num_Tokens, Embed_Dim]
        
        # Calculamos la media a lo largo de la dimensión de los tokens (dim=1)
        # Salida: [Batch, Embed_Dim]
        return torch.mean(x, dim=1)
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseAggregation:
        return GAPAggregation(False, execution_state)
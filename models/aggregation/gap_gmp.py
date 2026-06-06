import torch
import torch.nn as nn
from models.aggregation.base import BaseAggregation
from models.ExecutionState import ExecutionState

class GAPGMPAggregation(BaseAggregation):
    def __init__(self, has_cls_token: bool = False, execution_state: ExecutionState = None):
        super().__init__(has_cls_token, execution_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Num_Tokens, Embed_Dim]
        
        # 1. Global Average Pooling -> [Batch, Embed_Dim]
        gap = torch.mean(x, dim=1)
        
        # 2. Global Max Pooling -> [Batch, Embed_Dim]
        gmp, _ = torch.max(x, dim=1)
        
        # 3. Fusionamos sumando y promediando (Element-wise)
        # Salida: [Batch, Embed_Dim] -> ¡Idéntico tamaño que CLSAggregation!
        out = (gap + gmp) * 0.5
        
        return out
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseAggregation:
        return GAPGMPAggregation(False, execution_state)
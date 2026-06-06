import torch
import torch.nn as nn
from models.aggregation.base import BaseAggregation
from models.ExecutionState import ExecutionState

class GCTAggregation(BaseAggregation):
    def __init__(self, has_cls_token: bool = False, execution_state: ExecutionState = None):
        # Para GCT pasamos has_cls_token=False, ya que no usa el CLS tradicional 
        # (aunque sume +1 token al estado, gestionamos el GCT de forma independiente)
        super().__init__(has_cls_token, execution_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Secuencia de salida del Encoder [Batch, Num_Tokens, Embed_Dim] 
               (Donde Num_Tokens incluye los N parches + el token GCT)
        Returns:
            Vector global consolidado [Batch, Num_Tokens * Embed_Dim]
        """
        B = x.shape[0]
        
        # Según el apartado 3.2 y 4 del paper (CPE): Concatena todos los tokens de la secuencia
        # Pasamos de [B, N+1, D] a un único vector plano por muestra: [B, (N+1) * D]
        flat_features = x.reshape(B, -1)
        
        return flat_features
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseAggregation:
        return GCTAggregation(
            has_cls_token=config.get("has_cls_token", False), 
            execution_state=execution_state
        )
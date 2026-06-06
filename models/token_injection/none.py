import torch
from models.token_injection.base import BaseTokenInjection, TokenStrategyOutput
from models.ExecutionState import ExecutionState

class IdentityTokenInjection(BaseTokenInjection):
    def __init__(self, execution_state: ExecutionState):
        super().__init__(execution_state)
        # No sumamos 1 a max_tokens porque no inyectamos ningún token extra
        pass

    def forward(self, patch_features: torch.Tensor, patch_coords: torch.Tensor) -> TokenStrategyOutput:
        # patch_features: [B, N, D]
        # patch_coords:   [B, N, 2]
        
        # Simplemente devolvemos los tensores tal y como entran, sin alterar dimensiones
        return TokenStrategyOutput(features=patch_features, coords=patch_coords)

    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseTokenInjection:
        return IdentityTokenInjection(execution_state=execution_state)
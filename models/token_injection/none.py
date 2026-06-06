import torch
from models.token_injection.base import BaseTokenInjection, TokenStrategyOutput
from models.ExecutionState import ExecutionState

class IdentityTokenInjection(BaseTokenInjection):
    def __init__(self, execution_state: ExecutionState):
        super().__init__(execution_state)
        # No sumamos 1 a max_tokens porque no inyectamos ningún token extra
        
        self.identity = torch.nn.Identity()

    def forward(self, patch_features: torch.Tensor, patch_coords: torch.Tensor) -> TokenStrategyOutput:
        # patch_features: [B, N, D]
        # patch_coords:   [B, N, 2]
        
        # Al pasar por self.identity, el trazador "ve" que el módulo se ejecutó
        # Esto es para evitar un warning en cada época
        features = self.identity(patch_features)
        return TokenStrategyOutput(features=features, coords=patch_coords)
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseTokenInjection:
        return IdentityTokenInjection(execution_state=execution_state)
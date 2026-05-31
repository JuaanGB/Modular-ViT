import torch
import torch.nn as nn
from models.token_injection.base import BaseTokenInjection, TokenStrategyOutput
from models.ExecutionState import ExecutionState

class CLSTokenInjection(BaseTokenInjection):
    def __init__(self, execution_state: ExecutionState):
        super().__init__(execution_state)
        # Creamos el parámetro del CLS token: un vector de tamaño [1, 1, D]
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.execution_state.embed_dim))
        self.execution_state.max_tokens += 1 # Con CLS, hay un token más por imagen

    def forward(self, patch_features: torch.Tensor, patch_coords: torch.Tensor) -> TokenStrategyOutput:
        # patch_features: [B, N, D]
        # patch_coords:   [B, N, 2]
        B = patch_features.shape[0]

        # 1. Expandimos el CLS token para todo el lote -> [B, 1, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # 2. Concatenamos el CLS token al principio de las características de la imagen
        # [B, 1, D] mezclado con [B, N, D] -> Da como resultado [B, N+1, D]
        extended_features = torch.cat((cls_tokens, patch_features), dim=1)

        # 3. Creamos las coordenadas para el CLS token: [-1, -1] para cada elemento del lote
        # Forma: [B, 1, 2]
        cls_coords = torch.full((B, 1, 2), fill_value=-1, dtype=patch_coords.dtype, device=patch_coords.device)

        # 4. Concatenamos las coordenadas del CLS al principio. Desplaza el resto automáticamente.
        # [B, 1, 2] mezclado con [B, N, 2] -> Da como resultado [B, N+1, 2]
        extended_coords = torch.cat((cls_coords, patch_coords), dim=1)

        return TokenStrategyOutput(features=extended_features, coords=extended_coords)

    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseTokenInjection:
        return CLSTokenInjection(execution_state=execution_state)
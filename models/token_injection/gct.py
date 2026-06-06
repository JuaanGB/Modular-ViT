import torch
import torch.nn as nn
import torch.nn.functional as F
from models.token_injection.base import BaseTokenInjection, TokenStrategyOutput
from models.ExecutionState import ExecutionState

class GCTTokenInjection(BaseTokenInjection):
    def __init__(self, execution_state: ExecutionState):
        super().__init__(execution_state)
        # Sumamos el token GCT al contador global
        self.execution_state.max_tokens += 1

    def forward(self, patch_features: torch.Tensor, patch_coords: torch.Tensor) -> TokenStrategyOutput:
        # patch_features: [B, N, D]
        # patch_coords:   [B, N, 2]
        B = patch_features.shape[0]

        # 1. Recuperamos de forma segura las variables del ExecutionState
        raw_images = self.execution_state.raw_images
        patch_layer = self.execution_state.patch_layer
        patch_size = self.execution_state.patch_size

        assert raw_images is not None, "GCT requiere que la imagen original esté en ExecutionState."
        assert patch_layer is not None, "GCT requiere la capa de proyección en ExecutionState."

        # 2. Redimensionamos la imagen al tamaño de un parche (Apartado 3.2 del Paper)
        gct_resized = F.interpolate(
            raw_images, 
            size=(patch_size, patch_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 3. Proyectamos usando la misma capa original y aplanamos -> [B, 1, D]
        gct_token = patch_layer(gct_resized).flatten(2).transpose(1, 2)
        
        # 4. Concatenamos el token al inicio de la secuencia -> [B, N+1, D]
        extended_features = torch.cat((gct_token, patch_features), dim=1)

        # 5. Creamos las coordenadas virtuales (-1, -1) para el GCT
        gct_coords = torch.full((B, 1, 2), fill_value=-1, dtype=patch_coords.dtype, device=patch_coords.device)
        extended_coords = torch.cat((gct_coords, patch_coords), dim=1)

        return TokenStrategyOutput(features=extended_features, coords=extended_coords)

    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BaseTokenInjection:
        return GCTTokenInjection(execution_state=execution_state)
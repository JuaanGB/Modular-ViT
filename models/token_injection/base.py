import torch
import torch.nn as nn
from dataclasses import dataclass
from models.ExecutionState import ExecutionState

@dataclass
class TokenStrategyOutput:
    features: torch.Tensor  # [B, N+1, D] -> Ya incluye el CLS token
    coords: torch.Tensor    # [B, N+1, 2]   -> Coordenadas movidas (+1 para el CLS)

class BaseTokenInjection(nn.Module):
    """
    Clase abstracta para gestionar tokens especiales al inicio de la secuencia.
    Mecanismos futuros: CLS Injection (Vanilla), None (para GAP/GMP).
    """
    def __init__(self, execution_state: ExecutionState):
        super().__init__()
        self.execution_state = execution_state
        self.lazy_loaded = False

    def lazy_load(self):
        "Establece como parámetros locales a la clase atributos globales de la ejecución. Es necesario"
        "llamar a esté método antes del forward"
        if not self.lazy_loaded:
            self.embed_dim = self.execution_state.embed_dim
            self.grid_size = self.execution_state.grid_size
            self.lazy_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de tokens [Batch, Num_Patches, Embed_Dim]
        Returns:
            Tensor modificado [Batch, Num_Patches (+ 1 si CLS), Embed_Dim]
        """
        raise NotImplementedError
    
    @staticmethod
    def create_from_config(config: dict) -> "BaseTokenInjection":
        raise NotImplementedError
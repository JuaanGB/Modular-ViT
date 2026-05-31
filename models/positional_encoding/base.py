import torch
import torch.nn as nn
from models.ExecutionState import ExecutionState 

class BasePositionalEncoding(nn.Module):
    """
    Clase abstracta para codificar la información espacial.
    Mecanismos futuros: Learnable/Sinusoidal (Suman a la entrada), 
                        Axial RoPE (No suma, opera dentro de la atención).
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
            self.max_tokens = self.execution_state.max_tokens
            self.lazy_loaded = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [Batch, Num_Tokens, Embed_Dim]
        Returns:
            Tensor modificado o idéntico [Batch, Num_Tokens, Embed_Dim]
        """
        raise NotImplementedError

    def get_rope_frequencies(self, *args, **kwargs):
        """
        Método auxiliar para que mecanismos como Axial RoPE puedan exportar
        sus frecuencias/matrices de rotación a las capas de atención internas.
        Por defecto devuelve None si es un método aditivo (Vanilla).
        """
        return None
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> "BasePositionalEncoding":
        raise NotImplementedError
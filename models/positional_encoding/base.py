import torch
import torch.nn as nn

class BasePositionalEncoding(nn.Module):
    """
    Clase abstracta para codificar la información espacial.
    Mecanismos futuros: Learnable/Sinusoidal (Suman a la entrada), 
                        Axial RoPE (No suma, opera dentro de la atención).
    """
    def __init__(self, embed_dim: int, max_tokens: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

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
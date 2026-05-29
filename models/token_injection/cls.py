import torch
import torch.nn as nn
from models.token_injection.base import BaseTokenInjection

class CLSTokenInjection(BaseTokenInjection):
    def __init__(self, embed_dim: int = 768):
        super().__init__(embed_dim)
        
        # Inicializamos el token CLS de forma aleatoria/aprendible
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Expandimos el token CLS para que coincida con el tamaño del Batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, Embed_Dim]
        
        # Concatenamos al inicio de la secuencia
        x = torch.cat((cls_tokens, x), dim=1) # [B, Num_Patches + 1, Embed_Dim]
        return x

    @property
    def has_cls_token(self) -> bool:
        return True
import torch
import torch.nn as nn
from models.positional_encoding.base import BasePositionalEncoding

class LearnablePositionalEncoding(BasePositionalEncoding):
    def __init__(self, embed_dim: int = 768, max_tokens: int = 197): 
        # 197 tokens = (224/16)^2 = 196 parches + 1 token CLS
        super().__init__(embed_dim, max_tokens)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Num_Tokens, Embed_Dim]
        # Sumamos los vectores de posición de forma aditiva
        return x + self.pos_embed
import torch
import torch.nn as nn
from models.encoder.attention import ModularAttention
from models.encoder.mlp import MLP

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = ModularAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, hidden_features, dropout)

    def forward(self, x: torch.Tensor, pos_frequencies=None) -> torch.Tensor:
        # Primer bloque residual: Pre-LayerNorm -> Atención
        x = x + self.attn(self.ln1(x), pos_frequencies=pos_frequencies)
        # Segundo bloque residual: Pre-LayerNorm -> MLP
        x = x + self.mlp(self.ln2(x))
        return x
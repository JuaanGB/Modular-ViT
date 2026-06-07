import torch
import torch.nn as nn

class ModularAttention(nn.Module):
    """
    Mecanismo de Multi-Head Self-Attention preparado para aceptar
    modificaciones posicionales rotacionales en el producto QK^T.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_frequencies=None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor [Batch, Num_Tokens, Embed_Dim]
            pos_frequencies: Frecuencias/Ángulos para RoPE. Si es None, actúa de forma clásica.
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Cada uno: [B, num_heads, N, head_dim]

        # --- AJUSTES EXTRA EN LA ATENCIÓN PARA AXIAL ROPE ---
        if pos_frequencies is not None:
            x_freqs = pos_frequencies["x_freqs"]
            y_freqs = pos_frequencies["y_freqs"]

            rotary_dim = self.head_dim // 2

            q_x = q[..., :rotary_dim]
            q_y = q[..., rotary_dim:]

            k_x = k[..., :rotary_dim]
            k_y = k[..., rotary_dim:]

            q_x = self.apply_rotary(q_x, x_freqs.unsqueeze(1))
            q_y = self.apply_rotary(q_y, y_freqs.unsqueeze(1))

            k_x = self.apply_rotary(k_x, x_freqs.unsqueeze(1))
            k_y = self.apply_rotary(k_y,y_freqs.unsqueeze(1))

            q = torch.cat([q_x, q_y], dim=-1)
            k = torch.cat([k_x, k_y], dim=-1)
        # -----------------------------------------------------------

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        # --- CONTROL DE MÁSCARA DINÁMICA ---
        if mask is not None:
            # Reajustamos la forma de la máscara de [B, N] a [B, 1, 1, N] para que se acople por broadcasting
            # Cambiamos los elementos True (padding) por un número muy bajo (-infinito o -1e9)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2) # Formato: [B, 1, 1, N]
            attn = attn.masked_fill(mask_expanded, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return self.proj_drop(out)
    
    def rotate_half(self, x):

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack(
            (-x2, x1),
            dim=-1
        ).flatten(-2)
    
    def apply_rotary(self, x, freqs):

        cos = freqs.cos()
        sin = freqs.sin()

        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)

        return x * cos + self.rotate_half(x) * sin
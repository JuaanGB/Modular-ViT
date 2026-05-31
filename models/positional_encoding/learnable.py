import torch
import torch.nn as nn
from models.positional_encoding.base import BasePositionalEncoding
from models.ExecutionState import ExecutionState

class LearnablePositionalEncoding(BasePositionalEncoding):
    def __init__(self, execution_state: ExecutionState): 
        # 197 tokens = (224/16)^2 = 196 parches + 1 token CLS
        super().__init__(execution_state)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.execution_state.max_tokens, self.execution_state.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, Num_Tokens, Embed_Dim]
        # Sumamos los vectores de posición de forma aditiva
        print(f"[+] LearnablePositionalEncoding received {self.max_tokens} tokens.")
        return x + self.pos_embed
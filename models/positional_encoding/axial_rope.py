import torch
from models.positional_encoding.base import BasePositionalEncoding
from models.ExecutionState import ExecutionState

class AxialRoPE(BasePositionalEncoding):

    def __init__(self, execution_state: ExecutionState, base: float = 10000.0):
        super().__init__(execution_state)
        self.base = base

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        B, N, _ = coords.shape

        return torch.zeros(
            B,
            N,
            self.execution_state.embed_dim,
            device=coords.device
        )

    def get_rope_frequencies(self):
        coords = self.execution_state.token_coords

        if coords is None:
            return None

        device = coords.device
        head_dim = self.execution_state.head_dim

        assert head_dim % 4 == 0

        rotary_dim = head_dim // 2

        inv_freq = 1.0 / (self.base ** (torch.arange(0, rotary_dim, 2, device=device ).float() / rotary_dim))

        x_pos = coords[..., 1].float()
        y_pos = coords[..., 0].float()

        x_freqs = torch.einsum("bn,d->bnd", x_pos, inv_freq)
        y_freqs = torch.einsum("bn,d->bnd",y_pos, inv_freq)

        return {
            "x_freqs": x_freqs,
            "y_freqs": y_freqs
        }

    @staticmethod
    def create_from_config(config, execution_state):
        return AxialRoPE(
            execution_state, 
            base=config.get("base", 10000.0)
        )
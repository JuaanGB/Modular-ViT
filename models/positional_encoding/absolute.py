import torch
import torch.nn as nn
from models.positional_encoding.base import BasePositionalEncoding
from models.ExecutionState import ExecutionState

class Absolute2DPositionalEncoding(BasePositionalEncoding):
    def __init__(self, execution_state: ExecutionState):
        super().__init__(execution_state)

        # En codificación absoluta 2D, solemos dividir la dimensión a la mitad:
        # Una mitad del embedding codifica la "i" y la otra mitad la "j"
        self.row_embed = nn.Embedding(
            self.execution_state.grid_size,
            self.execution_state.embed_dim // 2
        )

        self.col_embed = nn.Embedding(
            self.execution_state.grid_size,
            self.execution_state.embed_dim // 2
        )

        # El token CLS no representa ninguna posición espacial real dentro
        # de la imagen. Reservamos la coordenada (-1, -1) para identificarlo
        # y le asociamos una codificación posicional aprendible propia.
        #
        # Esta decisión es consistente con ViT: el CLS tiene un contenido
        # aprendible (cls_token) y también una posición aprendible.
        self.cls_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.execution_state.embed_dim)
        )

        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords tiene forma [B, N, 2]:
        # "Para cada imagen, cada token tiene un par de coordenadas"

        B, N, _ = coords.shape

        # Detectamos qué tokens son CLS.
        # Por convenio, el CLS utiliza la coordenada especial (-1, -1).
        cls_mask = (
            (coords[..., 0] == -1) &
            (coords[..., 1] == -1)
        )  # [B, N]

        # Inicializamos la salida final.
        pos_embeddings = torch.zeros(
            B,
            N,
            self.execution_state.embed_dim,
            device=coords.device,
            dtype=self.cls_pos_embed.dtype
        )

        # Procesamos únicamente los tokens que corresponden a patches reales.
        patch_mask = ~cls_mask

        if patch_mask.any():
            # Separamos las coordenadas i (filas) y j (columnas)
            i_indices = coords[..., 0][patch_mask].long()
            j_indices = coords[..., 1][patch_mask].long()

            # Buscamos sus respectivos embeddings vectoriales
            i_features = self.row_embed(i_indices)  # [num_patches, D//2]
            j_features = self.col_embed(j_indices)  # [num_patches, D//2]

            # Concatenamos ambos para recuperar la dimensión total D (768)
            patch_embeddings = torch.cat(
                [i_features, j_features],
                dim=-1
            )  # [num_patches, D]

            pos_embeddings[patch_mask] = patch_embeddings

        # Asignamos el embedding especial a todos los CLS tokens.
        pos_embeddings[cls_mask] = self.cls_pos_embed.squeeze(0).squeeze(0)

        return pos_embeddings
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BasePositionalEncoding:
        return Absolute2DPositionalEncoding(execution_state=execution_state)
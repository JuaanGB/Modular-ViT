import torch
import torch.nn as nn
from datetime import datetime

from models.patch_embedding.base import BasePatchEmbedding
from models.token_injection.base import BaseTokenInjection
from models.positional_encoding.base import BasePositionalEncoding
from models.aggregation.base import BaseAggregation
from models.encoder.attention import ModularAttention # El bloque encoder usará esta atención

class ModularViT(nn.Module):
    def __init__(
        self,
        patch_embedding: BasePatchEmbedding,
        token_injection: BaseTokenInjection,
        positional_encoding: BasePositionalEncoding,
        encoder_blocks: nn.ModuleList, # Lista de bloques que usan ModularAttention
        aggregation: BaseAggregation,
        num_classes: int
    ):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.token_injection = token_injection
        self.positional_encoding = positional_encoding
        self.encoder_blocks = encoder_blocks
        self.aggregation = aggregation
        
        # Capa final de clasificación (Ajustable dinámicamente según la salida de la agregación)
        # Si usas GAP+GMP, la agregación devolverá embed_dim * 2, por eso calculamos dinámicamente.
        # Para la tarea 1 asumimos que se inicializa correctamente según el módulo de agregación.
        self.head = nn.Linear(patch_embedding.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] (Imágenes de entrada)
        
        # --- PASO 1: Parcheado ---
        # Pasamos de imágenes a parches planos y sus coordenadas (i, j) originales
        # Salida: features [B, N, D] y coords [B, N, 2]
        patch_out = self.patch_embedding(x)
        
        # --- PASO 2: Estrategia de Tokens (¡Tu nuevo módulo!) ---
        # Añade el token [CLS] al principio y desplaza todas las coordenadas
        # Salida: features [B, N+1, D] y coords [B, N+1, 2] donde el CLS tiene (-1, -1)
        # Previa lazy_load para inicializar valores globales
        self.token_injection.lazy_load()
        token_out = self.token_injection(patch_out.features, patch_out.coords)
        
        # --- PASO 3: Codificación Posicional ---
        # El codificador recibe las coordenadas ya desplazadas [B, N', 2]
        # Devuelve un embedding para cada posición, incluyendo la del CLS (-1, -1)
        # Salida: [B, N+1, D]
        # Previa lazy_load para inicializar valores globales
        self.positional_encoding.lazy_load()
        pos_embeddings = self.positional_encoding(token_out.coords)
        
        # --- PASO 4: Combinación ---
        # Sumamos los embeddings de posición a los parches de características + CLS (si fuese la estrategia)
        x = token_out.features + pos_embeddings  # [B, N', D]
        
        # Extraer frecuencias si el método es relacional (como RoPE)
        rope_freqs = self.positional_encoding.get_rope_frequencies(x)

        # 4. Paso por el Transformer Encoder pasando el gancho de la posición
        for block in self.encoder_blocks:
            # Asumiendo que el bloque pasa las frecuencias a la capa ModularAttention
            x = block(x, pos_frequencies=rope_freqs)
            
        # 5. Agregación del contexto global
        global_features = self.aggregation(x)
        
        # 6. Clasificación final
        return self.head(global_features)
    
    def get_experiment_name(self, dataset: str) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        return (
            f"ViT_{dataset}_"
            f"embeddim{self.patch_embedding.embed_dim}_"
            f"{type(self.patch_embedding).__name__}_patchsize{self.patch_embedding.patch_size}_"
            f"{type(self.token_injection).__name__}_"
            f"{type(self.positional_encoding).__name__}_"
            f"{type(self.aggregation).__name__}_"
            f"{timestamp}"
        )
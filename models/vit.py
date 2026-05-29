import torch
import torch.nn as nn
from models.patch_embedding.base import BasePatchEmbedding
from models.token_injection.base import BaseTokenInjection
from models.positional_encoding.base import BasePositionalEncoding
from models.aggregation.base import BaseAggregation
from models.encoder.attention import ModularAttention # El bloque encoder usará esta atención

from models.vit import ModularViT
from models.patch_embedding.vanilla import VanillaPatchEmbedding
from models.token_injection.cls import CLSTokenInjection
from models.positional_encoding.learnable import LearnablePositionalEncoding
from models.encoder.encoder_block import TransformerEncoderBlock
from models.aggregation.cls import CLSAggregation

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
        # 1. Fragmentación en parches
        x = self.patch_embedding(x)
        
        # 2. Inyección de tokens ([CLS] o ninguno)
        x = self.token_injection(x)
        
        # 3. Codificación posicional (Aditiva o pasiva)
        x = self.positional_encoding(x)
        
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
    

def create_vanilla_vit_base(img_size: int = 224, num_classes: int = 10) -> ModularViT:
    """Instancia la arquitectura exacta de ViT-Base/16 original de 2020."""
    embed_dim = 768
    patch_size = 16
    
    patch_emb = VanillaPatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
    tok_inj = CLSTokenInjection(embed_dim=embed_dim)
    
    # Num patches (196) + 1 CLS = 197 tokens max
    pos_enc = LearnablePositionalEncoding(embed_dim=embed_dim, max_tokens=patch_emb.num_patches + 1)
    
    # ViT-Base cuenta con 12 bloques idénticos de codificación
    encoder_blocks = nn.ModuleList([
        TransformerEncoderBlock(embed_dim=embed_dim, num_heads=12, mlp_ratio=4.0)
        for _ in range(12)
    ])
    
    agg = CLSAggregation(has_cls_token=True)
    
    return ModularViT(
        patch_embedding=patch_emb,
        token_injection=tok_inj,
        positional_encoding=pos_enc,
        encoder_blocks=encoder_blocks,
        aggregation=agg,
        num_classes=num_classes
    )
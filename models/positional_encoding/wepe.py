import torch
import torch.nn as nn
import math
from models.positional_encoding.base import BasePositionalEncoding
from models.ExecutionState import ExecutionState

class Weierstrass2DPositionalEncoding(BasePositionalEncoding):
    def __init__(self, execution_state: ExecutionState, a: float = 0.5, b: float = 3.0, num_terms: int = 6):
        super().__init__(execution_state)
        
        self.a = a
        self.b = b
        self.num_terms = num_terms
        self.embed_dim = self.execution_state.embed_dim
        
        # Al igual que en tu código, dividimos el embedding: mitad para "i", mitad para "j"
        self.coord_dim = self.embed_dim // 2
        
        # Verificación de seguridad
        if self.embed_dim % 2 != 0:
            raise ValueError("El embed_dim debe ser par para dividirlo equivalentemente entre filas y columnas.")
            
        # Para generar las proyecciones de Weierstrass, necesitamos proyectar la coordenada
        # a lo largo de varias frecuencias. Generamos coeficientes internos fijos.
        # Repartiremos los términos fractales equitativamente para llenar el tamaño 'coord_dim'
        # Cada término 'n' generará proyecciones tanto en seno como en coseno.
        self.projections_per_term = self.coord_dim // self.num_terms
        
        if self.projections_per_term == 0:
            raise ValueError(f"num_terms ({num_terms}) es demasiado alto para la dimensión asignada ({self.coord_dim}).")

        # Registramos las potencias de 'a' y 'b' como buffers fijos (no entrenables)
        # n va desde 0 hasta num_terms - 1
        n = torch.arange(self.num_terms, dtype=torch.float32)
        a_pow = self.a ** n  # Amplitudes escaladas [num_terms]
        b_pow = (self.b ** n) * math.pi  # Frecuencias escaladas [num_terms]
        
        self.register_buffer('a_pow', a_pow)
        self.register_buffer('b_pow', b_pow)

        # Matriz de proyección lineal aleatoria (pero fija) para esparcir cada término fractal 
        # en las dimensiones correspondientes a los canales del embedding.
        # Esto evita patrones armónicos repetitivos y destructivos en los canales.
        self.register_buffer('projection_weights', torch.randn(self.num_terms, self.projections_per_term))

        # El token CLS sigue manteniendo su naturaleza abstracta y aprendible
        self.cls_pos_embed = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim)
        )
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def _get_weierstrass_encoding(self, coords_flat: torch.Tensor) -> torch.Tensor:
        """
        Calcula la embedding fractal de Weierstrass para un tensor de coordenadas unidimensionales.
        coords_flat: [num_patches] (valores de i o j indexados)
        """
        # Normalizamos las coordenadas al rango [0, 1] usando el tamaño del grid 
        # para que la codificación matemática sea estable independientemente de la resolución
        coords_norm = coords_flat.float() / float(self.execution_state.grid_size - 1)
        
        # Expandimos dimensiones para operar con los buffers: [num_patches, 1]
        coords_norm = coords_norm.unsqueeze(-1)
        
        # Evaluamos los términos de la serie: [num_patches, num_terms]
        # b_pow tiene forma [num_terms], por difusión resulta en la multiplicación de cada término
        angles = coords_norm * self.b_pow  
        
        # Aplicamos la función fractal base (Sumamos la contribución de componentes seno y coseno)
        # Amplitud * cos(frecuencia) -> [num_patches, num_terms]
        weierstrass_series = self.a_pow * torch.cos(angles) + self.a_pow * torch.sin(angles)
        
        # Proyectamos las propiedades fractales al sub-espacio de canales (features) asignado
        # [num_patches, num_terms] x [num_terms, projections_per_term] -> [num_patches, coord_dim_aproximado]
        features = torch.matmul(weierstrass_series, self.projection_weights)
        
        # Si por divisiones enteras nos faltan dimensiones para completar exactamente coord_dim, hacemos padding
        if features.shape[-1] < self.coord_dim:
            padding = self.coord_dim - features.shape[-1]
            features = torch.nn.functional.pad(features, (0, padding))
            
        return features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 2]
        B, N, _ = coords.shape

        # Cls token mask (-1, -1)
        cls_mask = (coords[..., 0] == -1) & (coords[..., 1] == -1)  # [B, N]

        # Inicializamos el contenedor del batch
        pos_embeddings = torch.zeros(
            B, N, self.embed_dim,
            device=coords.device,
            dtype=self.cls_pos_embed.dtype
        )

        patch_mask = ~cls_mask

        if patch_mask.any():
            # Extraemos coordenadas espaciales puras de los parches reales
            i_indices = coords[..., 0][patch_mask]
            j_indices = coords[..., 1][patch_mask]

            # Computamos la codificación matemática fractal de Weierstrass para filas y columnas
            i_features = self._get_weierstrass_encoding(i_indices)  # [num_patches, D//2]
            j_features = self._get_weierstrass_encoding(j_indices)  # [num_patches, D//2]

            # Concatenamos horizontalmente para restaurar la dimensión total (D)
            patch_embeddings = torch.cat([i_features, j_features], dim=-1)  # [num_patches, D]

            pos_embeddings[patch_mask] = patch_embeddings

        # Inyectamos el parámetro aprendido al CLS token
        pos_embeddings[cls_mask] = self.cls_pos_embed.squeeze(0).squeeze(0)

        return pos_embeddings
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BasePositionalEncoding:
        # Extraemos los hiperparámetros fractales del diccionario de configuración
        a = config.get("a", 0.5)
        b = config.get("b", 3.0)
        num_terms = config.get("num_terms", 6)
        
        return Weierstrass2DPositionalEncoding(
            execution_state=execution_state,
            a=a,
            b=b,
            num_terms=num_terms
        )
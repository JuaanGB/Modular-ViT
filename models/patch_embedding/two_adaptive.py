import torch
import torch.nn as nn
import sys
from models.patch_embedding.base import BasePatchEmbedding, PatchOutput
from models.ExecutionState import ExecutionState

class TwoAPTPatchEmbedding(BasePatchEmbedding):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, 
                 threshold: float, bins: int, execution_state: ExecutionState):
        super().__init__(img_size, patch_size, in_channels, embed_dim, execution_state)
        
        self.threshold = threshold
        self.bins = bins
        
        self.p_small = patch_size
        self.grid_small = img_size // self.p_small
        
        self.p_large = 2 * patch_size
        if img_size % self.p_large != 0:
            print(f"[-] ERROR: Tamaño de imagen ({img_size}) no divisible por escala grande ({self.p_large})")
            sys.exit(1)
            
        self.grid_large = img_size // self.p_large
        self.max_tokens = self.grid_small ** 2
        
        self.proj_small = nn.Conv2d(in_channels, embed_dim, kernel_size=self.p_small, stride=self.p_small)
        self.proj_large = nn.Conv2d(in_channels, embed_dim, kernel_size=self.p_large, stride=self.p_large)
        
        self.execution_state.grid_size = self.grid_small
        self.execution_state.embed_dim = embed_dim
        self.execution_state.max_tokens = self.max_tokens

    def _calculate_entropy_large_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula la entropía sobre parches grandes transformando primero a escala de grises
        y asumiendo que la imagen ya está normalizada en el rango [0, 1].
        """
        B, C, H, W = x.shape
        p = self.p_large
        g = self.grid_large
        
        # 1. Corrección 1: Transformación estricta a escala de grises (Luma)
        if C == 3:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x # Si ya es monocromática
            
        # 2. Descomponer la versión en grises en parches grandes
        patches = gray.view(B, 1, g, p, g, p).permute(0, 2, 4, 1, 3, 5)
        flattened_patches = patches.reshape(B, g, g, -1) # [B, g, g, p*p]
        
        # 3. Corrección 2: Discretización estable basada en rango [0, 1] fijo
        scaled = flattened_patches * (self.bins - 1)
        hist_indices = scaled.long().clamp(0, self.bins - 1)
        
        # 4. Histograma vectorizado (Cuidado con OOM en batches gigantescos, monitorizar)
        one_hot = torch.nn.functional.one_hot(hist_indices, num_classes=self.bins).float()
        counts = one_hot.sum(dim=-2) # [B, g, g, bins]
        
        # 5. Entropía de Shannon
        probs = counts / (flattened_patches.shape[-1])
        probs = torch.clamp(probs, min=1e-8)
        entropy = -(probs * torch.log2(probs)).sum(dim=-1) # [B, g, g]
        
        return entropy

    def forward(self, x: torch.Tensor) -> PatchOutput:
        B, C, H, W = x.shape
        device = x.device
        
        # 1. Proyecciones estándar
        feat_small_all = self.proj_small(x).flatten(2).transpose(1, 2) # [B, grid_small^2, D]
        feat_large_all = self.proj_large(x).flatten(2).transpose(1, 2) # [B, grid_large^2, D]
        
        # 2. Entropía y Máscara de decisión
        entropy_large = self._calculate_entropy_large_patches(x) # [B, grid_large, grid_large]
        keep_large_mask = (entropy_large < self.threshold).flatten(1) # [B, grid_large^2]
        
        # 3. VECTORIZACIÓN DE ÍNDICES
        # En lugar de ifs y loops, preparamos un mapeo estructural fijo.
        # Cada parche grande 'i' mapea a 4 parches pequeños específicos.
        idx_large_arr = torch.arange(self.grid_large ** 2, device=device)
        
        # Reconstruimos la correspondencia de índices pequeños para cada índice grande
        i_l = idx_large_arr // self.grid_large
        j_l = idx_large_arr % self.grid_large
        i_s = i_l * 2
        j_s = j_l * 2
        
        idx_s0 = i_s * self.grid_small + j_s
        idx_s1 = idx_s0 + 1
        idx_s2 = idx_s0 + self.grid_small
        idx_s3 = idx_s2 + 1
        
        # Índices pequeños asociados a cada bloque grande de tamaño [grid_large^2, 4]
        small_mappings = torch.stack([idx_s0, idx_s1, idx_s2, idx_s3], dim=-1)

        # 4. Asignación masiva paralela
        out_features = torch.zeros(B, self.max_tokens, self.embed_dim, device=device)
        out_coords = torch.zeros(B, self.max_tokens, 2, device=device)
        attn_mask = torch.ones(B, self.max_tokens, dtype=torch.bool, device=device)
        
        # Para coordinar de forma masiva, calculamos cuántos tokens escribe cada imagen
        # Un True (grande) = 1 token. Un False (pequeño) = 4 tokens.
        tokens_per_large = keep_large_mask.long() # [B, grid_large^2] -> 1 si es grande
        tokens_per_small = (~keep_large_mask).long() * 4 # 4 si es pequeño
        tokens_per_block = tokens_per_large + tokens_per_small # [B, grid_large^2]
        
        # El bucle de Batch se mantiene SOLO para la escritura final indexada, que es ultrarrápida
        # eliminando los bucles espaciales (i_l, j_l) y los appends de listas que eran el cuello de botella.
        for b in range(B):
            mask_b = keep_large_mask[b]
            
            # Extraer características válidas de golpe mediante máscaras booleanas
            large_feats = feat_large_all[b, mask_b] # [Num_Grandes, D]
            
            # Extraer los pequeños usando el mapa de índices precalculado
            small_indices_b = small_mappings[~mask_b].flatten()
            small_feats = feat_small_all[b, small_indices_b] # [Num_Pequeños * 4, D]
            
            # Coordenadas correspondientes vectorizadas
            # (Para máxima velocidad, puedes precalcular un grid base estático en __init__ 
            # y filtrar con mask_b de la misma forma que los features)
            
            # Combinamos de golpe en la secuencia final
            valid_tokens = torch.cat([large_feats, small_feats], dim=0)
            N_actual = valid_tokens.shape[0]
            
            out_features[b, :N_actual] = valid_tokens
            attn_mask[b, :N_actual] = False
            
            # Nota: Para las coordenadas, aplica la misma lógica de extracción masiva.

        self.execution_state.attn_mask = attn_mask
        return PatchOutput(features=out_features, coords=out_coords)

    @staticmethod
    def create_from_config(config: dict, img_size: int, in_channels: int, embed_dim: int, execution_state: ExecutionState) -> BasePatchEmbedding:
        size = config.get("size", 0)
        threshold = config.get("threshold", None)
        bins = config.get("bins", 16)
        
        if size <= 0:
            print(f"[-] ERROR: El tamaño del patch base ('size') debe ser superior a 0")
            sys.exit(1)
        if threshold is None:
            print(f"[-] ERROR: 2APT requiere definir un 'threshold' de entropía.")
            sys.exit(1)
            
        return TwoAPTPatchEmbedding(
            img_size=img_size,
            patch_size=size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            threshold=float(threshold),
            bins=int(bins),
            execution_state=execution_state
        )
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
        
        feat_small_all = self.proj_small(x).flatten(2).transpose(1, 2)
        feat_large_all = self.proj_large(x).flatten(2).transpose(1, 2)
        
        entropy_large = self._calculate_entropy_large_patches(x)
        keep_large_mask = entropy_large < self.threshold
        
        out_features = torch.zeros(B, self.max_tokens, self.embed_dim, device=device)
        out_coords = torch.zeros(B, self.max_tokens, 2, device=device)
        
        # Nueva máscara de atención: True en lo que es PADDING (ceros)
        # Inicializada a True (todo es padding por defecto)
        attn_mask = torch.ones(B, self.max_tokens, dtype=torch.bool, device=device)
        
        # Eliminamos por completo los bucles anidados i_l y j_l espaciales.
        # Solo dejamos el bucle del Batch (B) que suele ser pequeño (ej. 16, 32, 64)
        for b in range(B):
            tokens_list = []
            coords_list = [] 
            
            # Aplanamos la máscara de esta imagen para recorrerla en 1D rápido
            keep_large_flat = keep_large_mask[b].flatten()
            
            idx_large = 0
            for i_l in range(self.grid_large):
                for j_l in range(self.grid_large):
                    if keep_large_flat[idx_large]:
                        tokens_list.append(feat_large_all[b, idx_large])
                        coords_list.append([i_l * 2 + 0.5, j_l * 2 + 0.5])
                    else:
                        # Extraer los 4 índices pequeños correspondientes de golpe
                        i_s0, j_s0 = 2 * i_l, 2 * j_l
                        indices_s = [
                            (i_s0) * self.grid_small + j_s0,
                            (i_s0) * self.grid_small + (j_s0 + 1),
                            (i_s0 + 1) * self.grid_small + j_s0,
                            (i_s0 + 1) * self.grid_small + (j_s0 + 1)
                        ]
                        for idx_s in indices_s:
                            tokens_list.append(feat_small_all[b, idx_s])
                        
                        coords_list.append([float(i_s0), float(j_s0)])
                        coords_list.append([float(i_s0), float(j_s0 + 1)])
                        coords_list.append([float(i_s0 + 1), float(j_s0)])
                        coords_list.append([float(i_s0 + 1), float(j_s0 + 1)])
                        
                    idx_large += 1
            
            img_tokens = torch.stack(tokens_list, dim=0)
            img_coords = torch.tensor(coords_list, device=device, dtype=torch.float32) 
            
            N_actual = img_tokens.shape[0]
            out_features[b, :N_actual] = img_tokens
            out_coords[b, :N_actual] = img_coords
            
            # En las posiciones válidas, ponemos False (NO es padding, SÍ atender)
            attn_mask[b, :N_actual] = False

        # Guardamos la máscara en el estado de ejecución global para que el encoder la lea
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
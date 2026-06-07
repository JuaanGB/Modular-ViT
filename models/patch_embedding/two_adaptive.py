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
        
        # Corrección 9: Telemetría temporal para calibrar el Threshold
        if self.training and torch.rand(1).item() < 0.01: # Imprime el 1% de las veces para no saturar la consola
            print(f"[2APT Calibración] Media: {entropy.mean().item():.2f} | Mín: {entropy.min().item():.2f} | Máx: {entropy.max().item():.2f}")
            
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
        
        # Variables para métricas sugeridas en el Punto 8
        total_large_tokens = 0
        total_small_tokens = 0
        
        for b in range(B):
            tokens_list = []
            # Corrección 4: Almacenamos listas nativas de Python, no tensores individuales
            coords_list = [] 
            
            idx_large = 0
            for i_l in range(self.grid_large):
                for j_l in range(self.grid_large):
                    
                    if keep_large_mask[b, i_l, j_l]:
                        tokens_list.append(feat_large_all[b, idx_large])
                        coords_list.append([i_l * 2 + 0.5, j_l * 2 + 0.5]) # Coordenada float centradora
                        total_large_tokens += 1
                    else:
                        for offset_i in range(2):
                            for offset_j in range(2):
                                i_s = 2 * i_l + offset_i
                                j_s = 2 * j_l + offset_j
                                idx_small = i_s * self.grid_small + j_s
                                
                                tokens_list.append(feat_small_all[b, idx_small])
                                coords_list.append([float(i_s), float(j_s)])
                                total_small_tokens += 1
                                
                    idx_large += 1
            
            img_tokens = torch.stack(tokens_list, dim=0)
            # Corrección 4: Creamos un único tensor consolidado al final del bucle de la imagen
            img_coords = torch.tensor(coords_list, device=device, dtype=torch.float32) 
            
            N_actual = img_tokens.shape[0]
            out_features[b, :N_actual] = img_tokens
            out_coords[b, :N_actual] = img_coords
            
            # --- PRINT DE DEPURACIÓN LEVE ---
            # print(f"[2APT Debug] Img {b:02d} -> Grandes: {total_large_tokens:3d} | Pequeños: {total_small_tokens:3d} | Total Tokens: {N_actual}/{self.max_tokens}")
        
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
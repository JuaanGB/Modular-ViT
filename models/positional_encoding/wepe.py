import torch
import torch.nn as nn
import math
from models.positional_encoding.base import BasePositionalEncoding
from models.ExecutionState import ExecutionState

class Weierstrass2DPositionalEncoding(BasePositionalEncoding):
    def __init__(self, execution_state: ExecutionState, alpha_u: float = 0.5, alpha_v: float = 0.5, M_terms: int = 4, N_terms: int = 4):
        super().__init__(execution_state)
        
        self.embed_dim = self.execution_state.embed_dim
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.M_terms = M_terms
        self.N_terms = N_terms
        
        # Parámetro constante real fijo omega_1 (lemniscático estándar según el paper)
        # 2 * omega_1 ~ 5.244 -> omega_1 = 2.622
        self.omega_1 = 2.622
        
        # omega_3 es puramente imaginario en una red ortogonal, hacemos que su magnitud sea entrenable
        # Inicialmente simétrico a omega_1 para formar una red cuadrada perfecta
        initial_alpha_learn = math.log(math.exp(self.omega_1) - 1.0) # Inversa de softplus
        self.alpha_learn = nn.Parameter(torch.tensor(initial_alpha_learn, dtype=torch.float32))
        
        # Factor de escala adaptivo para la compresión tanh (inicializado en el estándar 0.15)
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.exp(0.15) - 1.0), dtype=torch.float32))
        
        # Pre-generación de los coeficientes de la red (m, n) excluyendo el origen (0,0)
        grid_m = []
        grid_n = []
        for m in range(-self.M_terms, self.M_terms + 1):
            for n in range(-self.N_terms, self.N_terms + 1):
                if m == 0 and n == 0:
                    continue
                grid_m.append(m)
                grid_n.append(n)
                
        # Registramos los coeficientes m y n como buffers fijos
        self.register_buffer('grid_m', torch.tensor(grid_m, dtype=torch.float32))
        self.register_buffer('grid_n', torch.tensor(grid_n, dtype=torch.float32))
        
        # Capa lineal modular para proyectar la característica geométrica de 4D al embed_dim del ViT
        self.proj_layer = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
        
        # El token CLS mantiene su naturaleza abstracta e independiente
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

    def _complex_inv_power(self, real: torch.Tensor, imag: torch.Tensor, power: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula de forma estable el recíproco de un número complejo elevado a una potencia (1 / z^p)."""
        # Magnitud al cuadrado
        r2 = real**2 + imag**2 + 1e-12
        r = torch.sqrt(r2)
        
        # Ángulo fase
        theta = torch.atan2(imag, real)
        
        # Por propiedad de De Moivre: 1 / z^p = r^(-p) * (cos(-p*theta) + i*sin(-p*theta))
        inv_r_pow = 1.0 / (r ** power)
        out_real = inv_r_pow * torch.cos(-power * theta)
        out_imag = inv_r_pow * torch.sin(-power * theta)
        return out_real, out_imag

    def _get_weierstrass_encoding(self, i_indices: torch.Tensor, j_indices: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Calcula de forma exacta las proyecciones complejas basadas en la función de Weierstrass 
        y su derivada matemática para la grilla bidimensional.
        """
        # 1. Normalización formal de coordenadas según ecuaciones del paper
        u = (j_indices.float() + 0.5) / float(W)
        v = (i_indices.float() + 0.5) / float(H)
        
        # 2. Recuperación de los semiperiodos (half-periods) matemáticos
        omega_3_imag = torch.nn.functional.softplus(self.alpha_learn)
        
        # 3. Mapeo al plano complejo z (z_real, z_imag)
        # z = alpha_u * u * 2*Re(omega_1) + i * alpha_v * v * 2*Im(omega_3)
        # Dado que omega_1 es real puro e omega_3 es imaginario puro:
        z_real = self.alpha_u * u * (2.0 * self.omega_1)
        z_imag = self.alpha_v * v * (2.0 * omega_3_imag)
        
        # Expandimos dimensiones para procesar por broadcasting junto a la red (grid_m, grid_n)
        # [num_patches, 1]
        z_real_flat = z_real.unsqueeze(-1)
        z_imag_flat = z_imag.unsqueeze(-1)
        
        # 4. Cálculo del término fundamental de polo en el origen (1/z^2 y -2/z^3)
        wp_origin_real, wp_origin_imag = self._complex_inv_power(z_real_flat, z_imag_flat, power=2)
        wd_origin_real, wd_origin_imag = self._complex_inv_power(z_real_flat, z_imag_flat, power=3)
        wd_origin_real = -2.0 * wd_origin_real
        wd_origin_imag = -2.0 * wd_origin_imag
        
        # 5. Red de periodos periódicos: omega_mn = 2*m*omega_1 + i * 2*n*omega_3_imag
        # grid_m, grid_n tienen forma [num_terms]
        omega_mn_real = 2.0 * self.grid_m * self.omega_1
        omega_mn_imag = 2.0 * self.grid_n * omega_3_imag
        
        # Coordenadas relativas a los nodos de la red: (z - omega_mn)
        z_minus_w_real = z_real_flat - omega_mn_real.unsqueeze(0)
        z_minus_w_imag = z_imag_flat - omega_mn_imag.unsqueeze(0)
        
        # Evaluamos los sumatorios truncados de la serie de Weierstrass
        # Términos de la función de Weierstrass p(z): 1 / (z - w)^2 - 1 / w^2
        inv_z_w_2_real, inv_z_w_2_imag = self._complex_inv_power(z_minus_w_real, z_minus_w_imag, power=2)
        inv_w_2_real, inv_w_2_imag = self._complex_inv_power(omega_mn_real, omega_mn_imag, power=2)
        
        wp_lattice_real = torch.sum(inv_z_w_2_real - inv_w_2_real, dim=-1, keepdim=True)
        wp_lattice_imag = torch.sum(inv_z_w_2_imag - inv_w_2_imag, dim=-1, keepdim=True)
        
        # Términos de la derivada p'(z): -2 / (z - w)^3
        inv_z_w_3_real, inv_z_w_3_imag = self._complex_inv_power(z_minus_w_real, z_minus_w_imag, power=3)
        wd_lattice_real = torch.sum(-2.0 * inv_z_w_3_real, dim=-1, keepdim=True)
        wd_lattice_imag = torch.sum(-2.0 * inv_z_w_3_imag, dim=-1, keepdim=True)
        
        # Agregamos componente del origen + sumatorios de red
        f1 = wp_origin_real + wp_lattice_real
        f2 = wp_origin_imag + wp_lattice_imag
        f3 = wd_origin_real + wd_lattice_real
        f4 = wd_origin_imag + wd_lattice_imag
        
        # Concatenamos para conformar el vector descriptor geométrico unificado de 4D: f = [f1, f2, f3, f4]
        f = torch.cat([f1, f2, f3, f4], dim=-1)
        
        # 6. Compresión adaptativa basada en tanh para garantizar estabilidad numérica ante polos
        alpha_scale = torch.nn.functional.softplus(self.alpha_raw)
        f_stable = torch.tanh(alpha_scale * f)
        
        # 7. Proyección final modular al hiperespacio d de canales (features) asignado al patch embedding
        features = self.proj_layer(f_stable)
        return features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, 2]
        B, N, _ = coords.shape

        # Máscara para segregar el token CLS mapeado convencionalmente en (-1, -1)
        cls_mask = (coords[..., 0] == -1) & (coords[..., 1] == -1)  # [B, N]

        # Inicialización del contenedor principal de embeddings para el lote procesado
        pos_embeddings = torch.zeros(
            B, N, self.embed_dim,
            device=coords.device,
            dtype=self.cls_pos_embed.dtype
        )

        patch_mask = ~cls_mask

        if patch_mask.any():
            # Extraemos las posiciones espaciales absolutas discreta de los parches
            i_indices = coords[..., 0][patch_mask]
            j_indices = coords[..., 1][patch_mask]
            
            # Dinámicamente inferimos H y W basándonos en los límites máximos observados en el estado de ejecución
            # Esto dota al módulo de total flexibilidad frente a cambios súbitos en la resolución de entrada (resolution-agnostic)
            H = int(i_indices.max().item()) + 1
            W = int(j_indices.max().item()) + 1

            # Computamos la codificación holística analítica de Weierstrass en 2D
            patch_embeddings = self._get_weierstrass_encoding(i_indices, j_indices, H, W)  # [num_patches, embed_dim]

            pos_embeddings[patch_mask] = patch_embeddings

        # Asignamos de forma segura el vector libre aprendido correspondiente al CLS token
        pos_embeddings[cls_mask] = self.cls_pos_embed.squeeze(0).squeeze(0)

        return pos_embeddings
    
    @staticmethod
    def create_from_config(config: dict, execution_state: ExecutionState) -> BasePositionalEncoding:
        # Extraemos de forma limpia los hiperparámetros de escala y términos de la red geométrica
        alpha_u = config.get("alpha_u", 0.5)
        alpha_v = config.get("alpha_v", 0.5)
        M_terms = config.get("M_terms", 4)
        N_terms = config.get("N_terms", 4)
        
        return Weierstrass2DPositionalEncoding(
            execution_state=execution_state,
            alpha_u=alpha_u,
            alpha_v=alpha_v,
            M_terms=M_terms,
            N_terms=N_terms
        )
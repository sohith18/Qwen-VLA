import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp

# ==============================================================================
# HELPERS
# ==============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half  = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args      = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega  = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega  = 1.0 / 10000 ** omega
    pos    = pos.reshape(-1)
    out    = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ==============================================================================
# DiTBlock — replaced timm.Attention with nn.MultiheadAttention
# Identical math, but supports need_weights=True for attention visualization
# ==============================================================================

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads   = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(
            hidden_size, num_heads,
            batch_first=True, dropout=0.0, bias=True,
        )
        self.norm2         = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim     = int(hidden_size * mlp_ratio)
        approx_gelu        = lambda: nn.GELU(approximate="tanh")
        self.mlp           = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim,
                                 act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, need_weights=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x_norm          = modulate(self.norm1(x), shift_msa, scale_msa)
        x_sa, attn_w    = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=need_weights,
            average_attn_weights=True,   # [B, T, T] averaged over heads
        )
        x = x + gate_msa.unsqueeze(1) * x_sa
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, attn_w   # attn_w is None when need_weights=False


# ==============================================================================
# Final Layer
# ==============================================================================

class ActionFinalLayer(nn.Module):
    def __init__(self, hidden_size, action_dim):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear           = nn.Linear(hidden_size, action_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ==============================================================================
# ActionDiT
# ==============================================================================

class ActionDiT(nn.Module):
    def __init__(
        self,
        action_dim=7,
        global_cond_dim=256,
        state_dim=16,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        action_horizon=16,
    ):
        super().__init__()
        self.action_dim     = action_dim
        self.action_horizon = action_horizon
        self.hidden_size    = hidden_size

        self.x_embedder   = nn.Linear(action_dim, hidden_size, bias=True)
        self.pos_embed     = nn.Parameter(
            torch.zeros(1, action_horizon, hidden_size), requires_grad=False
        )
        self.cond_seq_proj = nn.Linear(global_cond_dim, hidden_size)
        self.t_embedder    = TimestepEmbedder(hidden_size)
        self.state_mlp     = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = ActionFinalLayer(hidden_size, action_dim)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos       = np.arange(self.action_horizon)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.hidden_size, pos)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.cond_seq_proj.weight, std=0.02)
        nn.init.constant_(self.cond_seq_proj.bias,  0)
        nn.init.normal_(self.state_mlp[0].weight,  std=0.02)
        nn.init.constant_(self.state_mlp[0].bias,   0)
        nn.init.normal_(self.state_mlp[2].weight,  std=0.02)
        nn.init.constant_(self.state_mlp[2].bias,   0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias,   0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias,   0)

    def forward(self, sample, timestep, global_cond=None, states=None,
                return_unet_attn=False):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif timestep.dim() == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        x_actions  = self.x_embedder(sample) + self.pos_embed   # [B, 16, 384]
        x_vlm      = self.cond_seq_proj(global_cond)             # [B, T, 384]
        x_combined = torch.cat([x_vlm, x_actions], dim=1)        # [B, T+16, 384]
        vlm_seq_len = x_vlm.shape[1]

        t = self.t_embedder(timestep)
        c = t
        if states is not None:
            c = c + self.state_mlp(states)

        all_attn_weights = [] if return_unet_attn else None

        for block in self.blocks:
            x_combined, attn_w = block(
                x_combined, c,
                need_weights=return_unet_attn,
            )
            if return_unet_attn and attn_w is not None:
                all_attn_weights.append(attn_w)   # [B, T+16, T+16]

        x_out = x_combined[:, -self.action_horizon:, :]
        x_out = self.final_layer(x_out, c)

        if return_unet_attn:
            return x_out, all_attn_weights, vlm_seq_len
        return x_out
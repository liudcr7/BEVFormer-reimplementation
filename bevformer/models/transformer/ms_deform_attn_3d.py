from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.models.utils.builder import ATTENTION
except Exception:
    try:
        from mmdet.models.builder import ATTENTION
    except Exception:
        ATTENTION = None

@ATTENTION.register_module()
class MSDeformableAttention3D(nn.Module):
    """
    Multi-Scale Deformable Attention (image-plane sampling) adapted for BEVFormer.

    This implementation samples, per head and per level, `num_points` locations
    around provided reference points using learnable offsets and attention weights,
    then aggregates features via bilinear sampling (grid_sample) and projects to
    the output dimension.

    Expected inputs:
        - query: [B, Nq, C]
        - feats: list of L feature maps, each [B, C, H_l, W_l]
        - reference_points: [B, Nq, L, 2] in normalized coords [-1, 1] (x,y)

    Shapes are aligned with common Deformable-DETR style, simplified for clarity.
    """
    def __init__(self, embed_dims: int = 256, num_levels: int = 4, num_points: int = 8, num_heads: int = 8):
        super().__init__()
        assert embed_dims % num_heads == 0, 'embed_dims must be divisible by num_heads'
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        # initialize offsets in a circular pattern as in Deformable DETR
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * 3.14159265 / self.num_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)  # [H, 2]
        grid = (grid / grid.abs().max(-1, keepdim=True).values).view(self.num_heads, 1, 1, 2)
        grid = grid.repeat(1, self.num_levels, self.num_points, 1)  # [H, L, P, 2]
        with torch.no_grad():
            self.sampling_offsets.bias.zero_()
            self.sampling_offsets.bias.data = grid.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query: torch.Tensor, feats: List[torch.Tensor], reference_points: torch.Tensor) -> torch.Tensor:
        B, Nq, C = query.shape
        L = len(feats)
        assert L == self.num_levels, f"num_levels mismatch: feats={L}, expected={self.num_levels}"
        assert reference_points.shape == (B, Nq, L, 2), 'reference_points should be [B,Nq,L,2] in [-1,1]'

        # project values per level to head dims
        values = [self.value_proj(f.permute(0,2,3,1)).contiguous() for f in feats]  # list of [B, H, W, C]

        # predict offsets & weights from queries
        offset = self.sampling_offsets(query)  # [B, Nq, H*L*P*2]
        weight = self.attention_weights(query)  # [B, Nq, H*L*P]
        offset = offset.view(B, Nq, self.num_heads, L, self.num_points, 2)
        weight = weight.view(B, Nq, self.num_heads, L, self.num_points)
        weight = weight.softmax(dim=-1)

        # reference points -> per head/per point sampling coords
        # ref: [B,Nq,L,2] -> [B,Nq,1,L,1,2]
        ref = reference_points[:, :, None, :, None, :]  # broadcast to heads, points
        # offsets are in normalized coords; scale lightly by 0.5 / min(H,W) via tanh for stability
        samp = ref + torch.tanh(offset) * 0.25  # [B,Nq,H,L,P,2]

        # grid_sample expects coords in [-1,1] as (x,y) with shape [B, H_out, W_out, 2] or [B, N, 1, 2]
        # We will sample per level and per head, then aggregate with weights.
        out = query.new_zeros(B, Nq, self.num_heads, self.head_dim)
        for l, v in enumerate(values):
            # v: [B, H, W, C]; reshape to heads
            H_l, W_l = v.shape[1], v.shape[2]
            v = v.view(B, H_l, W_l, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # [B, H, heads, H_l, W_l, d]
            v = v.permute(0,1,3,4,2)  # [B, heads, H_l, W_l, d]

            # coords: [B,Nq,heads,points,2]
            coords = samp[:, :, :, l, :, :]  # [B,Nq,H,P,2]
            # flatten queries for sampling
            B_, N_, Hh, P, _ = coords.shape
            coords = coords.reshape(B_*Hh, N_*P, 1, 2)

            # sample from feature map v for each head separately
            v_rep = v.unsqueeze(1).repeat(1, Hh, 1, 1, 1)  # [B, heads, H_l, W_l, d] -> repeat along batch dim
            v_rep = v_rep.reshape(B_*Hh, H_l, W_l, self.head_dim).permute(0,3,1,2)  # [B*Hh, d, H_l, W_l]
            sampled = F.grid_sample(v_rep, coords, mode='bilinear', padding_mode='zeros', align_corners=True)
            # sampled: [B*Hh, d, N_*P, 1]
            sampled = sampled.squeeze(-1).permute(0,2,1)  # [B*Hh, N_*P, d]
            sampled = sampled.view(B_, Hh, N_, P, self.head_dim).permute(0,2,1,3,4)  # [B, Nq, heads, P, d]

            # apply attention weights and sum over points
            w = weight[:, :, :, l, :].unsqueeze(-1)  # [B, Nq, heads, P, 1]
            agg = (sampled * w).sum(dim=3)  # [B, Nq, heads, d]
            out = out + agg

        out = out.reshape(B, Nq, C)
        out = self.output_proj(out)
        return out

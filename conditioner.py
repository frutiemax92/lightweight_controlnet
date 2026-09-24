import math
from dataclasses import dataclass
from typing import Optional

import torch

from lightweight_controlnet.config import ReferenceControlConfig
from lightweight_controlnet.reference_encoder import ReferenceFeatures


@dataclass
class ReferenceConditioning:
    """What every rank modulator of the model reads. Computed once per forward pass."""
    tokens: torch.Tensor                            # (batch, num_tokens, cond_dim)
    global_vector: torch.Tensor                     # (batch, cond_dim)
    padding_mask: Optional[torch.Tensor] = None     # (batch, num_tokens), True on padded tokens


def sincos_2d_embedding(grid: tuple[int, int], dim: int, device, dtype) -> torch.Tensor:
    """
    Factorized 2d sin-cos embedding of a (patches_h, patches_w) grid, flattened row major.

    Aspect ratio handling, part three: the grid itself carries the shape of the reference, so the
    modulators see where a feature sits in a 52x26 reference and not only its index in a sequence.
    """
    def axis_embedding(length, axis_dim):
        positions = torch.arange(length, device=device, dtype=torch.float32)
        # normalized coordinates, so a position means the same thing at every grid size
        positions = positions / max(length - 1, 1)
        frequencies = torch.exp(torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)
                                * (-math.log(10000.0) / axis_dim))
        angles = positions[:, None] * frequencies[None] * length
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    half = dim // 2
    assert half % 2 == 0, 'cond_dim must be a multiple of 4 for the 2d sin-cos embedding'
    patches_h, patches_w = grid
    embedding_h = axis_embedding(patches_h, half)[:, None].expand(patches_h, patches_w, half)
    embedding_w = axis_embedding(patches_w, half)[None, :].expand(patches_h, patches_w, half)
    return torch.cat([embedding_h, embedding_w], dim=-1).reshape(patches_h * patches_w, dim).to(dtype)


class ReferenceConditioner(torch.nn.Module):
    """
    The shared, trainable part of the reference path. It runs once per step, not once per layer,
    which is what keeps the memory cost of the whole thing close to a plain LoRA run.

    It resamples the variable length patch grid into a fixed, small set of latent tokens, so the
    per-adapter attention is O(rank * num_latents) whatever the resolution of the reference is.
    """
    def __init__(self, config: ReferenceControlConfig):
        super().__init__()
        self.config = config
        cond_dim = config.cond_dim

        self.input_norm = torch.nn.LayerNorm(config.reference_dim)
        self.input_proj = torch.nn.Linear(config.reference_dim, cond_dim)
        self.global_proj = torch.nn.Linear(config.reference_dim, cond_dim)

        # the aspect ratio of the reference, injected the way sdxl injects its size conditioning
        self.aspect_proj = torch.nn.Sequential(
            torch.nn.Linear(2, cond_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(cond_dim, cond_dim),
        )

        if config.num_latents > 0:
            self.latents = torch.nn.Parameter(torch.randn(config.num_latents, cond_dim) * 0.02)
            self.latent_norm = torch.nn.LayerNorm(cond_dim)
            self.resampler = torch.nn.MultiheadAttention(cond_dim, config.num_heads, batch_first=True)
        else:
            self.latents = None

        self.output_norm = torch.nn.LayerNorm(cond_dim)

    def forward(self, features: ReferenceFeatures) -> ReferenceConditioning:
        parameter = next(self.parameters())
        features = features.to(dtype=parameter.dtype)

        tokens = self.input_proj(self.input_norm(features.patch_tokens))
        tokens = tokens + sincos_2d_embedding(features.grid, self.config.cond_dim,
                                              tokens.device, tokens.dtype)[None]

        log_ratio = torch.log(features.aspect_ratio.clamp_min(1e-3))
        num_patches = features.grid[0] * features.grid[1]
        shape_vector = torch.stack([log_ratio, torch.full_like(log_ratio, math.log(num_patches))], dim=-1)
        global_vector = self.global_proj(features.cls_token) + self.aspect_proj(shape_vector)

        padding_mask = features.padding_mask
        if self.latents is not None:
            latents = self.latent_norm(self.latents)[None].expand(tokens.shape[0], -1, -1)
            tokens, _ = self.resampler(latents, tokens, tokens, key_padding_mask=padding_mask,
                                       need_weights=False)
            # the latents are a fixed length, the padding is resolved by the resampler
            padding_mask = None

        return ReferenceConditioning(tokens=self.output_norm(tokens),
                                     global_vector=global_vector,
                                     padding_mask=padding_mask)

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from lightweight_controlnet.config import FourierControlConfig
from lightweight_controlnet.reference_encoder import ReferenceFeatures


@dataclass
class ReferenceConditioning:
    """What every controlled layer of the model reads. Computed once per forward pass.

    It is one tensor: a (batch, cond_dim) summary of the reference image. Each controlled layer
    projects it into its own reference block of z, so this vector is the entire channel the
    reference has into the model - widening it widens that channel everywhere at once, and costs
    cond_dim * n_reference per layer.

    There is no null version of this. A step with no reference image is `None`, and a `None` fills
    the reference block of z with zeros, which is the plain FourierFT path.
    """
    vector: torch.Tensor        # (batch, cond_dim)

    def to(self, *args, **kwargs):
        return ReferenceConditioning(vector=self.vector.to(*args, **kwargs))


class FieldNorm(torch.nn.Module):
    """Centre and scale a whole token field, instead of each token on its own.

    A LayerNorm here is wrong twice over, and both mistakes destroy exactly the signal the patch
    field exists to carry.

    It removes each token's mean over the channels, which leaves the mean *over patches* untouched -
    and that component is the one saying nothing about what is where. In a dinov2 feature field it
    is as large as the part that does (measured at ||mean|| 32 against ||deviation|| 38), and a
    projection with no reason to prefer the deviation will happily amplify it instead.

    And it rescales every token to the same length, which erases the contrast a reference is made
    of. On a shape mask that is 96% background most patches carry nothing; normalizing them up to
    match the few that carry the shape is the opposite of what a layer reading this needs to see.

    So the mean is taken across the patch axis and one scale is shared by the whole field. A token
    that has nothing to say comes out small, and one that has something comes out large.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, tokens: torch.Tensor, padding_mask=None) -> torch.Tensor:
        if padding_mask is None:
            mean = tokens.mean(dim=1, keepdim=True)
            centred = tokens - mean
            scale = centred.pow(2).mean(dim=(1, 2), keepdim=True)
        else:
            # a padded patch is not part of the reference, so it is in neither the mean nor the scale
            keep = (~padding_mask)[..., None].to(tokens.dtype)
            counts = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (tokens * keep).sum(dim=1, keepdim=True) / counts
            centred = (tokens - mean) * keep
            scale = centred.pow(2).sum(dim=(1, 2), keepdim=True) / (counts.squeeze(-1)[..., None] * tokens.shape[-1])
        return centred / scale.clamp_min(1e-12).sqrt() * self.weight


class ReferenceConditioner(torch.nn.Module):
    """
    The shared, trainable part of the control path. It runs once per step, not once per layer, which
    is what keeps the whole mechanism close to the cost of the frozen model plus one FFT per layer.

    It turns the frozen dinov2 output into the (batch, cond_dim) vector every controlled layer
    projects into its own spectrum. Three things go into it:

    the patch field, average pooled onto a pool_grid x pool_grid grid and flattened. Pooling rather
    than taking a global mean is what keeps a coarse sense of layout - sky up, ground down - in a
    vector of fixed width. It does not make the control path spatial: z' is one number per output
    channel and lands uniformly over the frame whatever is in this vector. What it buys is a
    reference description that can distinguish two images with the same average content.

    the cls token, which is dinov2's own summary and carries what the pooling averages away.

    and the shape of the reference - its aspect ratio and how many patches it was read at - injected
    the way sdxl injects its size conditioning, because the pooled grid is the same size whether the
    reference was square or a panorama.
    """
    def __init__(self, config: FourierControlConfig):
        super().__init__()
        self.config = config
        cond_dim = config.cond_dim
        grid = config.pool_grid

        self.pool_grid = (grid, grid)

        self.input_norm = torch.nn.LayerNorm(config.reference_dim)
        self.token_proj = torch.nn.Linear(config.reference_dim, cond_dim)
        # the field is normalized after the projection rather than before, see FieldNorm: what the
        # pooling must not average away is where the reference differs from itself
        self.field_norm = FieldNorm(cond_dim)
        self.pool_proj = torch.nn.Linear(cond_dim * grid * grid, cond_dim)

        self.cls_proj = torch.nn.Linear(config.reference_dim, cond_dim)
        self.shape_proj = torch.nn.Sequential(
            torch.nn.Linear(2, cond_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(cond_dim, cond_dim),
        )

        self.mlp_norm = torch.nn.LayerNorm(cond_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, 4 * cond_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * cond_dim, cond_dim),
        )
        # a normalized summary, so the per-layer projections that read it see an input of a size
        # they can be initialized for whatever the reference encoder's own scale happens to be
        self.out_norm = torch.nn.LayerNorm(cond_dim)

    def _pooled(self, features: ReferenceFeatures) -> torch.Tensor:
        tokens = self.field_norm(self.token_proj(self.input_norm(features.patch_tokens)),
                                 features.padding_mask)
        batch, _, dim = tokens.shape
        patches_h, patches_w = features.grid
        field = tokens.transpose(1, 2).reshape(batch, dim, patches_h, patches_w)
        if features.padding_mask is None:
            pooled = F.adaptive_avg_pool2d(field, self.pool_grid)
        else:
            # a padded patch contributes to neither the sum nor the count of the cell it lands in
            keep = (~features.padding_mask).to(tokens.dtype).reshape(batch, 1, patches_h, patches_w)
            pooled = (F.adaptive_avg_pool2d(field * keep, self.pool_grid)
                      / F.adaptive_avg_pool2d(keep, self.pool_grid).clamp_min(1e-6))
        return pooled.flatten(1)

    def forward(self, features: ReferenceFeatures) -> ReferenceConditioning:
        parameter = next(self.parameters())
        features = features.to(dtype=parameter.dtype)

        log_ratio = torch.log(features.aspect_ratio.clamp_min(1e-3))
        num_patches = features.grid[0] * features.grid[1]
        shape_vector = torch.stack([log_ratio, torch.full_like(log_ratio, math.log(num_patches))], dim=-1)

        vector = (self.pool_proj(self._pooled(features))
                  + self.cls_proj(self.input_norm(features.cls_token))
                  + self.shape_proj(shape_vector))
        vector = vector + self.mlp(self.mlp_norm(vector))
        return ReferenceConditioning(vector=self.out_norm(vector))

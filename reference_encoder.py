import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from lightweight_controlnet.config import ReferenceControlConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ReferenceFeatures:
    """
    Output of the frozen reference encoder. Everything downstream only sees this,
    so the features can just as well be read back from a precomputed dataset.
    """
    patch_tokens: torch.Tensor          # (batch, num_patches, reference_dim)
    cls_token: torch.Tensor             # (batch, reference_dim)
    grid: tuple[int, int]               # (patches_h, patches_w) of the batch
    aspect_ratio: torch.Tensor          # (batch,) width / height of the original images
    padding_mask: Optional[torch.Tensor] = None     # (batch, num_patches), True on padded patches

    def to(self, *args, **kwargs):
        return ReferenceFeatures(
            patch_tokens=self.patch_tokens.to(*args, **kwargs),
            cls_token=self.cls_token.to(*args, **kwargs),
            grid=self.grid,
            aspect_ratio=self.aspect_ratio.to(*args, **kwargs),
            padding_mask=None if self.padding_mask is None else self.padding_mask.to(self.patch_tokens.device),
        )


def resolve_grid(height: int,
                 width: int,
                 patch_size: int = 14,
                 pixel_budget: int = 518,
                 min_patches_per_side: int = 8) -> tuple[int, int]:
    """
    Aspect ratio handling, part one.

    Instead of the square resize + center crop of the default dinov2 processor, we keep the
    aspect ratio and spend a constant pixel budget on it: the reference is resized so that it
    covers about pixel_budget**2 pixels with both sides a multiple of the patch size. Dinov2
    interpolates its position embeddings to whatever grid it receives, so a 37x37, a 52x26 or
    a 26x52 grid are all valid inputs and the patch grid stays a faithful map of the reference.
    """
    ratio = width / height
    target_patches = (pixel_budget / patch_size) ** 2

    patches_h = math.sqrt(target_patches / ratio)
    patches_w = target_patches / max(patches_h, 1e-6)

    patches_h = max(min_patches_per_side, int(round(patches_h)))
    patches_w = max(min_patches_per_side, int(round(patches_w)))
    return patches_h, patches_w


def preprocess_reference(images: Sequence[torch.Tensor],
                         config: ReferenceControlConfig,
                         device=None,
                         dtype=torch.float32):
    """
    Turns a list of float tensors in [0, 1] of shape (3, h, w) - each one possibly with its own
    aspect ratio - into a batch ready for the encoder.

    Aspect ratio handling, part two. Images of the batch that resolve to the same patch grid are
    stacked as they are. When they do not, we resize every image to the grid of the largest one and
    mark the patches that fall outside its own grid as padding, so the cross attention never reads
    them. Feeding the reference through an aspect ratio bucket sampler - the batch layout this repo
    already uses for the diffusion path - keeps the padding branch unused.
    """
    grids = [resolve_grid(image.shape[-2], image.shape[-1],
                          config.patch_size, config.pixel_budget, config.min_patches_per_side)
             for image in images]
    aspect_ratio = torch.tensor([image.shape[-1] / image.shape[-2] for image in images], dtype=torch.float32)

    grid_h = max(grid[0] for grid in grids)
    grid_w = max(grid[1] for grid in grids)
    uniform = all(grid == (grid_h, grid_w) for grid in grids)

    pixels = []
    for image, grid in zip(images, grids):
        own_h, own_w = (grid if uniform else (min(grid[0], grid_h), min(grid[1], grid_w)))
        resized = F.interpolate(image[None].float(),
                                size=(own_h * config.patch_size, own_w * config.patch_size),
                                mode='bicubic',
                                align_corners=False,
                                antialias=True).clamp(0, 1)
        if (own_h, own_w) != (grid_h, grid_w):
            resized = F.pad(resized, (0, (grid_w - own_w) * config.patch_size,
                                      0, (grid_h - own_h) * config.patch_size))
        pixels.append(resized[0])

    pixel_values = torch.stack(pixels)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std

    padding_mask = None
    if not uniform:
        padding_mask = torch.ones(len(images), grid_h, grid_w, dtype=torch.bool)
        for index, grid in enumerate(grids):
            padding_mask[index, :min(grid[0], grid_h), :min(grid[1], grid_w)] = False
        padding_mask = padding_mask.flatten(1)

    if device is not None:
        pixel_values = pixel_values.to(device)
        aspect_ratio = aspect_ratio.to(device)
        padding_mask = None if padding_mask is None else padding_mask.to(device)
    return pixel_values.to(dtype), (grid_h, grid_w), aspect_ratio, padding_mask


class ReferenceEncoder(torch.nn.Module):
    """
    Frozen dinov2 wrapper. It never holds gradients, it can be kept in bfloat16 next to the frozen
    base model, and it can be dropped entirely at training time by feeding ReferenceFeatures that
    were precomputed offline.
    """
    def __init__(self, config: ReferenceControlConfig, model=None):
        super().__init__()
        self.config = config
        if model is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(config.reference_model)
        self.model = model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Sequence[torch.Tensor]) -> ReferenceFeatures:
        parameter = next(self.model.parameters())
        pixel_values, grid, aspect_ratio, padding_mask = preprocess_reference(
            images, self.config, device=parameter.device, dtype=parameter.dtype)

        hidden_states = self.model(pixel_values).last_hidden_state
        return ReferenceFeatures(patch_tokens=hidden_states[:, 1:],
                                 cls_token=hidden_states[:, 0],
                                 grid=grid,
                                 aspect_ratio=aspect_ratio,
                                 padding_mask=padding_mask)

from typing import Optional

import torch

from lightweight_controlnet.config import ReferenceControlConfig
from lightweight_controlnet.conditioner import ReferenceConditioning


class ConditioningContext:
    """
    Holds the conditioning of the current step. Every modulator of the model points to the same
    instance, so the reference features travel to the adapters without threading an extra argument
    through the whole base model.
    """
    def __init__(self):
        self.current: Optional[ReferenceConditioning] = None

    def set(self, conditioning: Optional[ReferenceConditioning]):
        self.current = conditioning

    def clear(self):
        self.current = None


class RankModulator(torch.nn.Module):
    """
    The controlnet connection of this design, one per patched LoRA layer.

    A LoRA layer computes y = W0 @ x + scale * B(A(x)). This module sits on the rank r tensor
    z = A(x), between A of shape (M, r) and B of shape (r, N):

        z <- z * (1 + gamma) + beta                       global, aspect ratio aware
        z <- z + gate * out_proj(attention(z, reference))  spatial

    Both branches are zero initialized, so a freshly patched adapter reproduces its unpatched
    output exactly and the reference influence grows from zero, the way a controlnet zero
    convolution does. Exactly one tensor per branch is zeroed - the film projection and out_proj.
    Stacking a zero gate on top of a zero out_proj would make the branch unrecoverable: the
    gradient of the gate is the zeroed projection output and the gradient of the projection is
    scaled by the zero gate, so both would stay at zero forever.

    It is the cheap place to do this: everything here is a function of r, not of the M and N of
    the base layer, so a 2048 wide attention layer with rank 16 costs a couple of 16 wide
    projections instead of a full copy of the block.
    """
    def __init__(self, rank: int, config: ReferenceControlConfig):
        super().__init__()
        self.rank = rank
        self.config = config
        cond_dim = config.cond_dim

        self.use_film = config.use_film
        self.use_cross_attention = config.use_cross_attention

        if self.use_film:
            self.film = torch.nn.Linear(cond_dim, 2 * rank)
            torch.nn.init.zeros_(self.film.weight)
            torch.nn.init.zeros_(self.film.bias)

        if self.use_cross_attention:
            self.num_heads = config.num_heads if rank % config.num_heads == 0 else 1
            self.query_norm = torch.nn.LayerNorm(rank)
            self.query_proj = torch.nn.Linear(rank, rank, bias=False)
            self.key_proj = torch.nn.Linear(cond_dim, rank, bias=False)
            self.value_proj = torch.nn.Linear(cond_dim, rank, bias=False)
            self.out_proj = torch.nn.Linear(rank, rank, bias=False)
            torch.nn.init.zeros_(self.out_proj.weight)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        head_dim = self.rank // self.num_heads
        return tensor.view(batch, length, self.num_heads, head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, conditioning: ReferenceConditioning) -> torch.Tensor:
        # hidden_states: (batch, num_tokens, rank)
        dtype = hidden_states.dtype
        tokens = conditioning.tokens.to(dtype)
        global_vector = conditioning.global_vector.to(dtype)

        if self.use_film:
            gamma, beta = self.film(global_vector).chunk(2, dim=-1)
            hidden_states = hidden_states * (1 + gamma[:, None]) + beta[:, None]

        if self.use_cross_attention:
            query = self._split_heads(self.query_proj(self.query_norm(hidden_states)))
            key = self._split_heads(self.key_proj(tokens))
            value = self._split_heads(self.value_proj(tokens))

            mask = None
            if conditioning.padding_mask is not None:
                mask = ~conditioning.padding_mask[:, None, None]

            attention = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
            attention = attention.transpose(1, 2).reshape(hidden_states.shape)
            hidden_states = hidden_states + self.out_proj(attention)

        return hidden_states


class ModulatedLoraA(torch.nn.Module):
    """
    Drop in replacement for the lora_A module of a peft LoRA layer: it runs A, then modulates the
    rank tensor it produced. Patching at this level means peft keeps owning the LoRA weights, the
    optimizer and the checkpointing, and no peft internal is monkey patched.
    """
    def __init__(self,
                 lora_A: torch.nn.Module,
                 modulator: RankModulator,
                 context: ConditioningContext):
        super().__init__()
        self.lora_A = lora_A
        self.modulator = modulator
        self.context = context

    @property
    def weight(self):
        # peft reads lora_A.weight when it merges or inspects the adapter
        return self.lora_A.weight

    @property
    def bias(self):
        return self.lora_A.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lora_A(x)
        conditioning = self.context.current
        if conditioning is None:
            # no reference given for this step, the adapter behaves as a plain LoRA
            return hidden_states

        if hidden_states.dim() == 4:
            # conv2d lora: (batch, rank, h, w) -> (batch, h * w, rank)
            batch, rank, height, width = hidden_states.shape
            flat = hidden_states.flatten(2).transpose(1, 2)
            flat = self.modulator(flat, conditioning)
            return flat.transpose(1, 2).reshape(batch, rank, height, width)

        if hidden_states.dim() == 2:
            # (batch, rank)
            return self.modulator(hidden_states[:, None], conditioning)[:, 0]

        shape = hidden_states.shape
        flat = hidden_states.reshape(shape[0], -1, shape[-1])
        return self.modulator(flat, conditioning).reshape(shape)

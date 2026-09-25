from contextlib import contextmanager
from typing import Optional

import torch

from lightweight_controlnet.config import ReferenceControlConfig
from lightweight_controlnet.conditioner import ReferenceConditioning

# sentinel for "no modulator has run yet", so that a pass that deliberately ran without a
# reference stays distinguishable from one that never happened
_NOTHING = object()

# -1 outside the autograd engine, the id of the running graph task inside it. It is a private
# symbol, so its absence is tolerated: a torch without it falls back to treating an unscoped
# forward as a replay, which keeps the conditioning of the last pass instead of silently dropping
# it, and that is the safer of the two guesses.
_current_graph_task_id = getattr(torch._C, '_current_graph_task_id', None)


def in_backward() -> bool:
    """True while the autograd engine is running.

    That is the only time a module's forward runs without its caller being somewhere up the python
    stack: gradient checkpointing replays the forward of a checkpointed block from inside backward,
    long after the `with control.reference(...)` that wrapped the original one has closed. The
    replay runs on an autograd worker thread, which is why none of this state is thread local.
    """
    if _current_graph_task_id is None:
        return True
    return _current_graph_task_id() != -1


class ConditioningContext:
    """
    Holds the conditioning of the current step. Every modulator of the model points to the same
    instance, so the reference features travel to the adapters without threading an extra argument
    through the whole base model.

    Gradient checkpointing is what makes this more than a variable. A checkpointed block runs its
    forward twice - once for real, once replayed during backward to rebuild the activations it
    threw away - and the replay has to see exactly what the first run saw, or the modulators take
    the plain lora path the second time around and torch reports the two passes saving a different
    number of tensors. An open scope covers that on its own, so a backward pass inside the
    `with control.reference(...)` block needs nothing else. For the usual training loop, which
    closes the block before it calls backward, `for_forward` remembers what the last real forward
    pass observed and serves that to the replay - a reference if the pass had one, and None if it
    ran as a plain lora, so a dropped out step replays as a dropped out step.

    The one shape this cannot reconstruct is two forward passes under different references with
    both backward passes after them, since only one value is remembered. Keep each backward inside
    its own scope and that case resolves itself: an open scope always wins.
    """
    def __init__(self):
        # index 0 is the ambient value set() writes, the rest are the scopes currently open
        self._stack: list[Optional[ReferenceConditioning]] = [None]
        # what the last forward pass that was not a replay ran with
        self._last_seen = _NOTHING

    @property
    def current(self) -> Optional[ReferenceConditioning]:
        """The conditioning of the innermost open scope, or the ambient one."""
        return self._stack[-1]

    def set(self, conditioning: Optional[ReferenceConditioning]):
        """Set the conditioning of the innermost open scope, or the ambient one.

        An ambient conditioning stays in place until it is replaced or cleared, which is what a
        caller driving the model by hand wants. `scope()` is the safer form, and the one the
        trainers use.
        """
        self._stack[-1] = conditioning

    def clear(self):
        self.set(None)

    @contextmanager
    def scope(self, conditioning: Optional[ReferenceConditioning]):
        """Run a block under this conditioning, None included."""
        self._stack.append(conditioning)
        try:
            yield conditioning
        finally:
            self._stack.pop()

    def for_forward(self) -> Optional[ReferenceConditioning]:
        """The conditioning a modulator about to run has to use.

        An open scope, or the ambient value, answers it outright, and the answer is recorded as
        what this pass is running with. Nothing open while backward is running is a checkpointed
        block being replayed, and it is served that record instead, so that it rebuilds the graph
        the forward pass built.
        """
        if len(self._stack) > 1 or not in_backward():
            self._last_seen = self._stack[-1]
            return self._last_seen
        conditioning = self._stack[-1]
        if conditioning is not None:
            return conditioning
        return None if self._last_seen is _NOTHING else self._last_seen


class RankModulator(torch.nn.Module):
    """
    The controlnet connection of this design, one per patched LoRA layer.

    A LoRA layer computes y = W0 @ x + scale * B(A(x)). This module sits on the rank r tensor
    z = A(x), between A of shape (M, r) and B of shape (r, N):

        z <- z * (1 + s * gamma) + s * beta                   global, aspect ratio aware
        z <- z + s * out_proj(attention(z, reference))         spatial

    Both branches are zero initialized, so a freshly patched adapter reproduces its unpatched
    output exactly and the reference influence grows from zero, the way a controlnet zero
    convolution does. Exactly one tensor per branch is zeroed - the film projection and out_proj.
    Stacking a zero gate on top of a zero out_proj would make the branch unrecoverable: the
    gradient of the gate is the zeroed projection output and the gradient of the projection is
    scaled by the zero gate, so both would stay at zero forever.

    `strength` is the s above: an inference time dial on how far the reference moves the rank
    tensor, not a parameter. It is a plain python float rather than a buffer, so it never reaches
    a state dict and never takes a gradient, and it defaults to 1.0 - the value training runs at.
    Setting it to 0.0 skips both branches and reproduces the unconditioned adapter exactly, the
    same output `reference(None)` gives; above 1.0 it over drives the reference. Set it through
    `ReferenceControl.strength`, which writes it to every modulator at once.

    It is the cheap place to do this: everything here is a function of r, not of the M and N of
    the base layer, so a 2048 wide attention layer with rank 16 costs a couple of 16 wide
    projections instead of a full copy of the block.
    """
    def __init__(self, rank: int, config: ReferenceControlConfig):
        super().__init__()
        self.rank = rank
        self.config = config
        cond_dim = config.cond_dim
        # inference time only, see the class docstring. Not a parameter and not a buffer.
        self.strength = 1.0

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
        strength = self.strength
        if strength == 0.0:
            # the reference is dialled out, leave the rank tensor untouched
            return hidden_states

        dtype = hidden_states.dtype
        tokens = conditioning.tokens.to(dtype)
        global_vector = conditioning.global_vector.to(dtype)

        if self.use_film:
            gamma, beta = self.film(global_vector).chunk(2, dim=-1)
            if strength != 1.0:
                gamma, beta = gamma * strength, beta * strength
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
            residual = self.out_proj(attention)
            hidden_states = hidden_states + (residual if strength == 1.0 else residual * strength)

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
        conditioning = self.context.for_forward()
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

"""Where the controlled layers read the reference of the current step from.

The conditioner runs once per forward pass, the hooks run once per layer, and nothing in the base
model would carry a summary vector from the one to the others. This does, without threading an
argument through a model this package did not write.
"""
from contextlib import contextmanager
from typing import Optional

import torch

from lightweight_controlnet.conditioner import ReferenceConditioning

# sentinel for "no layer has run yet", so that a pass that deliberately ran without a reference
# stays distinguishable from one that never happened
_NOTHING = object()

# -1 outside the autograd engine, the id of the running graph task inside it. It is a private
# symbol, so its absence is tolerated: a torch without it falls back to treating an unscoped
# forward as a replay, which keeps the conditioning of the last pass instead of silently dropping
# it, and that is the safer of the two guesses.
_current_graph_task_id = getattr(torch._C, '_current_graph_task_id', None)


def in_backward() -> bool:
    """True while the autograd engine is running.

    That is the only time a layer's hook runs without its caller being somewhere up the python
    stack: gradient checkpointing replays the forward of a checkpointed block from inside backward,
    long after the `with control.reference(...)` that wrapped the original one has closed. The
    replay runs on an autograd worker thread, which is why none of this state is thread local.
    """
    if _current_graph_task_id is None:
        return True
    return _current_graph_task_id() != -1


class ConditioningContext:
    """
    Holds the conditioning of the current step. Every controlled layer of the model points to the
    same instance.

    A None in it means "this step carries no reference image", and that is a real state of the
    model rather than the control path standing down: the reference block of z is filled with
    zeros, the learned block is written as always, and what the layers get is plain FourierFT. It
    is the path a reference dropout step trains and the one an inference run with no reference
    image takes.

    Gradient checkpointing is what makes this more than a variable. A checkpointed block runs its
    forward twice - once for real, once replayed during backward to rebuild the activations it
    threw away - and the replay has to see exactly what the first run saw, or the layers add a
    different z' the second time around and torch reports the two passes saving a different number
    of tensors. An open scope covers that on its own, so a backward pass inside the
    `with control.reference(...)` block needs nothing else. For the usual training loop, which
    closes the block before it calls backward, `for_forward` remembers what the last real forward
    pass observed and serves that to the replay - a reference if the pass had one, and None if it
    had none, so a dropped out step replays as a dropped out step.

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
        """The conditioning a layer about to run has to use.

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

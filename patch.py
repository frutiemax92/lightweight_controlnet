import os
import re
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import torch

from lightweight_controlnet.config import ReferenceControlConfig
from lightweight_controlnet.conditioner import ReferenceConditioner, ReferenceConditioning
from lightweight_controlnet.modulator import ConditioningContext, ModulatedLoraA, RankModulator
from lightweight_controlnet.reference_encoder import ReferenceEncoder, ReferenceFeatures


def _matches(name: str, target_modules) -> bool:
    if target_modules is None:
        return True
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return any(name == target or name.endswith('.' + target) for target in target_modules)


def _rank_of(lora_A: torch.nn.Module) -> int:
    if hasattr(lora_A, 'out_features'):
        return lora_A.out_features
    if hasattr(lora_A, 'out_channels'):
        return lora_A.out_channels
    raise ValueError(f'cannot read the rank of {type(lora_A)}')


class ReferenceControl(torch.nn.Module):
    """
    Owns the reference path of a patched model: the frozen encoder, the shared conditioner, and a
    handle on every rank modulator that was inserted in the LoRA adapters.

    The modulators themselves live inside the base model, so they follow it on .to(), under
    autocast and through gradient checkpointing. This object only groups them for the optimizer
    and for checkpointing.
    """
    def __init__(self,
                 config: ReferenceControlConfig,
                 conditioner: ReferenceConditioner,
                 context: ConditioningContext,
                 modulators: dict[str, RankModulator],
                 encoder: Optional[ReferenceEncoder] = None):
        super().__init__()
        self.config = config
        self.conditioner = conditioner
        self.context = context
        # plain dict on purpose: these modules are already registered inside the base model
        object.__setattr__(self, '_modulators', modulators)
        object.__setattr__(self, '_encoder', encoder)

    @property
    def modulators(self) -> dict[str, RankModulator]:
        return self._modulators

    @property
    def strength(self) -> float:
        """How far the reference is allowed to move the rank tensor, 1.0 being the value the model
        was trained at. It is an inference dial: nothing about it is saved, loaded or learned, so
        leaving it alone keeps a training run exactly as it was.

        0.0 bypasses the modulators entirely, which gives the plain LoRA output `reference(None)`
        gives; between 0 and 1 the reference is blended in; above 1 it is over driven, and far
        above it the model leaves the distribution it was trained on.
        """
        strengths = {modulator.strength for modulator in self._modulators.values()}
        if len(strengths) > 1:
            raise ValueError(f'the modulators hold different strengths ({sorted(strengths)}), '
                             'read them from control.modulators instead')
        return strengths.pop() if strengths else 1.0

    @strength.setter
    def strength(self, value: float):
        value = float(value)
        for modulator in self._modulators.values():
            modulator.strength = value

    @contextmanager
    def strength_scope(self, value: float):
        """Run a block at a given strength and restore whatever was set before:

            with control.strength_scope(0.5), control.reference(images):
                image = pipeline(prompt).images[0]
        """
        previous = {name: modulator.strength for name, modulator in self._modulators.items()}
        self.strength = value
        try:
            yield value
        finally:
            for name, modulator in self._modulators.items():
                modulator.strength = previous[name]

    @property
    def encoder(self) -> Optional[ReferenceEncoder]:
        return self._encoder

    def shared_parameters(self):
        """
        The conditioner only. The modulators are registered inside the base model, so a trainer
        that already collects `[p for p in model.parameters() if p.requires_grad]` picks them up on
        its own; adding shared_parameters() to that list is what completes the reference path.
        """
        return self.conditioner.parameters()

    def trainable_parameters(self):
        seen = set()
        for parameter in self.conditioner.parameters():
            seen.add(id(parameter))
            yield parameter
        for modulator in self._modulators.values():
            for parameter in modulator.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def encode(self, images: Sequence[torch.Tensor]) -> ReferenceFeatures:
        if self._encoder is None:
            raise ValueError('no reference encoder was attached, pass precomputed ReferenceFeatures instead')
        return self._encoder(images)

    def conditioning_for(self,
                         reference: Union[Sequence[torch.Tensor], ReferenceFeatures, None],
                         ) -> Optional[ReferenceConditioning]:
        """Run the shared conditioner over a reference, without installing the result anywhere."""
        if reference is None:
            return None
        features = reference if isinstance(reference, ReferenceFeatures) else self.encode(reference)
        return self.conditioner(features)

    def prepare(self, reference: Union[Sequence[torch.Tensor], ReferenceFeatures]) -> ReferenceConditioning:
        """Install a conditioning that stays in place until it is replaced or cleared.

        `reference()` is the form to prefer, since it cannot be left behind on the way out.
        """
        conditioning = self.conditioning_for(reference)
        self.context.set(conditioning)
        return conditioning

    @contextmanager
    def reference(self, reference: Union[Sequence[torch.Tensor], ReferenceFeatures, None]):
        """
        with control.reference(images):
            noise_prediction = transformer(latents, timesteps, encoder_hidden_states)
            loss.backward()

        Passing None runs the model as a plain LoRA, which is what the reference dropout of
        classifier free guidance training needs.

        Keep the backward pass inside the block when the model is gradient checkpointed. The
        checkpointed blocks replay their forward from inside backward and the modulators read the
        conditioning as they go, so the scope has to still be open - or, failing that, be the last
        one this context closed, which is what makes the usual forward, backward, next step order
        work either way.
        """
        with self.context.scope(self.conditioning_for(reference)) as conditioning:
            yield conditioning

    def control_state_dict(self) -> dict[str, torch.Tensor]:
        state = {f'conditioner.{key}': value for key, value in self.conditioner.state_dict().items()}
        for name, modulator in self._modulators.items():
            for key, value in modulator.state_dict().items():
                state[f'modulators.{name}.{key}'] = value
        return state

    def load_control_state_dict(self, state: dict[str, torch.Tensor]):
        self.conditioner.load_state_dict(
            {key[len('conditioner.'):]: value for key, value in state.items()
             if key.startswith('conditioner.')})
        for name, modulator in self._modulators.items():
            prefix = f'modulators.{name}.'
            modulator.load_state_dict(
                {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.control_state_dict(), os.path.join(save_directory, 'reference_control.pt'))

    def load_pretrained(self, save_directory: str, map_location='cpu'):
        self.load_control_state_dict(
            torch.load(os.path.join(save_directory, 'reference_control.pt'), map_location=map_location))


def apply_reference_control(model: torch.nn.Module,
                            config: ReferenceControlConfig,
                            encoder: Optional[ReferenceEncoder] = None,
                            device=None,
                            dtype=None) -> ReferenceControl:
    """
    Walks the model, finds the LoRA layers of config.adapter_name, and replaces their lora_A by a
    modulated one. The base weights and the LoRA weights are left untouched, so the frozen base
    model stays frozen and the memory profile stays that of a LoRA run.
    """
    context = ConditioningContext()
    conditioner = ReferenceConditioner(config)
    modulators: dict[str, RankModulator] = {}

    resolved_dtype = dtype or getattr(torch, config.dtype)

    for name, module in list(model.named_modules()):
        lora_A = getattr(module, 'lora_A', None)
        if lora_A is None or not hasattr(lora_A, 'keys') or config.adapter_name not in lora_A:
            continue
        if not _matches(name, config.target_modules):
            continue
        inner = lora_A[config.adapter_name]
        if isinstance(inner, ModulatedLoraA):
            continue

        modulator = RankModulator(_rank_of(inner), config).to(dtype=resolved_dtype)
        lora_A[config.adapter_name] = ModulatedLoraA(inner, modulator, context)
        modulators[name.replace('.', '_')] = modulator

    if not modulators:
        raise ValueError('no LoRA layer matched, check target_modules and adapter_name')

    conditioner = conditioner.to(dtype=resolved_dtype)
    if device is None:
        device = next(model.parameters()).device
    conditioner = conditioner.to(device)
    for modulator in modulators.values():
        modulator.to(device)

    return ReferenceControl(config, conditioner, context, modulators, encoder)


def remove_reference_control(model: torch.nn.Module, config: ReferenceControlConfig):
    """Puts the original lora_A modules back, leaving a plain peft model behind."""
    for module in model.modules():
        lora_A = getattr(module, 'lora_A', None)
        if lora_A is None or not hasattr(lora_A, 'keys') or config.adapter_name not in lora_A:
            continue
        inner = lora_A[config.adapter_name]
        if isinstance(inner, ModulatedLoraA):
            lora_A[config.adapter_name] = inner.lora_A

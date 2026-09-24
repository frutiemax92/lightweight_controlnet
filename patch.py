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

    def prepare(self, reference: Union[Sequence[torch.Tensor], ReferenceFeatures]) -> ReferenceConditioning:
        features = reference if isinstance(reference, ReferenceFeatures) else self.encode(reference)
        conditioning = self.conditioner(features)
        self.context.set(conditioning)
        return conditioning

    @contextmanager
    def reference(self, reference: Union[Sequence[torch.Tensor], ReferenceFeatures, None]):
        """
        with control.reference(images):
            noise_prediction = transformer(latents, timesteps, encoder_hidden_states)

        Passing None runs the model as a plain LoRA, which is what the reference dropout of
        classifier free guidance training needs.
        """
        previous = self.context.current
        self.context.set(None if reference is None else self.prepare(reference))
        try:
            yield self.context.current
        finally:
            self.context.set(previous)

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

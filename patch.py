import os
import re
import warnings
from contextlib import contextmanager
from typing import Optional, Sequence, Union

import torch

from lightweight_controlnet.conditioner import ReferenceConditioner, ReferenceConditioning
from lightweight_controlnet.config import FourierControlConfig
from lightweight_controlnet.context import ConditioningContext
from lightweight_controlnet.reference_encoder import ReferenceEncoder, ReferenceFeatures
from lightweight_controlnet.spectrum import FourierSpectrum, add_delta, output_dim_of


def _matches(name: str, target_modules) -> bool:
    if target_modules is None:
        return True
    if isinstance(target_modules, str):
        return re.fullmatch(target_modules, name) is not None
    return any(name == target or name.endswith('.' + target) for target in target_modules)


def _sanitize(name: str) -> str:
    # a ModuleDict key cannot hold a dot, and the name is what the seed and the checkpoint are keyed on
    return name.replace('.', '_')


class FourierControl(torch.nn.Module):
    """
    Owns the whole control path of a model: the frozen reference encoder, the shared conditioner,
    and one FourierSpectrum per controlled layer.

    Nothing of it lives inside the model. The path is attached as a forward hook per layer, so the
    model's module tree, its state dict and the peft adapter it may be training are exactly what
    they were before - which is the difference that makes checkpointing ordinary here: a peft
    adapter saves as a peft adapter, a diffusers model saves as a diffusers model, and this saves
    next to them.

    What that costs is that the trainable weights are not reached by `model.parameters()`. Register
    this object as a submodule of the model - `model.add_module('fourier_control', control)` - and
    they are, which is what an optimizer, accelerate's prepare and the gradient synchronization of
    a multi gpu run all read.
    """
    def __init__(self,
                 config: FourierControlConfig,
                 conditioner: ReferenceConditioner,
                 context: ConditioningContext,
                 layers: torch.nn.ModuleDict,
                 handles: dict,
                 encoder: Optional[ReferenceEncoder] = None):
        super().__init__()
        self.config = config
        self.conditioner = conditioner
        self.layers = layers
        self.context = context
        object.__setattr__(self, '_handles', handles)
        object.__setattr__(self, '_encoder', encoder)

    # --- the inference dials ---------------------------------------------------------------------

    def _scales(self, attribute: str) -> float:
        values = {getattr(layer, attribute) for layer in self.layers.values()}
        if len(values) > 1:
            raise ValueError(f'the layers hold different {attribute} ({sorted(values)}), read them '
                             'from control.layers instead')
        return values.pop() if values else 1.0

    def _set_scales(self, attribute: str, value: float):
        for layer in self.layers.values():
            setattr(layer, attribute, float(value))

    @property
    def strength(self) -> float:
        """The gain the reference bins of z are written at, `config.reference_scaling` - 1.0 - being
        the value the model was trained at. It is an inference dial: a plain python float on every
        layer, so nothing about it is saved, loaded or learned, and leaving it alone keeps a training
        run exactly as it was.

        0.0 zeroes that block, which is not an approximation of "no reference" but exactly it: a
        step with no reference image fills the same entries with the same zeros. Above
        `reference_scaling` it overdrives the reference, and far above it the model leaves the
        distribution it was trained on.
        """
        return self._scales('reference_scaling')

    @strength.setter
    def strength(self, value: float):
        self._set_scales('reference_scaling', value)

    @property
    def spectrum_strength(self) -> float:
        """The gain the learned bins of z are written at, `config.spectrum_scaling` - 1.0.

        The same kind of dial as `strength`, on the other set of bins. 0.0 leaves the base model untouched
        by the learned part, so setting both to 0 is the base model exactly.
        """
        return self._scales('spectrum_scaling')

    @spectrum_strength.setter
    def spectrum_strength(self, value: float):
        self._set_scales('spectrum_scaling', value)

    @contextmanager
    def strength_scope(self, value: float, spectrum: Optional[float] = None):
        """Run a block at a given strength and restore whatever was set before:

            with control.strength_scope(0.5), control.reference(features):
                image = pipeline(prompt).images[0]
        """
        previous = {name: (layer.reference_scaling, layer.spectrum_scaling)
                    for name, layer in self.layers.items()}
        self.strength = value
        if spectrum is not None:
            self.spectrum_strength = spectrum
        try:
            yield value
        finally:
            for name, layer in self.layers.items():
                layer.reference_scaling, layer.spectrum_scaling = previous[name]

    # --- parameters ------------------------------------------------------------------------------

    @property
    def encoder(self) -> Optional[ReferenceEncoder]:
        return self._encoder

    def shared_parameters(self):
        """The conditioner only - the part that is not per layer."""
        return self.conditioner.parameters()

    def trainable_parameters(self):
        return self.parameters()

    def num_trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    # --- running it ------------------------------------------------------------------------------

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
        with control.reference(features):
            noise_prediction = transformer(latents, timesteps, encoder_hidden_states)
            loss.backward()

        Passing None fills the reference block of every spectrum with zeros, which is the plain
        FourierFT path - what the reference dropout of a training run produces, and what an
        inference run with no reference image takes.

        Keep the backward pass inside the block when the model is gradient checkpointed. The
        checkpointed blocks replay their forward from inside backward and the hooks read the
        conditioning as they go, so the scope has to still be open - or, failing that, be the last
        one this context closed, which is what makes the usual forward, backward, next step order
        work either way.
        """
        with self.context.scope(self.conditioning_for(reference)) as conditioning:
            yield conditioning

    # --- checkpointing ---------------------------------------------------------------------------

    def control_state_dict(self) -> dict[str, torch.Tensor]:
        return self.state_dict()

    def load_control_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True):
        return self.load_state_dict(state, strict=strict)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.control_state_dict(), os.path.join(save_directory, 'fourier_control.pt'))

    def load_pretrained(self, save_directory: str, map_location='cpu'):
        path = os.path.join(save_directory, 'fourier_control.pt')
        if not os.path.exists(path) and os.path.exists(os.path.join(save_directory, 'reference_control.pt')):
            raise ValueError(
                f'{save_directory} holds a reference_control.pt, which is a checkpoint of the cross '
                'attention architecture this replaced. Its weights describe modules that no longer '
                'exist; retrain, or check out the commit it was written by.')
        self.load_control_state_dict(torch.load(path, map_location=map_location))

    # --- detaching -------------------------------------------------------------------------------

    def remove_hooks(self):
        """Take the hooks off the model, leaving the weights here and the model as it was found."""
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()


def controllable_modules(model: torch.nn.Module, config: FourierControlConfig):
    """Every (name, module, out_features, channel_last) the config selects.

    A module that holds another controlled one is skipped: a peft LoRA layer wraps the Linear it
    replaced, both answer to the same target name, and hooking both would add the same offset twice.
    The outer one is the one kept, so the offset lands on the layer's whole output, the adapter
    included.
    """
    selected = {}
    for name, module in model.named_modules():
        if not name or not _matches(name, config.target_modules):
            continue
        dims = output_dim_of(module)
        if dims is None or dims[0] < 2:
            continue
        selected[name] = (module, dims[0], dims[1])

    outer = {name: value for name, value in selected.items()
             if not any(name.startswith(other + '.') for other in selected)}
    for name, (module, out_features, channel_last) in outer.items():
        yield name, module, out_features, channel_last


def apply_fourier_control(model: torch.nn.Module,
                          config: FourierControlConfig,
                          encoder: Optional[ReferenceEncoder] = None,
                          device=None,
                          dtype=None) -> FourierControl:
    """
    Walks the model, finds the layers the config names, and hooks each one so that the real part of
    the inverse FFT of its own sparse spectrum is added to its output.

    Nothing of the model is replaced or rewritten - not its modules, not its weights, not its state
    dict. The base weights stay frozen and the memory profile stays that of a forward pass plus one
    small FFT per controlled layer.
    """
    context = ConditioningContext()
    conditioner = ReferenceConditioner(config)
    layers = torch.nn.ModuleDict()
    handles = {}

    resolved_dtype = dtype or getattr(torch, config.dtype)

    def hook_for(spectrum: FourierSpectrum, channel_last: bool, name: str):
        def hook(module, args, output):
            if not torch.is_tensor(output):
                # a layer whose output is a tuple or a dataclass is not something an offset on "the"
                # output dimension describes, so it is left alone rather than guessed at
                warnings.warn(f'{name} returned a {type(output).__name__} rather than a tensor, '
                              'skipping its control path', RuntimeWarning)
                return output
            if output.dim() < 2:
                return output
            return add_delta(output, spectrum(context.for_forward()), channel_last)
        return hook

    for name, module, out_features, channel_last in controllable_modules(model, config):
        key = _sanitize(name)
        spectrum = FourierSpectrum(out_features, config, name=name).to(dtype=resolved_dtype)
        layers[key] = spectrum
        handles[key] = module.register_forward_hook(hook_for(spectrum, channel_last, name))

    if not layers:
        raise ValueError('no layer matched, check target_modules')

    if device is None:
        device = next((parameter.device for parameter in model.parameters()), None)

    conditioner = conditioner.to(dtype=resolved_dtype)
    control = FourierControl(config, conditioner, context, layers, handles, encoder)
    if device is not None:
        control.to(device)
    return control


def remove_fourier_control(control: FourierControl):
    """Takes the hooks off, leaving the model exactly as it was before it was patched."""
    control.remove_hooks()

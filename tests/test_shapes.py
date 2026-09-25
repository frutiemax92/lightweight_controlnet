"""
Shape and behaviour check of the reference controlled LoRA on a dummy model.

    python lightweight_controlnet/tests/test_shapes.py

The dummy model is a small stack of attention blocks that mimics a diffusion transformer, and the
reference features are drawn at random with the shapes dinov2 would produce, so the test runs on
cpu without downloading anything. Set LWCN_TEST_DINOV2=1 to also run the real encoder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from peft import LoraConfig, get_peft_model

from lightweight_controlnet import (
    ReferenceControlConfig,
    ReferenceFeatures,
    apply_reference_control,
    preprocess_reference,
    remove_reference_control,
    resolve_grid,
)

BATCH = 2
SEQUENCE = 32
WIDTH = 64
RANK = 8
REFERENCE_DIM = 768


class DummyBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.to_q = torch.nn.Linear(width, width)
        self.to_k = torch.nn.Linear(width, width)
        self.to_v = torch.nn.Linear(width, width)
        self.to_out = torch.nn.Linear(width, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        attention = torch.nn.functional.scaled_dot_product_attention(
            self.to_q(hidden_states), self.to_k(hidden_states), self.to_v(hidden_states))
        return residual + self.to_out(attention)


class DummyModel(torch.nn.Module):
    def __init__(self, width=WIDTH, depth=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DummyBlock(width) for _ in range(depth)])
        self.proj_out = torch.nn.Linear(width, width)
        # checkpointed one block at a time, the way diffusers checkpoints a unet or a transformer
        self.gradient_checkpointing = False

    def forward(self, hidden_states):
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                hidden_states = torch.utils.checkpoint.checkpoint(block, hidden_states,
                                                                  use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        return self.proj_out(hidden_states)


class DummyConvModel(torch.nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(3, channels, 3, padding=1)
        self.conv_out = torch.nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, pixel_values):
        return self.conv_out(torch.nn.functional.silu(self.conv_in(pixel_values)))


def build_reference_features(batch=BATCH, grid=(8, 12), padding_mask=None):
    patches = grid[0] * grid[1]
    return ReferenceFeatures(
        patch_tokens=torch.randn(batch, patches, REFERENCE_DIM),
        cls_token=torch.randn(batch, REFERENCE_DIM),
        grid=grid,
        aspect_ratio=torch.tensor([grid[1] / grid[0]] * batch),
        padding_mask=padding_mask,
    )


def simulate_trained_lora(model):
    """peft zero initializes lora_B, so an untrained adapter contributes nothing whatever the rank
    tensor looks like. Give it a trained looking B to make the modulation observable."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                torch.nn.init.normal_(parameter, std=0.1)
    return model


def build_patched_model(rank=RANK, target_modules=('to_q', 'to_v'), **config_kwargs):
    torch.manual_seed(0)
    model = DummyModel()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2,
                                             target_modules=list(target_modules)))
    config = ReferenceControlConfig(target_modules=None, cond_dim=64, num_latents=16,
                                    num_heads=4, **config_kwargs)
    control = apply_reference_control(model, config)
    return model, control, config


def test_resolve_grid():
    for height, width in [(1024, 1024), (768, 1344), (1344, 768), (2048, 512), (100, 1000)]:
        patches_h, patches_w = resolve_grid(height, width, patch_size=14, pixel_budget=518)
        source_ratio = width / height
        grid_ratio = patches_w / patches_h
        pixels = patches_h * patches_w * 14 * 14
        assert 0.85 <= (grid_ratio / source_ratio) <= 1.18, (height, width, grid_ratio, source_ratio)
        assert pixels <= 4 * 518 ** 2, (height, width, pixels)
        print(f'  {height}x{width} -> grid {patches_h}x{patches_w}, '
              f'ratio {source_ratio:.3f} vs {grid_ratio:.3f}, {patches_h * patches_w} patches')
    print('test_resolve_grid ok')


def test_preprocess_aspect_ratio():
    config = ReferenceControlConfig()

    uniform = [torch.rand(3, 512, 768), torch.rand(3, 1024, 1536)]
    pixel_values, grid, aspect_ratio, padding_mask = preprocess_reference(uniform, config)
    assert pixel_values.shape[0] == 2 and pixel_values.shape[1] == 3
    assert pixel_values.shape[-2] == grid[0] * config.patch_size
    assert pixel_values.shape[-1] == grid[1] * config.patch_size
    assert padding_mask is None, 'same aspect ratio must not need padding'
    assert torch.allclose(aspect_ratio, torch.tensor([1.5, 1.5]))
    print(f'  same ratio batch -> pixels {tuple(pixel_values.shape)}, grid {grid}, no padding')

    mixed = [torch.rand(3, 512, 1024), torch.rand(3, 1024, 512)]
    pixel_values, grid, aspect_ratio, padding_mask = preprocess_reference(mixed, config)
    assert padding_mask is not None and padding_mask.shape == (2, grid[0] * grid[1])
    assert padding_mask.any() and not padding_mask.all()
    print(f'  mixed ratio batch -> pixels {tuple(pixel_values.shape)}, grid {grid}, '
          f'padded patches {padding_mask.sum(dim=1).tolist()} of {grid[0] * grid[1]}')
    print('test_preprocess_aspect_ratio ok')


def test_shapes_and_identity():
    model, control, _ = build_patched_model()
    simulate_trained_lora(model)
    hidden_states = torch.randn(BATCH, SEQUENCE, WIDTH)

    with torch.no_grad():
        baseline = model(hidden_states)
        with control.reference(build_reference_features()):
            patched = model(hidden_states)

    assert patched.shape == baseline.shape == (BATCH, SEQUENCE, WIDTH), patched.shape
    assert torch.allclose(patched, baseline, atol=1e-6), (patched - baseline).abs().max()
    print(f'  output {tuple(patched.shape)}, zero initialized modulation is an exact identity '
          f'(max delta {(patched - baseline).abs().max():.2e})')

    # once the gates move, the reference must change the output
    with torch.no_grad():
        for modulator in control.modulators.values():
            torch.nn.init.normal_(modulator.out_proj.weight, std=0.1)
            torch.nn.init.normal_(modulator.film.weight, std=0.1)
        with control.reference(build_reference_features()):
            reference_a = model(hidden_states)
        with control.reference(build_reference_features()):
            reference_b = model(hidden_states)

    assert not torch.allclose(reference_a, baseline, atol=1e-5), 'the reference had no effect'
    assert not torch.allclose(reference_a, reference_b, atol=1e-5), 'two references gave one output'
    print(f'  active modulation moves the output by {(reference_a - baseline).abs().max():.4f}, '
          f'two references differ by {(reference_a - reference_b).abs().max():.4f}')
    print('test_shapes_and_identity ok')


def test_strength():
    model, control, _ = build_patched_model()
    simulate_trained_lora(model)
    hidden_states = torch.randn(BATCH, SEQUENCE, WIDTH)
    features = build_reference_features()

    with torch.no_grad():
        for modulator in control.modulators.values():
            torch.nn.init.normal_(modulator.out_proj.weight, std=0.1)
            torch.nn.init.normal_(modulator.film.weight, std=0.1)
        baseline = model(hidden_states)
        with control.reference(features):
            full = model(hidden_states)
            control.strength = 0.0
            off = model(hidden_states)
            control.strength = 0.5
            half = model(hidden_states)
            control.strength = 1.0
            back = model(hidden_states)

    assert control.strength == 1.0
    assert torch.allclose(off, baseline, atol=1e-6), 'strength 0 is not the unconditioned output'
    assert torch.allclose(back, full, atol=1e-6), 'strength did not restore'
    assert not torch.allclose(half, full, atol=1e-5), 'strength 0.5 changed nothing'
    delta_half = (half - baseline).abs().max()
    delta_full = (full - baseline).abs().max()
    assert delta_half < delta_full, (delta_half, delta_full)
    print(f'  strength 0 is the plain lora output, 0.5 moves it by {delta_half:.4f}, '
          f'1.0 by {delta_full:.4f}')

    # the scope restores whatever was set before, exception or not
    control.strength = 0.25
    with control.strength_scope(2.0):
        assert control.strength == 2.0
    assert control.strength == 0.25
    try:
        with control.strength_scope(2.0):
            raise RuntimeError
    except RuntimeError:
        pass
    assert control.strength == 0.25
    control.strength = 1.0

    # nothing about the dial reaches a checkpoint
    control.strength = 0.3
    assert not any('strength' in key for key in control.control_state_dict())
    control.strength = 1.0
    print('  strength stays out of the state dict and strength_scope restores')
    print('test_strength ok')


def test_variable_reference_shapes():
    model, control, _ = build_patched_model()
    hidden_states = torch.randn(BATCH, SEQUENCE, WIDTH)
    for grid in [(37, 37), (26, 52), (52, 26), (8, 60)]:
        with torch.no_grad(), control.reference(build_reference_features(grid=grid)):
            output = model(hidden_states)
        assert output.shape == (BATCH, SEQUENCE, WIDTH)
        print(f'  reference grid {grid[0]}x{grid[1]} ({grid[0] * grid[1]} patches) -> '
              f'output {tuple(output.shape)}')

    padded_grid = (16, 24)
    padding_mask = torch.zeros(BATCH, padded_grid[0] * padded_grid[1], dtype=torch.bool)
    padding_mask[1, padded_grid[0] * padded_grid[1] // 2:] = True
    features = build_reference_features(grid=padded_grid, padding_mask=padding_mask)
    with torch.no_grad(), control.reference(features):
        output = model(hidden_states)
    assert output.shape == (BATCH, SEQUENCE, WIDTH)
    print(f'  padded reference batch -> output {tuple(output.shape)}')
    print('test_variable_reference_shapes ok')


def test_gradients():
    model, control, _ = build_patched_model()
    hidden_states = torch.randn(BATCH, SEQUENCE, WIDTH)

    simulate_trained_lora(model)
    with control.reference(build_reference_features()):
        model(hidden_states).square().mean().backward()

    trainable = list(control.trainable_parameters())
    with_gradient = [parameter for parameter in trainable if parameter.grad is not None
                     and parameter.grad.abs().sum() > 0]
    assert with_gradient, 'no gradient reached the reference path'

    # guard against a dead branch: the two zero initialized tensors must still be able to leave zero
    for name, modulator in control.modulators.items():
        assert modulator.film.weight.grad.abs().sum() > 0, f'{name} film is dead'
        assert modulator.out_proj.weight.grad.abs().sum() > 0, f'{name} cross attention is dead'

    base_model = model.base_model.model
    frozen = [parameter for name, parameter in base_model.named_parameters()
              if 'lora_' not in name and 'modulator' not in name]
    assert all(not parameter.requires_grad for parameter in frozen), 'a base weight is trainable'

    lora_gradients = [parameter.grad for name, parameter in base_model.named_parameters()
                      if 'lora_B' in name and parameter.grad is not None]
    assert lora_gradients, 'no gradient reached lora_B'

    control_parameters = control.num_trainable_parameters()
    lora_parameters = sum(parameter.numel() for name, parameter in base_model.named_parameters()
                          if '.lora_' in name and 'modulator' not in name)
    base_parameters = sum(parameter.numel() for parameter in frozen)
    print(f'  frozen base {base_parameters}, lora {lora_parameters}, reference path '
          f'{control_parameters} ({len(control.modulators)} modulators + conditioner)')
    print(f'  {len(with_gradient)}/{len(trainable)} reference tensors got a gradient '
          f'(the zero projections keep the rest at zero on the first step)')
    print('test_gradients ok')


def _gradients_of(model, control, hidden_states, features, checkpointing, backward_inside_scope):
    """The reference path gradients of one step, run with or without gradient checkpointing."""
    base_model = model.base_model.model
    base_model.gradient_checkpointing = checkpointing
    for parameter in control.trainable_parameters():
        parameter.grad = None

    if backward_inside_scope:
        with control.reference(features):
            model(hidden_states).square().mean().backward()
    else:
        # what a training loop does: the block sets the conditioning up, and the backward pass
        # happens after it closed, which is when the checkpointed blocks replay their forward
        with control.reference(features):
            loss = model(hidden_states).square().mean()
        loss.backward()

    base_model.gradient_checkpointing = False
    # a pass that ran as a plain lora never reaches the modulators, so its gradient stays None
    return {name: (None if modulator.out_proj.weight.grad is None
                   else modulator.out_proj.weight.grad.clone())
            for name, modulator in control.modulators.items()}


def test_gradient_checkpointing():
    model, control, _ = build_patched_model()
    simulate_trained_lora(model)
    with torch.no_grad():
        for modulator in control.modulators.values():
            torch.nn.init.normal_(modulator.out_proj.weight, std=0.1)
            torch.nn.init.normal_(modulator.film.weight, std=0.1)

    torch.manual_seed(1)
    hidden_states = torch.randn(BATCH, SEQUENCE, WIDTH)
    features = build_reference_features()

    plain = _gradients_of(model, control, hidden_states, features, False, False)
    for backward_inside_scope in [True, False]:
        # the checkpointed run has to replay the modulated forward, not the plain lora one, whether
        # or not the reference block is still open when backward runs
        checkpointed = _gradients_of(model, control, hidden_states, features, True,
                                     backward_inside_scope)
        for name, gradient in plain.items():
            assert torch.allclose(gradient, checkpointed[name], atol=1e-5), \
                f'{name} differs with checkpointing, backward inside the scope: {backward_inside_scope}'
        where = 'inside' if backward_inside_scope else 'outside'
        print(f'  backward {where} the reference block: gradients match the uncheckpointed run '
              f'(max delta {max((plain[name] - checkpointed[name]).abs().max() for name in plain):.2e})')

    # the reference dropout of a training step must replay as a plain lora, not resurrect the
    # conditioning of the step before it
    dropped = _gradients_of(model, control, hidden_states, None, True, False)
    assert all(gradient is None or gradient.abs().sum() == 0 for gradient in dropped.values()), \
        'a dropped out step got a gradient through the reference path'
    print('  a step whose reference was dropped replays as a plain lora')

    # so must a forward that never opened a block at all, after one that did
    base_model = model.base_model.model
    for parameter in control.trainable_parameters():
        parameter.grad = None
    base_model.gradient_checkpointing = True
    model(hidden_states).square().mean().backward()
    base_model.gradient_checkpointing = False
    assert all(modulator.out_proj.weight.grad is None
               or modulator.out_proj.weight.grad.abs().sum() == 0
               for modulator in control.modulators.values()), \
        'an unreferenced forward pass picked up a stale conditioning'
    print('  an unreferenced forward pass after a referenced one stays a plain lora')
    print('test_gradient_checkpointing ok')


def test_conv_lora():
    torch.manual_seed(0)
    model = DummyConvModel()
    model = simulate_trained_lora(
        get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['conv_in'])))
    config = ReferenceControlConfig(cond_dim=64, num_latents=8, num_heads=4)
    control = apply_reference_control(model, config)

    pixel_values = torch.randn(BATCH, 3, 32, 32)
    with torch.no_grad():
        baseline = model(pixel_values)
        with control.reference(build_reference_features(grid=(8, 12))):
            patched = model(pixel_values)
    assert patched.shape == baseline.shape == (BATCH, 3, 32, 32)
    assert torch.allclose(patched, baseline, atol=1e-6)
    print(f'  conv2d lora output {tuple(patched.shape)}, identity at init')
    print('test_conv_lora ok')


def test_save_load_and_removal():
    model, control, config = build_patched_model()
    with torch.no_grad():
        for modulator in control.modulators.values():
            torch.nn.init.normal_(modulator.out_proj.weight, std=0.1)

    state = control.control_state_dict()
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.test_checkpoint')
    control.save_pretrained(directory)

    fresh_model, fresh_control, _ = build_patched_model()
    fresh_control.load_pretrained(directory)
    for key, value in state.items():
        assert torch.allclose(fresh_control.control_state_dict()[key], value), key
    print(f'  {len(state)} tensors saved and reloaded')

    hidden_states = torch.randn(BATCH, SEQUENCE, WIDTH)
    with torch.no_grad():
        with control.reference(build_reference_features()):
            before = model(hidden_states)
        remove_reference_control(model, config)
        plain = model(hidden_states)
    assert before.shape == plain.shape
    print('  the patch can be removed, the model is a plain peft model again')

    import shutil
    shutil.rmtree(directory, ignore_errors=True)
    print('test_save_load_and_removal ok')


def test_real_dinov2():
    if os.environ.get('LWCN_TEST_DINOV2') != '1':
        print('test_real_dinov2 skipped, set LWCN_TEST_DINOV2=1 to run it')
        return
    from lightweight_controlnet import ReferenceEncoder

    config = ReferenceControlConfig(cond_dim=64, num_latents=16, num_heads=4, pixel_budget=224)
    encoder = ReferenceEncoder(config)
    model, control, _ = build_patched_model()
    object.__setattr__(control, '_encoder', encoder)

    images = [torch.rand(3, 512, 768), torch.rand(3, 768, 512)]
    hidden_states = torch.randn(2, SEQUENCE, WIDTH)
    with torch.no_grad(), control.reference(images) as conditioning:
        output = model(hidden_states)
    print(f'  dinov2 tokens {tuple(conditioning.tokens.shape)}, output {tuple(output.shape)}')
    assert output.shape == (2, SEQUENCE, WIDTH)
    print('test_real_dinov2 ok')


if __name__ == '__main__':
    for test in [test_resolve_grid,
                 test_preprocess_aspect_ratio,
                 test_shapes_and_identity,
                 test_strength,
                 test_variable_reference_shapes,
                 test_gradients,
                 test_gradient_checkpointing,
                 test_conv_lora,
                 test_save_load_and_removal,
                 test_real_dinov2]:
        print(f'--- {test.__name__}')
        test()
    print('\nall checks passed')

"""
Shape and behaviour check of the fourierft control path on a dummy model.

    python lightweight_controlnet/tests/test_shapes.py

The dummy model is a small stack of attention blocks that mimics a diffusion transformer, and the
reference features are drawn at random with the shapes dinov2 would produce, so the test runs on cpu
without downloading anything. Set LWCN_TEST_DINOV2=1 to also run the real encoder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from peft import LoraConfig, get_peft_model

from lightweight_controlnet import (
    FourierControlConfig,
    FourierSpectrum,
    ReferenceConditioning,
    ReferenceFeatures,
    apply_fourier_control,
    choose_slots,
    output_dim_of,
    preprocess_reference,
    remove_fourier_control,
    resolve_grid,
)

BATCH = 2
SEQUENCE = 32
WIDTH = 64
RANK = 8
REFERENCE_DIM = 768
COND_DIM = 32


class DummyBlock(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.to_q = torch.nn.Linear(width, width)
        self.to_k = torch.nn.Linear(width, width)
        self.to_v = torch.nn.Linear(width, width)
        self.to_out = torch.nn.Linear(width, width)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        attention = torch.nn.functional.scaled_dot_product_attention(
            self.to_q(hidden_states), self.to_k(hidden_states), self.to_v(hidden_states))
        return residual + self.to_out(attention)


class DummyModel(torch.nn.Module):
    def __init__(self, width=WIDTH, depth=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DummyBlock(width) for _ in range(depth)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class DummyUnet(torch.nn.Module):
    """A convolution and a linear on the same model, so both broadcast axes are exercised."""
    def __init__(self, channels=16, width=WIDTH):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels * 2, 1)
        self.to_out = torch.nn.Linear(width, width)

    def forward(self, image, sequence):
        return self.conv2(torch.relu(self.conv1(image))), self.to_out(sequence)


def config(**overrides) -> FourierControlConfig:
    kwargs = dict(target_modules=['to_q', 'to_k', 'to_v', 'to_out'],
                  reference_dim=REFERENCE_DIM, cond_dim=COND_DIM)
    kwargs.update(overrides)
    return FourierControlConfig(**kwargs)


def features(batch=BATCH, grid=(7, 5), dim=REFERENCE_DIM, seed=None, **kwargs):
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    def randn(*shape):
        return torch.randn(*shape, generator=generator)
    return ReferenceFeatures(patch_tokens=randn(batch, grid[0] * grid[1], dim),
                             cls_token=randn(batch, dim),
                             grid=grid,
                             aspect_ratio=torch.full((batch,), grid[1] / grid[0]),
                             **kwargs)


def lora_model(width=WIDTH, depth=2, rank=RANK):
    model = DummyModel(width, depth)
    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank * 2,
                                            target_modules=['to_q', 'to_k', 'to_v', 'to_out']))


def excited(control, spectrum_std=0.05, reference_std=0.05, seed=0):
    """Move both zero initialized weights off zero, so the path is visible at all."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in control.layers.values():
            layer.spectrum.copy_(torch.randn(layer.spectrum.shape, generator=generator) * spectrum_std)
            layer.reference_proj.weight.copy_(
                torch.randn(layer.reference_proj.weight.shape, generator=generator) * reference_std)
    return control


# --- the reference encoder's aspect ratio handling, which the new architecture kept ---------------

def test_resolve_grid():
    square = resolve_grid(1024, 1024)
    assert square[0] == square[1], square
    wide = resolve_grid(512, 1024)
    assert wide[1] > wide[0], wide
    # the pixel budget is spent, not the shape: both grids hold about as many patches
    assert abs(square[0] * square[1] - wide[0] * wide[1]) / (square[0] * square[1]) < 0.15
    # a sliver still keeps some detail on its short side
    assert resolve_grid(64, 4096)[0] >= 8
    print('resolve_grid: ok')


def test_preprocess_aspect_ratio():
    settings = config()
    same = [torch.rand(3, 512, 1024), torch.rand(3, 256, 512)]
    pixel_values, grid, aspect_ratio, padding_mask = preprocess_reference(same, settings)
    assert padding_mask is None, 'two images of the same ratio need no padding'
    assert pixel_values.shape[-2] == grid[0] * settings.patch_size
    assert torch.allclose(aspect_ratio, torch.tensor([2.0, 2.0]))

    mixed = [torch.rand(3, 512, 1024), torch.rand(3, 1024, 512)]
    pixel_values, grid, aspect_ratio, padding_mask = preprocess_reference(mixed, settings)
    assert padding_mask is not None and padding_mask.shape == (2, grid[0] * grid[1])
    assert padding_mask.any(), 'a mixed batch has to mark some patches as padding'
    print('preprocess_reference aspect ratio: ok')


# --- where the two blocks land in z --------------------------------------------------------------

def test_slots_are_disjoint_and_in_range():
    for bins, learned, reference in ((64, 32, 16), (7, 3, 2), (2, 1, 1), (512, 256, 128)):
        a, b = choose_slots(bins, learned, reference, seed=1)
        assert a.numel() == learned and b.numel() == reference, bins
        assert a.min() >= 0 and a.max() < bins and b.min() >= 0 and b.max() < bins
        assert not set(a.tolist()) & set(b.tolist()), 'the two sets of bins overlap'
    print('bins are disjoint and in range: ok')


def test_bins_are_scattered_not_contiguous():
    """Both sets have to be spread over the spectrum, not sitting in one band of frequencies."""
    a, b = choose_slots(256, 64, 32, seed=3)
    for name, block in (('learned', a), ('reference', b)):
        gaps = block.diff()
        assert (gaps > 1).any(), f'the {name} bins came out contiguous'
        assert not (gaps == 1).all(), f'the {name} bins came out contiguous'
    # and they cover the whole range rather than clustering in one end of it
    assert a.min() < 256 * 0.2 and a.max() > 256 * 0.8, 'the learned bins do not span the spectrum'
    print('bins are scattered over the whole spectrum: ok')


def test_layers_place_their_bins_differently():
    settings = config()
    placements = {name: tuple(FourierSpectrum(WIDTH, settings, name=name).spectrum_indices.tolist())
                  for name in ('blocks.0.to_q', 'blocks.0.to_k', 'blocks.1.to_q')}
    assert len(set(placements.values())) == len(placements), \
        'every layer drew the same placement, so the seed is not per layer'
    # and the placement is reproducible, since a checkpoint written before this ran has to match
    again = FourierSpectrum(WIDTH, settings, name='blocks.0.to_q')
    assert tuple(again.spectrum_indices.tolist()) == placements['blocks.0.to_q']
    print('per layer placement, reproducible: ok')


def test_narrow_layers_are_clamped():
    settings = config(n_frequency=10_000, n_reference=10_000)
    for width in (2, 3, 8, 64):
        bins = settings.bins_for(width)
        learned, reference = settings.slots_for(width)
        assert learned >= 1 and reference >= 1, width
        assert learned + reference <= bins, (width, bins)
    print('narrow layers clamp their bin counts: ok')


def test_one_step_stays_small_next_to_the_layer():
    """The gain has to leave a single optimizer step small next to the layer's own output.

    This is the regression test for shipping FourierFT's gain of 150 without redoing the
    normalization argument for a 1d ortho transform. With norm='ortho' the 1/sqrt(N) the transform
    applies is cancelled by the ~N/2 bins that contribute, so std(z') already lands at roughly
    std(parameter) and there is nothing for a gain to compensate. At 150, one AdamW step at lr 5e-4
    moved a 1280 wide layer by 69% of its own output, and over the 454 layers an SDXL run hooks the
    first validation image came out as pure noise.
    """
    width, lr = 512, 5e-4

    class Wide(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = torch.nn.Linear(width, width)

        def forward(self, hidden_states):
            return self.to_q(hidden_states)

    torch.manual_seed(0)
    model = Wide()
    control = apply_fourier_control(model, config(target_modules=['to_q']))
    layer = control.layers['to_q']
    hidden = torch.randn(2, 16, width)
    output_scale = model.to_q(hidden).std().item()

    optimizer = torch.optim.AdamW(control.parameters(), lr=lr)
    with control.reference(features()):
        (model(hidden) - torch.randn(2, 16, width)).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(control.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        offset = layer(control.conditioning_for(features())).std().item()
    relative = offset / output_scale
    assert relative < 0.05, (
        f'one step at lr {lr} moved the layer by {relative:.1%} of its own output. A few hundred '
        f'hooked layers of that is noise - check spectrum_scaling / reference_scaling')
    print(f'one step moves the layer by {relative:.2%} of its output: ok')


def test_the_transform_is_full_rank():
    """The reason for irfft over Real(ifft(.)): every offset has to be reachable.

    Real(ifft(z)) of a real z is exactly even-symmetric, so its rank is N // 2 + 1 and entries past
    that buy nothing. irfft over a complex half-spectrum has 2 * bins - 2 = N free reals.
    """
    for N in (38, 64, 321):
        bins = N // 2 + 1
        columns = []
        for bin_index in range(bins):
            for part in (0, 1):
                z = torch.zeros(bins, 2)
                z[bin_index, part] = 1.0
                columns.append(torch.fft.irfft(torch.view_as_complex(z), n=N))
        rank = torch.linalg.matrix_rank(torch.stack(columns, dim=1), atol=1e-6).item()
        assert rank == N, f'N={N}: rank {rank}, expected full rank {N}'
    print('the transform reaches full rank: ok')


def test_the_offset_is_not_symmetric():
    """The symmetry the old parameterization could not avoid has to be gone."""
    layer = FourierSpectrum(256, config(), name='blocks.0.to_q')
    with torch.no_grad():
        layer.spectrum.normal_(std=0.01)
        layer.reference_proj.weight.normal_(std=0.01)
        offset = layer(ReferenceConditioning(vector=torch.randn(1, COND_DIM)))[0]
    mirror = torch.cat([offset[:1], offset[1:].flip(0)])
    assert (offset - mirror).abs().max() > 1e-3, \
        "z' is still even-symmetric, so the half-spectrum is not being used"
    print('the offset is not mirror-symmetric: ok')


def test_output_dim_of():
    assert output_dim_of(torch.nn.Linear(7, 13)) == (13, True)
    assert output_dim_of(torch.nn.Conv2d(7, 13, 3)) == (13, False)
    assert output_dim_of(torch.nn.LayerNorm(13)) is None
    # a peft lora layer reports the width of the Linear it replaced, and on the channel-last axis
    model = lora_model()
    lora_linear = model.base_model.model.blocks[0].to_q
    assert hasattr(lora_linear, 'get_base_layer'), 'the test model is not actually a peft model'
    assert output_dim_of(lora_linear) == (WIDTH, True)
    print('output_dim_of: ok')


# --- what z' is ----------------------------------------------------------------------------------

def test_the_transform_is_what_it_says():
    """z' has to be the irfft of the two sets of bins written into a zero half-spectrum."""
    settings = config(cond_dim=COND_DIM)
    layer = FourierSpectrum(WIDTH, settings, name='blocks.0.to_q')
    with torch.no_grad():
        layer.spectrum.normal_(std=0.01)
        layer.reference_proj.weight.normal_(std=0.01)

    vector = torch.randn(1, COND_DIM)
    got = layer(ReferenceConditioning(vector=vector))

    z = torch.zeros(1, layer.bins, 2)
    z[0, layer.spectrum_indices] = layer.spectrum.detach() * layer.spectrum_scaling
    z[0, layer.reference_indices] = (
        layer.reference_proj(vector).detach().unflatten(-1, (layer.n_reference, 2))
        * layer.reference_gain * layer.reference_scaling)
    expected = torch.fft.irfft(torch.view_as_complex(z), n=WIDTH, norm=settings.ifft_norm)
    assert torch.allclose(got, expected, atol=1e-6), (got[0, :4], expected[0, :4])
    assert got.shape == (1, WIDTH), got.shape
    print("z' is irfft(z, n=out_features): ok")


def test_the_offset_is_per_channel_and_uniform():
    model = DummyUnet()
    # only the last op of each branch, so what the offset does is visible in the output rather than
    # passed through a relu and a second convolution first
    control = excited(apply_fourier_control(model, config(target_modules=['conv2', 'to_out'])))
    image, sequence = torch.randn(BATCH, 16, 9, 7), torch.randn(BATCH, SEQUENCE, WIDTH)

    base = DummyUnet()
    base.load_state_dict(model.state_dict())
    with control.reference(features()):
        got_image, got_sequence = model(image, sequence)
    want_image, want_sequence = base(image, sequence)

    # a conv offset is constant over h and w, one number per output channel
    offset = (got_image - want_image)
    assert torch.allclose(offset, offset[..., :1, :1].expand_as(offset), atol=1e-5), \
        'the conv offset is not uniform over the feature map'
    # and a linear offset is constant over the tokens, one number per output feature
    offset = (got_sequence - want_sequence)
    assert torch.allclose(offset, offset[:, :1].expand_as(offset), atol=1e-5), \
        'the linear offset is not uniform over the sequence'
    print('the offset is one number per output channel, uniform over the frame: ok')


def test_fresh_patch_is_a_no_op():
    torch.manual_seed(0)
    model = DummyModel()
    base = DummyModel()
    base.load_state_dict(model.state_dict())
    control = apply_fourier_control(model, config())
    hidden = torch.randn(BATCH, SEQUENCE, WIDTH)
    with control.reference(features()):
        patched = model(hidden)
    assert torch.equal(patched, base(hidden)), 'a zero initialized control path is not a no-op'
    print('a fresh patch is bit identical to the base model: ok')


def test_no_reference_zeroes_the_reference_bins():
    """Without a reference, z' has to be the transform of the learned bins alone.

    Checked against a z built by hand rather than by transforming z' back, since rfft of z' would
    also have to be told which bins to look at.
    """
    model = DummyModel()
    control = excited(apply_fourier_control(model, config()))
    layer = next(iter(control.layers.values()))

    z = torch.zeros(1, layer.bins, 2)
    z[0, layer.spectrum_indices] = layer.spectrum.detach() * layer.spectrum_scaling
    expected = torch.fft.irfft(torch.view_as_complex(z), n=layer.out_features, norm=layer.ifft_norm)
    with torch.no_grad():
        got = layer(None)
    assert torch.allclose(got, expected, atol=1e-6), \
        'without a reference z is not the learned bins written into zeros'
    print('no reference leaves the reference bins at zero: ok')


def test_per_sample_conditioning():
    model = DummyModel()
    control = excited(apply_fourier_control(model, config()))
    hidden = torch.randn(BATCH, SEQUENCE, WIDTH)
    first = features(seed=1)
    second = ReferenceFeatures(patch_tokens=torch.cat([first.patch_tokens[:1],
                                                       torch.randn(1, *first.patch_tokens.shape[1:])]),
                              cls_token=torch.cat([first.cls_token[:1], torch.randn(1, REFERENCE_DIM)]),
                              grid=first.grid, aspect_ratio=first.aspect_ratio)
    with control.reference(first):
        a = model(hidden)
    with control.reference(second):
        b = model(hidden)
    assert torch.allclose(a[0], b[0], atol=1e-6), 'sample 0 shares its reference and should not move'
    assert not torch.allclose(a[1], b[1]), 'sample 1 has its own reference and should move'
    print('every sample of the batch gets its own z: ok')


# --- the dials -----------------------------------------------------------------------------------

def test_strength_dials():
    model = DummyModel()
    base = DummyModel()
    base.load_state_dict(model.state_dict())
    control = excited(apply_fourier_control(model, config()))
    hidden = torch.randn(BATCH, SEQUENCE, WIDTH)
    reference = features()

    with control.reference(None):
        without = model(hidden)
    with control.strength_scope(0.0), control.reference(reference):
        zeroed = model(hidden)
    assert torch.allclose(without, zeroed, atol=1e-6), \
        'strength 0 is not exactly the no-reference path'

    with control.strength_scope(0.0, spectrum=0.0), control.reference(reference):
        nothing = model(hidden)
    assert torch.allclose(nothing, base(hidden), atol=1e-6), \
        'both dials at 0 is not the base model'

    # and the scope restores what was set before it
    assert control.strength == config().reference_scaling
    assert control.spectrum_strength == config().spectrum_scaling

    # louder means above the trained gain, which is reference_scaling and not 1
    with control.strength_scope(2.0 * config().reference_scaling), control.reference(reference):
        louder = model(hidden)
    with control.reference(reference):
        trained = model(hidden)
    assert (louder - without).abs().mean() > (trained - without).abs().mean(), \
        'raising the strength did not make the reference louder'
    print('the strength dials: ok')


def test_scaling_defaults_come_from_the_config():
    control = apply_fourier_control(DummyModel(), config(spectrum_scaling=3.0, reference_scaling=7.0))
    assert control.spectrum_strength == 3.0 and control.strength == 7.0
    print('the dials come up at the configured scalings: ok')


# --- batching ------------------------------------------------------------------------------------

def test_cfg_batch_broadcast():
    model = DummyModel()
    control = excited(apply_fourier_control(model, config()))
    # one reference, a batch of four: the classifier free guidance shape
    with control.reference(features(batch=1)):
        model(torch.randn(4, SEQUENCE, WIDTH))
    # two references, a batch of four: the negative and positive halves concatenated
    with control.reference(features(batch=2)):
        out = model(torch.randn(4, SEQUENCE, WIDTH))
    assert out.shape == (4, SEQUENCE, WIDTH)
    # and a batch the reference does not divide is an error rather than a silent tile
    try:
        with control.reference(features(batch=3)):
            model(torch.randn(4, SEQUENCE, WIDTH))
    except ValueError:
        pass
    else:
        raise AssertionError('a reference batch of 3 under a model batch of 4 should raise')
    print('classifier free guidance batch broadcast: ok')


def test_variable_reference_shapes():
    model = DummyModel()
    control = excited(apply_fourier_control(model, config()))
    hidden = torch.randn(BATCH, SEQUENCE, WIDTH)
    for grid in ((7, 5), (5, 7), (16, 16), (8, 8)):
        with control.reference(features(grid=grid)):
            assert model(hidden).shape == hidden.shape, grid
    # a padded batch: the masked patches must not reach the pooled summary
    mask = torch.zeros(BATCH, 7 * 5, dtype=torch.bool)
    mask[1, 20:] = True
    padded = features(grid=(7, 5), padding_mask=mask, seed=5)
    with control.reference(padded):
        assert model(hidden).shape == hidden.shape
    unpadded = ReferenceFeatures(patch_tokens=padded.patch_tokens, cls_token=padded.cls_token,
                                 grid=padded.grid, aspect_ratio=padded.aspect_ratio)
    with torch.no_grad():
        with_mask = control.conditioning_for(padded).vector
        without_mask = control.conditioning_for(unpadded).vector
    assert not torch.allclose(with_mask[1], without_mask[1]), \
        'the padding mask changed nothing, so the padded patches were read'
    assert torch.allclose(with_mask[0], without_mask[0], atol=1e-5), \
        'the unpadded sample of the batch should be unaffected by the other one\'s mask'
    print('variable reference shapes and the padding mask: ok')


# --- gradients -----------------------------------------------------------------------------------

def test_gradients_reach_everything():
    model = lora_model()
    control = apply_fourier_control(model, config())
    with control.reference(features()):
        model(torch.randn(BATCH, SEQUENCE, WIDTH)).square().mean().backward()

    missing = [name for name, parameter in control.named_parameters()
               if parameter.grad is None or not parameter.grad.any()]
    # the conditioner is the documented exception on the very first step: it reaches the loss only
    # through the zero initialized reference projection
    unexpected = [name for name in missing if not name.startswith('conditioner.')]
    assert not unexpected, f'no gradient on {unexpected}'
    assert any(name.startswith('layers.') for name, parameter in control.named_parameters()
               if parameter.grad is not None and parameter.grad.any()), \
        'no layer of the control path took a gradient'

    # and from the second step on the conditioner does get one, since the projection has moved
    with torch.no_grad():
        for layer in control.layers.values():
            layer.reference_proj.weight.normal_(std=1e-3)
    control.zero_grad(set_to_none=True)
    with control.reference(features()):
        model(torch.randn(BATCH, SEQUENCE, WIDTH)).square().mean().backward()
    missing = [name for name, parameter in control.named_parameters()
               if parameter.grad is None or not parameter.grad.any()]
    assert not missing, f'no gradient on {missing} once the projection is off zero'
    print('gradients reach every parameter: ok')


def test_gradient_checkpointing():
    """The usual loop shape: the backward runs after the reference scope has closed."""
    from torch.utils.checkpoint import checkpoint

    class Checkpointed(DummyModel):
        def forward(self, hidden_states):
            for block in self.blocks:
                hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            return hidden_states

    model = Checkpointed()
    control = excited(apply_fourier_control(model, config()))
    hidden = torch.randn(BATCH, SEQUENCE, WIDTH)

    with control.reference(features(seed=2)):
        out = model(hidden)
    out.square().mean().backward()
    assert all(parameter.grad is not None for parameter in control.layers.parameters())

    # a step whose reference was dropped has to replay as a dropped step, or the two passes disagree
    control.zero_grad(set_to_none=True)
    with control.reference(None):
        out = model(hidden)
    out.square().mean().backward()
    for layer in control.layers.values():
        grad = layer.reference_proj.weight.grad
        assert grad is None or not grad.any(), \
            'the replay of a dropped step read a reference the forward pass did not'

    # and the same model gives the same answer whether or not it is checkpointed
    plain = DummyModel()
    plain.load_state_dict({key: value for key, value in model.state_dict().items()})
    plain_control = apply_fourier_control(plain, config())
    plain_control.load_control_state_dict(control.control_state_dict())
    with control.reference(features(seed=2)):
        checkpointed_out = model(hidden)
    with plain_control.reference(features(seed=2)):
        plain_out = plain(hidden)
    assert torch.allclose(checkpointed_out, plain_out, atol=1e-5)
    print('gradient checkpointing: ok')


# --- living next to peft -------------------------------------------------------------------------

def test_the_hook_lands_on_the_lora_layer():
    model = lora_model()
    control = apply_fourier_control(model, config())
    # one hook per target, on the lora layer and not also on the Linear inside it
    assert len(control.layers) == 2 * 4, control.layers.keys()
    for name in control.layers:
        assert 'base_layer' not in name and 'lora_' not in name, name
    print('one hook per target, on the outer lora layer: ok')


def test_the_peft_checkpoint_stays_clean():
    from peft import get_peft_model_state_dict
    model = lora_model()
    control = apply_fourier_control(model, config())
    model.add_module('fourier_control', control)
    adapter = get_peft_model_state_dict(model)
    assert adapter, 'the adapter state dict came out empty'
    assert not any('fourier_control' in key or 'spectrum' in key for key in adapter), \
        'the control path leaked into the peft adapter'
    print('the peft adapter stays clean with the control path attached: ok')


def test_the_model_state_dict_is_untouched():
    model = DummyModel()
    before = set(model.state_dict())
    control = apply_fourier_control(model, config())
    assert set(model.state_dict()) == before, \
        'attaching the control path changed the model state dict'
    # it only appears once it is deliberately registered, which is what the trainers do for DDP
    model.add_module('fourier_control', control)
    assert any(key.startswith('fourier_control.') for key in model.state_dict())
    print('hooking does not change the model state dict: ok')


# --- checkpointing and removal -------------------------------------------------------------------

def test_save_load_and_removal(tmp='out/test_fourier_control'):
    torch.manual_seed(0)
    model = DummyModel()
    pristine = DummyModel()
    pristine.load_state_dict(model.state_dict())
    control = excited(apply_fourier_control(model, config()))
    hidden = torch.randn(BATCH, SEQUENCE, WIDTH)
    reference = features(seed=7)
    with control.reference(reference):
        expected = model(hidden)

    control.save_pretrained(tmp)
    assert os.path.exists(os.path.join(tmp, 'fourier_control.pt'))
    assert os.path.exists(os.path.join(tmp, 'fourier_control_config.json'))

    reloaded_config = FourierControlConfig.from_pretrained(tmp)
    assert reloaded_config.cond_dim == COND_DIM

    fresh_model = DummyModel()
    fresh_model.load_state_dict(pristine.state_dict())
    fresh = apply_fourier_control(fresh_model, reloaded_config)
    fresh.load_pretrained(tmp)
    with fresh.reference(reference):
        got = fresh_model(hidden)
    assert torch.allclose(got, expected, atol=1e-6), 'a reloaded control path does not reproduce itself'

    # the bin placement travels with the weights, not with the seed
    for name, layer in control.layers.items():
        assert torch.equal(layer.spectrum_indices, fresh.layers[name].spectrum_indices)

    remove_fourier_control(control)
    assert torch.equal(model(hidden), pristine(hidden)), \
        'removing the control path did not restore the model'
    print('save, load and removal: ok')


def test_a_legacy_checkpoint_says_so(tmp='out/test_legacy_control'):
    os.makedirs(tmp, exist_ok=True)
    open(os.path.join(tmp, 'reference_control_config.json'), 'w').write('{}')
    torch.save({}, os.path.join(tmp, 'reference_control.pt'))
    try:
        FourierControlConfig.from_pretrained(tmp)
    except ValueError as error:
        assert 'cross attention' in str(error), error
    else:
        raise AssertionError('a checkpoint of the old architecture should be refused by name')
    print('a checkpoint of the replaced architecture is refused by name: ok')


# --- the real encoder, opt in --------------------------------------------------------------------

def test_real_dinov2():
    if os.environ.get('LWCN_TEST_DINOV2') != '1':
        print('real dinov2: skipped (set LWCN_TEST_DINOV2=1)')
        return
    from lightweight_controlnet import ReferenceEncoder
    settings = config(reference_model=os.environ.get('LWCN_TEST_DINOV2_MODEL', 'facebook/dinov2-base'),
                      reference_dim=int(os.environ.get('LWCN_TEST_DINOV2_DIM', 768)))
    encoder = ReferenceEncoder(settings)
    extracted = encoder([torch.rand(3, 480, 640), torch.rand(3, 480, 640)])
    assert extracted.patch_tokens.shape[0] == 2
    assert extracted.patch_tokens.shape[1] == extracted.grid[0] * extracted.grid[1]
    assert extracted.patch_tokens.shape[2] == settings.reference_dim

    model = DummyModel()
    control = excited(apply_fourier_control(model, settings))
    with control.reference(extracted):
        assert model(torch.randn(2, SEQUENCE, WIDTH)).shape == (2, SEQUENCE, WIDTH)
    print('real dinov2: ok')


if __name__ == '__main__':
    torch.manual_seed(0)
    test_resolve_grid()
    test_preprocess_aspect_ratio()
    test_slots_are_disjoint_and_in_range()
    test_bins_are_scattered_not_contiguous()
    test_layers_place_their_bins_differently()
    test_narrow_layers_are_clamped()
    test_one_step_stays_small_next_to_the_layer()
    test_the_transform_is_full_rank()
    test_the_offset_is_not_symmetric()
    test_output_dim_of()
    test_the_transform_is_what_it_says()
    test_the_offset_is_per_channel_and_uniform()
    test_fresh_patch_is_a_no_op()
    test_no_reference_zeroes_the_reference_bins()
    test_per_sample_conditioning()
    test_strength_dials()
    test_scaling_defaults_come_from_the_config()
    test_cfg_batch_broadcast()
    test_variable_reference_shapes()
    test_gradients_reach_everything()
    test_gradient_checkpointing()
    test_the_hook_lands_on_the_lora_layer()
    test_the_peft_checkpoint_stays_clean()
    test_the_model_state_dict_is_untouched()
    test_save_load_and_removal()
    test_a_legacy_checkpoint_says_so()
    test_real_dinov2()
    print('\nall good')

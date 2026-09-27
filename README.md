# Lightweight ControlNet

A reference image conditioning path built out of FourierFT: a few thousand numbers and one 1d FFT per
layer, instead of a second copy of the model.

A ControlNet duplicates the trainable half of the base model and wires it back layer by layer, so the
VRAM bill is roughly twice the model plus the activations of the copy. That is the part this module
removes. What replaces it is a sparse spectrum per layer, half of it learned and half of it written by
the reference image, whose inverse transform is added to that layer's output.

## Where the features go

[FourierFT](https://arxiv.org/abs/2405.03003) observes that a weight delta does not have to be
parameterized in weight space: a handful of entries of its spectrum, inverse transformed, is a dense
delta at a fraction of the parameters. This takes the same idea to the *output* of a layer, and puts
the reference image in the spectrum next to the learned entries.

For every controlled layer, a complex half-spectrum with `out_features // 2 + 1` bins is built out of
zeros, two disjoint sets of its bins are written, and its inverse real FFT is added to the layer's
output:

```
z  = zeros(bins) complex,  z[learned_bins]   = spectrum  * spectrum_scaling
                           z[reference_bins] = reference * reference_scaling
z' = irfft(z, n=out_features)
y  = layer(x) + z'
```

`spectrum` is `n_frequency` learned complex coefficients - two reals each - zero initialized as in
FourierFT. `reference` is a `cond_dim -> 2 * n_reference` projection of a summary vector describing the
reference image, computed once per step by a shared conditioner and read by every layer through its own
projection. Both sets are scattered over individual bins of one per-layer permutation, the way
FourierFT picks its frequencies, so each reaches every scale of the spectrum instead of occupying one
band of neighbouring frequencies. They never overlap: a step with no reference image is exactly the
reference bins left at zero, and that must not disturb the learned ones.

"Output dimension" is `out_features` for a Linear and `out_channels` for a Conv2d, so `z'` is **one
number per output channel**.

### Why `irfft` and not `Real(ifft(.))`

FourierFT takes the real part of a full inverse transform of a real spectrum. That output is exactly
even-symmetric - `z'[c] == z'[N-c]`, with `max |z'[c] - z'[N-c]| = 0.00e+00` - so the map has rank
`floor(N/2)+1` rather than `N`. Two things follow, both bad: entries written past that rank cost
parameters and optimizer state without reaching any offset the others could not (at `N=3072` with the
ratios summing to `0.75`, 767 of 2304 entries were linearly redundant), and every entry ties channel
`c` to channel `N-c`, a pair with nothing to do with each other.

An inverse *real* FFT over a complex half-spectrum is the same idea without the symmetry to pay for:
`2 * bins - 2` free reals, which is exactly `N`. Measured by building the map column by column, the
rank comes out `N` at `N = 38, 64, 320, 321, 768, 3072`. The parameter count does not change - a
fraction `f` of the bins is `2 * f * bins = f * N` reals, exactly what a fraction `f` of the entries
used to cost - so this is the same size with nothing redundant in it.

One small waste is inherent to a real output rather than to the parameterization: `irfft` ignores the
imaginary part of bin 0, and of the Nyquist bin when `N` is even, because a real signal cannot use
them. That is at most 2 reals out of roughly `N`, against the `N/2` the symmetric form wasted.

### What that does and does not buy

The reference contribution is per sample for free. Every image of a batch writes its own block and
gets its own `z'`, at the cost of one 1d FFT per layer - which is what makes this usable at all. The
weight-space form of FourierFT cannot do that: a per-sample delta there is a per-sample `out x in`
matrix, and the layer becomes a batched matmul over it.

The offset is uniform over the frame. One number per channel is added to every token of a sequence and
over the whole extent of a feature map, so there is no per-position channel: the path cannot draw a
mask or hold an edge. What it is instead is a global instruction, the same category as sdxl's
`crop_coords_top_left` micro-conditioning - also a spatially uniform per-channel bias, and one that
does move the framing of an image. Coarse placement and size are therefore reachable; the edge accuracy
of a real ControlNet is not.

Whether *where* survives into the summary at all is `pool_grid`'s job. Measured on four 160px blobs at
the four corners of a 512px frame, through dinov2-base and the conditioner: at `pool_grid 1` the four
vectors sit at cosine 0.996 of each other - a 6-9% relative change, less than a size change produces -
so position is all but absent. At `pool_grid 2`, the default, the corners separate to cosine 0.95 and a
24-31% relative change, as strong as the size signal. Raise it for references whose whole content is
layout; it costs one shared `cond_dim * g^2 -> cond_dim` projection and nothing per layer.

### Nothing of the model is rewritten

The path is attached as a `register_forward_hook` per layer. The model's module tree, its parameters
and its state dict are exactly what they were, which is what makes everything downstream ordinary:

- a peft adapter trained next to this saves as an ordinary peft adapter, with nothing to unpatch
  first and nothing for a loader to not recognize,
- a diffusers model saves as a diffusers model,
- the base weights are untouched, so a frozen base model stays frozen and a 4bit one stays 4bit.

The layers can be peft LoRA layers, and the hook then fires after the adapter so the offset lands on the
layer's whole output. But they do not have to be, and a run with no adapter at all is the cheapest form
of this: freeze the base model, hook it, and the control path is the only thing that trains. That is
what `common/lw_controlnet.py` does when a training config leaves `lora_rank` out.

The price of hooking is that the trainable weights are **not** reached by `model.parameters()`, since
they live in the `FourierControl` object. Register it as a submodule - `model.add_module(
'fourier_control', control)` - and they are, which is what an optimizer, accelerate's `prepare` and
the gradient synchronization of a multi gpu run all read. `common/lw_controlnet.attach_control` does
this, and `detached_control` takes it off again for the duration of a save.

### Two scalings, and why 1.0 rather than FourierFT's 150

Under `ifft_norm: ortho` on a 1-D transform there is nothing to compensate for. The `1/sqrt(N)` the
transform applies is cancelled by the `~N/2` bins that contribute, so `std(z')` comes out at roughly
`std(parameter)` whatever `N` is - measured at 0.58 to 1.28 across `N` from 320 to 3072 and fill
ratios from 0.25 to 0.75. That near-invariance is also what lets one gain cover a 320 channel
convolution and a 3072 wide linear without being retuned per width.

FourierFT's 150 belongs to its own setting: a 2-D transform spreading ~1000 entries over
`out * in ~ 1e7`, a fill ratio near `1e-3` that really does attenuate the output by orders of
magnitude. Ours is 0.5 to 0.75 and attenuates nothing, so 150 is 150x too hot. Measured on a 1280 wide
layer at `lr 5e-4`, one AdamW step at gain 150 moves the layer by **69% of its own output**; over the
454 layers an SDXL run hooks, that is noise rather than an image. At gain 1.0 the same step moves it
by 0.46%.

Both gains are plain python floats on every layer, not parameters: they never reach a state dict, and
they are therefore inference dials as well. `control.strength` writes the reference one everywhere,
`control.spectrum_strength` the learned one.

The reference projection carries an extra `1/sqrt(cond_dim)` correction, and it is load bearing. The
two blocks are written at the same gain but they are not the same kind of weight: the spectrum is
written into `z` directly, so one optimizer step moves it by about the learning rate, while the
reference block is the output of a `cond_dim` wide matmul, so one step of the same size moves it by
`sqrt(cond_dim)` times as much. Measured without the correction, at `lr=1e-4` on a 3072 wide layer
whose output has std 0.58, the reference block reached `|z'| = 0.89` after a **single** step while the
learned block was at 0.009 - the reference drowned the layer before the model had learned anything to
say with it. With the correction the two land at 0.056 and 0.009 and grow together.

### Zero initialization

The spectrum and the reference projection are both zero, so a freshly attached path is bit identical
to the model underneath it. The projection has to be zero rather than small for the reason above:
the offset lands at the scale of the parameters themselves, so random values there would move the
layer's output by about as much as the layer does.

One consequence is worth knowing: the shared conditioner reaches the loss only through that zero
projection, so it takes no gradient on the very first step. The projection itself does, and from the
second step on so does the conditioner. This is the same shape as peft's zero `lora_B`, and unlike the
architecture this replaced it is not a trap - the control path adds to the layer's output directly, so
its own gradient never waits on another weight, and `lw_controlnet_freeze_path` is safe from step 0.

## The reference summary

The shared `ReferenceConditioner` runs once per step and turns the frozen dinov2 output into the
`(batch, cond_dim)` vector every layer projects from. Three things go into it:

- the patch field, average pooled onto a `pool_grid x pool_grid` grid and flattened, so a coarse sense
  of layout survives instead of only a global mean,
- the cls token, dinov2's own summary, which carries what the pooling averages away,
- the shape of the reference - its aspect ratio and how many patches it was read at - injected the way
  sdxl injects its size conditioning, since the pooled grid is the same size whether the reference was
  square or a panorama.

The patch field is normalized by `FieldNorm` rather than a `LayerNorm`, and that choice matters. A
LayerNorm removes each token's mean over the channels, which leaves the mean *over patches* untouched -
the component saying nothing about what is where, and as large as the part that does (measured at
`||mean|| 32` against `||deviation|| 38`). It also rescales every token to the same length, which
erases the contrast a reference is made of: on a shape mask that is 96% background, normalizing the
empty patches up to match the few that carry the shape is the opposite of what a reader needs. So the
mean is taken across the patch axis and one scale is shared by the whole field.

## Aspect ratio

The reference does not go through the aspect ratio buckets of the diffusion path. Instead of the square
resize and center crop of the default dinov2 processor, it is resized to a constant pixel budget with
its ratio kept: `resolve_grid` picks a patch grid holding about `pixel_budget**2` pixels, and dinov2
interpolates its position embeddings to whatever grid it receives, so `37x37`, `52x26` and `26x52` are
all valid. A reference whose subject is off to one side stays off to one side.

Images of one batch that resolve to different grids are resized to the largest and the patches outside
their own grid are marked as padding, which the pooling excludes from both its sum and its count.
Feeding the references through a bucket sampler keeps that branch unused.

## Cost

Per controlled layer: `2 * n_frequency` parameters for the spectrum and `cond_dim * 2 * n_reference`
for the projection, plus one inverse real FFT of length `out_features` per step. None of that grows with the width of the
base layer beyond the transform itself, and no activation is kept beyond `z'`, which is one row per
sample.

At the defaults - `cond_dim=256`, `frequency_ratio=0.5`, `reference_ratio=0.25` - a 3072 wide layer
has 1537 bins, spends 768 of them on the spectrum and 384 on the reference, and costs
`1536 + 256 * 768 = 198k` parameters, of which the projection dominates. Turn `cond_dim` or
`reference_ratio` down to shrink it; both narrow the channel the reference has into the model, which is
the trade this mechanism is made of.

Shared, once: the conditioner, which is a few hundred thousand parameters, and the frozen dinov2, which
can be dropped entirely by precomputing `ReferenceFeatures` offline.

## Usage

```python
from lightweight_controlnet import FourierControlConfig, apply_fourier_control

config = FourierControlConfig(target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'], cond_dim=256)
control = apply_fourier_control(transformer, config, device='cuda')

# so that the optimizer, accelerate and DDP see the weights
transformer.add_module('fourier_control', control)
optimizer = torch.optim.AdamW(control.parameters(), lr=1e-4)

with control.reference(reference_features):      # or None, to train the unconditional path
    loss = ...
    loss.backward()
```

`reference_features` is a `ReferenceFeatures`, either precomputed offline or produced by
`ReferenceEncoder(config)(images)`. Passing a list of `(3, h, w)` float tensors works too when an
encoder was attached with `apply_fourier_control(..., encoder=...)`.

Nothing about the resolution, the patch size or the text sequence length has to be declared. `z` is as
long as the layer's output dimension, so the sequence layout never enters - which is also why a
validation or inference pass cannot be misconfigured into reading the reference differently than
training did.

## Control strength

```python
with control.strength_scope(0.0):                # exactly the model with no reference image
    ...
control.strength = 2.0 * config.reference_scaling    # over drive the reference
control.spectrum_strength = 0.0                      # and now the base model, learned block off
```

`strength` comes up at `config.reference_scaling` and `spectrum_strength` at
`config.spectrum_scaling`, the values training runs at, so leaving them alone keeps a run as it was.
`strength = 0` is not an approximation of "no reference": it zeroes the same entries a reference-free
step leaves at zero, so the two outputs are identical.

## Gradient checkpointing and checkpoints

A checkpointed block replays its forward from inside backward, and the hooks read the conditioning as
they go. Keeping the backward inside the `with control.reference(...)` block is the simple way to make
that correct. `ConditioningContext` also remembers what the last real forward pass observed and serves
that to a replay, so the usual forward / close / backward order works too - including a dropped step,
which replays as a dropped step. The shape it cannot reconstruct is two forward passes under different
references with both backwards after them; keep each backward in its own scope.

`control.save_pretrained(directory)` writes `fourier_control.pt` and `fourier_control_config.json`.
The bin placements are saved as buffers, so a checkpoint does not depend on `random_loc_seed` staying
the same in a later run. A peft adapter written into the same directory needs no
special handling, since the model was never patched.

A directory holding `reference_control.pt` is a checkpoint of the cross attention architecture this
replaced; loading one is refused by name rather than half succeeding.

### What this replaced, and why

Two earlier revisions conditioned the LoRA rank tensor. The first made both projections bilinear in the
activation and the reference, which reached each layer as four numbers - a style dial and little else.
The second replaced it with a per-layer cross attention from the rank tensor to a resampled memory of
the reference, with 2d position embeddings on both sides so that the layers could learn a spatial
correspondence. Neither produced a model that followed its reference.

This revision gives up on spatial placement rather than on the reference, and buys directness with it:
the offset is added to the layer's output, not routed through peft's zero initialized `lora_B`, so
there is no path that can be invisible at initialization and no sequence layout to declare or get
wrong. The two designs are not compatible and their checkpoints are not interchangeable.

## Test

```
python lightweight_controlnet/tests/test_shapes.py
LWCN_TEST_DINOV2=1 python lightweight_controlnet/tests/test_shapes.py   # also run the real encoder
```

Runs on cpu without downloading anything: the reference features are drawn at random with the shapes
dinov2 would produce.

## Files

| file | what is in it |
| --- | --- |
| `config.py` | `FourierControlConfig`, and the per-layer slot counts it resolves |
| `reference_encoder.py` | the frozen dinov2 wrapper, `ReferenceFeatures`, and the aspect ratio handling |
| `conditioner.py` | the shared conditioner and `FieldNorm`: dinov2 output to one summary vector |
| `spectrum.py` | `FourierSpectrum` - the sparse half-spectrum, the transform, and the broadcast of `z'` |
| `context.py` | `ConditioningContext`, including the gradient checkpointing replay |
| `patch.py` | `FourierControl`, the hooks, the dials and the checkpointing |
| `example_usage.py` | a runnable training step against a dummy transformer |

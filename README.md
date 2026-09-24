# Lightweight ControlNet

A reference image conditioning path that costs a LoRA instead of a second copy of the model.

A ControlNet duplicates the trainable half of the base model and wires it back layer by layer, so
the VRAM bill is roughly twice the model plus the activations of the copy. That is the part this
module removes: the model already has a trainable side channel of its own, the LoRA adapters, so
the conditioning is injected there instead of in a duplicated trunk.

## Where the features go

A LoRA layer computes

```
y = W0 @ x + scale * B(A(x))          A: (M, r), B: (r, N)
```

The rank `r` tensor `z = A(x)` is the whole capacity of the adapter: everything the adapter can
add to the layer has to pass through it. So that is the connection point. `RankModulator` sits
between `A` and `B` and rewrites `z` as a function of the reference image:

```
z <- z * (1 + gamma(c)) + beta(c)                 global, FiLM
z <- z + out_proj(attention(q = z, k = v = c))    spatial, cross attention
```

where `c` is the reference conditioning. The base weights `W0` never move, `A` and `B` stay the
LoRA weights peft already owns, and everything added is a function of `r`, not of `M` or `N`.
A 2048 wide attention layer at rank 16 gets a handful of 16 wide projections rather than a copy of
the block, which is what makes this affordable on a small card.

Both branches start as an exact identity, the way a ControlNet zero convolution does: `film` is
zero initialized so `gamma = beta = 0`, and `out_proj` is zero initialized so the attention adds
nothing. A model patched at step zero produces bit identical output, so an already trained LoRA can
be patched and resumed without a discontinuity.

Only one tensor per branch is zeroed, on purpose. An earlier revision multiplied the zeroed
`out_proj` by a zero initialized scalar gate as well; the test caught that this is unrecoverable -
the gradient of the gate is the zeroed projection output and the gradient of the projection is
scaled by the zero gate, so both stay at zero forever and the spatial branch never trains.

### Two stages, not one per layer

`ReferenceConditioner` runs once per step and is shared by every modulator:

```
image -> [frozen dinov2] -> patch grid + cls  -> [resampler] -> 64 latent tokens (cond_dim)
                                              -> [proj + aspect ratio mlp] -> global vector
```

The resampler turns a variable length patch grid into a fixed, small set of latents, so the
per-adapter attention is `O(r * num_latents)` regardless of the resolution of the reference. The
per-layer modulators are the only thing that scales with the number of patched layers, and they
are tiny: at `r = 16`, `cond_dim = 256` that is about 17k parameters per layer.

The dinov2 encoder is frozen, runs under `no_grad`, holds no optimizer state and can be kept in
bfloat16 next to the frozen base model. Better, its output is a plain `ReferenceFeatures` dataclass,
so it can be precomputed offline into the dataset shards exactly like the other precomputed
features in this repository, and then the encoder does not have to be resident at all during
training.

## Aspect ratio

The default dinov2 processor resizes to 224x224 and center crops, which both distorts the
reference and throws away its edges. Three things replace it.

1. **A pixel budget instead of a fixed square.** `resolve_grid` picks a patch grid whose ratio
   matches the source and whose area is about `pixel_budget**2` pixels, both sides a multiple of
   the patch size. Dinov2 interpolates its position embeddings to whatever grid it is given, so
   37x37, 28x49 and 49x28 are all valid inputs at a constant compute cost:

   ```
   1024x1024 -> 37x37 (1369 patches)    768x1344 -> 28x49 (1372)    2048x512 -> 74x18 (1332)
   ```

   A `min_patches_per_side` floor keeps very elongated references from collapsing to a single row.

2. **A 2d position embedding on the patch tokens.** `sincos_2d_embedding` is built from the actual
   `(patches_h, patches_w)` of the batch with normalized coordinates, so a feature keeps the same
   meaning whether the reference was 26x52 or 52x26, and the modulators see where something sits
   in the reference rather than only its index in a sequence.

3. **The aspect ratio as an explicit conditioning scalar.** `log(w/h)` and `log(num_patches)` go
   through a small MLP into the global vector, the way SDXL feeds its size conditioning, so the
   model can tell a portrait reference from a landscape one instead of having to infer it.

Batches that mix aspect ratios are resized to the grid of the largest member and the patches
outside each image are marked in a padding mask that the resampler attends around. Feeding the
reference through the aspect ratio bucket sampler this repository already uses for the diffusion
path keeps that branch unused and wastes nothing.

## Usage

```python
from peft import LoraConfig, get_peft_model
from lightweight_controlnet import ReferenceControlConfig, ReferenceEncoder, apply_reference_control

transformer = get_peft_model(transformer, LoraConfig(r=16, lora_alpha=32,
                                                     target_modules=['to_q', 'to_k', 'to_v']))

config = ReferenceControlConfig(adapter_name='default', cond_dim=256, num_latents=64)
control = apply_reference_control(transformer, config, encoder=ReferenceEncoder(config))

# the modulators live inside the base model, so model.parameters() already returns them;
# only the shared conditioner has to be added
optimizer = torch.optim.AdamW([
    {'params': [p for p in transformer.parameters() if p.requires_grad]},
    {'params': list(control.shared_parameters())},
], lr=1e-4)

with control.reference(reference_images):          # or precomputed ReferenceFeatures
    noise_prediction = transformer(latents, timestep, encoder_hidden_states).sample
```

`control.reference(None)` runs the model as a plain LoRA, which is what reference dropout for
classifier free guidance needs. `remove_reference_control(model, config)` puts the original
`lora_A` modules back. `control.save_pretrained(...)` writes the conditioner and the modulators
next to the peft adapter, which stays a normal peft checkpoint.

See `example_usage.py` for a full training step, including reference dropout and the precomputed
features path.

## Test

```
python lightweight_controlnet/tests/test_shapes.py        # add LWCN_TEST_DINOV2=1 for the real encoder
```

The dummy model is a small attention stack standing in for a diffusion transformer. The test
checks the resolved grids against the source ratios, the padding mask on a mixed ratio batch, that
the patched model is shape preserving and bit identical at init, that two different references give
two different outputs once the projections are non zero, that reference grids from 8x60 to 37x37 all
work on the same patched model, that gradients reach the reference path and the LoRA while the base
weights stay frozen, that neither zero initialized branch is dead, the Conv2d LoRA path, and save,
reload and removal of the patch.

## Files

| file | what it holds |
| --- | --- |
| `config.py` | `ReferenceControlConfig`, the PEFT style config |
| `reference_encoder.py` | aspect ratio preserving preprocessing, frozen dinov2, `ReferenceFeatures` |
| `conditioner.py` | the shared trunk: resampler, 2d position embedding, aspect ratio conditioning |
| `modulator.py` | `RankModulator` and the `lora_A` wrapper that inserts it |
| `patch.py` | `apply_reference_control` / `remove_reference_control`, `ReferenceControl` |
| `tests/test_shapes.py` | the checks above |

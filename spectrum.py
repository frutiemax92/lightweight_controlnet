"""The control path of one layer: a sparse half-spectrum, an inverse real FFT, and an offset on the
layer's output."""
import zlib
from typing import Optional

import torch

from lightweight_controlnet.conditioner import ReferenceConditioning
from lightweight_controlnet.config import FourierControlConfig

try:                                                # transformers is optional for this package
    from transformers.pytorch_utils import Conv1D
except Exception:                                   # pragma: no cover
    Conv1D = ()


def output_dim_of(module: torch.nn.Module):
    """The (output dimension, channel_last) of a layer, or None when it has no single one.

    channel_last says which axis of the layer's output the dimension indexes: the last one for a
    Linear, whose output is (..., out_features), and axis 1 for a convolution, whose output is
    (batch, out_channels, ...). That is the axis z' is broadcast along.

    A peft or bitsandbytes wrapper is unwrapped first, so a 4bit Linear under a LoRA reports the
    out_features of the Linear it replaced rather than the shape of its packed weight.
    """
    base = module.get_base_layer() if hasattr(module, 'get_base_layer') else module
    if isinstance(base, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
        return base.out_channels, False
    if isinstance(base, torch.nn.Linear):           # bitsandbytes' Linear4bit/Linear8bit included
        return base.out_features, True
    if Conv1D and isinstance(base, Conv1D):
        # gpt2 style: the weight is stored (in_features, out_features)
        shape = base.weight.ds_shape if hasattr(base.weight, 'ds_shape') else base.weight.shape
        return int(shape[1]), True
    # anything else that names its own width, which is how peft's own layers describe themselves
    if hasattr(base, 'out_features') and isinstance(getattr(base, 'out_features'), int):
        return base.out_features, True
    if hasattr(base, 'out_channels') and isinstance(getattr(base, 'out_channels'), int):
        return base.out_channels, False
    return None


def layer_seed(name: str, config: FourierControlConfig) -> int:
    """A per-layer seed for the bin placement, stable across processes and runs.

    crc32 rather than hash(): python salts hash() per process, and two ranks of a distributed run
    that place their bins differently would train two different models.
    """
    return (config.random_loc_seed + zlib.crc32(name.encode())) % (2 ** 31)


def choose_slots(bins: int, n_learned: int, n_reference: int, seed: int):
    """Which bins of the half-spectrum the learned coefficients and the reference get.

    Both are scattered over individual bins, drawn from one permutation, the way FourierFT picks its
    frequencies - so each block reaches every scale of the spectrum at once instead of occupying one
    band of neighbouring frequencies.

    They never overlap: the reference has to be able to go to zero without taking the learned part of
    the spectrum with it, since that is what a step with no reference image is.
    """
    permutation = torch.randperm(bins, generator=torch.Generator().manual_seed(seed))
    return (permutation[:n_learned].sort().values,
            permutation[n_learned:n_learned + n_reference].sort().values)


class FourierSpectrum(torch.nn.Module):
    """
    The control path of one layer.

    A complex half-spectrum with `out_features // 2 + 1` bins is built out of zeros, two disjoint
    sets of bins are written into it, and the inverse real FFT of it is what the layer's output is
    offset by:

        z  = zeros(bins) complex,  with  z[learned_bins]   = spectrum   * spectrum_scaling
                                         z[reference_bins] = reference  * reference_scaling
        z' = irfft(z, n=out_features)
        y  = layer(x) + z'

    `irfft` rather than `Real(ifft(...))`, and that is not a detail. Taking the real part of a full
    inverse transform of a *real* spectrum - what FourierFT does - returns an exactly even-symmetric
    vector: `z'[c] == z'[N - c]` to the last bit, so the map has rank `N // 2 + 1` rather than `N`.
    Entries written past that rank cost parameters and optimiser state without reaching any offset
    the others could not, and every one of them ties channel `c` to channel `N - c`, a pair with
    nothing to do with each other. An inverse *real* FFT over a complex half-spectrum is the version
    of the same idea with no symmetry to pay for: `2 * bins - 2` free reals, which is exactly `N`, so
    every offset is reachable and every parameter is independent.

    Each written bin is therefore a complex coefficient, two learned reals. `spectrum` is
    `(n_frequency, 2)` and the reference projection produces `n_reference * 2` numbers, so a ratio of
    the bins costs the same as that ratio of `out_features` did under the old form - the parameter
    count is unchanged and only the redundancy is gone.

    The reference part is per sample: every image of the batch writes its own bins and gets its own
    `z'`, at the cost of one 1d FFT per layer. When the step carries no reference image those bins
    stay zero, which leaves plain FourierFT behind.

    Both sets are written through a scaling factor that is a plain python float rather than a
    parameter, so neither reaches a state dict and both are inference dials. They come up at
    `config.spectrum_scaling` and `config.reference_scaling`, which default to 1.0 rather than to the
    150 FourierFT uses. With norm='ortho' on a 1d transform there is nothing to compensate for: the 1/sqrt(N) the transform applies is cancelled by the ~N/2 bins that contribute, so std(z') comes out at roughly std(parameter) whatever N is. FourierFT's 150 belongs to its own setting - a 2d transform where ~1000 entries are spread over out*in ~ 1e7, a fill ratio near 1e-3 that does attenuate hugely. Ours is 0.5 to 0.75 and attenuates nothing.

    Two things are zero initialized - the spectrum, as in FourierFT, and the reference projection -
    so a freshly attached path is bit identical to the model underneath it. The projection is what
    has to be zero rather than small: with norm='ortho' the offset comes out at roughly the scale of
    the parameters themselves, so random values there would move the layer's output by about as much
    as the layer does before a single step has run. The cost
    is that the shared conditioner sees no gradient on the very first step, because it reaches the
    loss only through that zero projection; the projection itself does get one, and from the second
    step on so does the conditioner.

    The projection's output carries a 1/sqrt(cond_dim) correction, and it is not cosmetic. The two
    halves are written at the same gain but they are not the same kind of weight: the spectrum is
    written into z directly, so one optimizer step moves it by about the learning rate, while the
    reference bins are the output of a cond_dim wide matmul, so one step of the same size moves them
    by sqrt(cond_dim) times as much - 16 times, at the default 256. Without the correction the
    reference reached the scale of the layer's own output after a single step at lr 1e-4 while the
    learned part was still two orders of magnitude below it, and the reference drowned the model
    before it had learned anything to say. The correction puts one step on either side at the same
    size, which is what makes one learning rate cover both.

    One small waste is inherent to a real output rather than to this parameterization: irfft ignores
    the imaginary part of bin 0, and of the Nyquist bin when out_features is even, because a real
    signal cannot use them. Those one or two reals take no gradient if a block happens to cover
    those bins. That is at most 2 numbers out of roughly N, against the `N // 2` the symmetric form
    wasted.
    """
    def __init__(self, out_features: int, config: FourierControlConfig, name: str = ''):
        super().__init__()
        self.out_features = int(out_features)
        # the half-spectrum irfft consumes: the only bins that exist for a real output
        self.bins = self.out_features // 2 + 1
        n_learned, n_reference = config.slots_for(self.out_features)

        self.ifft_norm = config.ifft_norm
        # inference time dials, see the class docstring. Not parameters and not buffers.
        self.spectrum_scaling = float(config.spectrum_scaling)
        self.reference_scaling = float(config.reference_scaling)

        # (bins, 2): the real and the imaginary part of each learned coefficient
        self.spectrum = torch.nn.Parameter(torch.zeros(n_learned, 2))
        self.reference_proj = torch.nn.Linear(config.cond_dim, n_reference * 2, bias=False)
        torch.nn.init.zeros_(self.reference_proj.weight)
        # fan-in correction, see the class docstring: it is what puts one optimizer step on the
        # reference bins and one on the spectrum at the same size
        self.reference_gain = float(config.cond_dim) ** -0.5

        learned_bins, reference_bins = choose_slots(
            self.bins, n_learned, n_reference, layer_seed(name, config))
        # buffers rather than a seed recomputed on load, so a checkpoint keeps its own placement
        # whatever random_loc_seed a later run is configured with
        self.register_buffer('spectrum_indices', learned_bins.long(), persistent=True)
        self.register_buffer('reference_indices', reference_bins.long(), persistent=True)

    @property
    def n_frequency(self) -> int:
        """Learned bins. Each one holds two reals, so the parameter count is twice this."""
        return self.spectrum.shape[0]

    @property
    def n_reference(self) -> int:
        return self.reference_proj.out_features // 2

    def forward(self, conditioning: Optional[ReferenceConditioning]) -> torch.Tensor:
        """(rows, out_features) real offset, with `rows` the reference batch or 1 without one."""
        # the transform has no half precision kernel, and the spectrum is a handful of numbers per
        # layer, so it runs in float32 whatever the model is in and the result is cast on the way out
        z = self.spectrum.new_zeros(1, self.bins, 2, dtype=torch.float32)
        z = z.index_copy(1, self.spectrum_indices,
                         (self.spectrum.float() * self.spectrum_scaling)[None])

        if conditioning is not None and self.reference_scaling != 0.0:
            weight = self.reference_proj.weight
            reference = self.reference_proj(conditioning.vector.to(weight.dtype)).float()
            reference = reference.unflatten(-1, (self.n_reference, 2)) * (
                self.reference_gain * self.reference_scaling)
            rows = reference.shape[0]
            # the two sets of bins are disjoint, so adding the reference is writing it
            z = z + torch.zeros(rows, self.bins, 2, dtype=torch.float32, device=z.device
                                ).index_copy(1, self.reference_indices, reference)

        return torch.fft.irfft(torch.view_as_complex(z.contiguous()),
                               n=self.out_features, norm=self.ifft_norm)


def broadcast_delta(delta: torch.Tensor, batch: int) -> torch.Tensor:
    """Tile z' up to the batch the layer is running on.

    Classifier free guidance runs the model on the negative and the positive halves concatenated, so
    the sampling batch is a whole multiple of the batch the reference was encoded at - one reference
    per image, two passes over it. Tiling rather than interleaving is what matches that
    concatenation. A single row broadcasts on its own and is left alone, which covers both the
    one-reference case the pipelines actually hit and a step with no reference at all.
    """
    rows = delta.shape[0]
    if rows == batch or rows == 1:
        return delta
    if rows == 0 or batch % rows:
        raise ValueError(f'the reference conditioning has batch {rows}, which does not divide the '
                         f'{batch} the model is running on')
    return delta.repeat(batch // rows, 1)


def add_delta(output: torch.Tensor, delta: torch.Tensor, channel_last: bool) -> torch.Tensor:
    """output + z', broadcast along every axis of the output that is not the channel axis.

    A Linear on a token sequence gets the same offset on every token, a Conv2d the same offset over
    its whole feature map. That is what a spectrum whose transform is as long as the output dimension
    can say: one number per channel, and nothing about where in the frame it applies.
    """
    delta = broadcast_delta(delta, output.shape[0])
    if channel_last:
        shape = (delta.shape[0],) + (1,) * (output.dim() - 2) + (delta.shape[1],)
    else:
        shape = (delta.shape[0], delta.shape[1]) + (1,) * (output.dim() - 2)
    return output + delta.reshape(shape).to(output.dtype)

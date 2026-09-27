import json
import os
import warnings
from dataclasses import dataclass, asdict, field, fields
from typing import Optional, Union


@dataclass
class FourierControlConfig:
    """
    The code style is inspired from PEFT.

    This config describes a FourierFT style control path. For every layer it is applied to, a complex
    half-spectrum with `out_features // 2 + 1` bins is built out of zeros, two disjoint sets of its
    bins are written - one from learned parameters, one from a projection of the reference image
    features - and its inverse real FFT is added to the layer's output:

        z  = zeros(bins) complex, with scattered bins set from spectrum * spectrum_scaling
                                  and from reference * reference_scaling
        z' = irfft(z, n=out_features)
        y  = layer(x) + z'

    An inverse *real* FFT rather than the real part of a full one, which is what FourierFT takes. The
    real part of the inverse transform of a real spectrum is exactly even-symmetric -
    `z'[c] == z'[N - c]` to the last bit - so that map has rank `N // 2 + 1` instead of `N`, and every
    entry written past that rank buys parameters and optimiser state without reaching any offset the
    others could not, while tying channel `c` to channel `N - c`, a pair with nothing to do with each
    other. irfft over a complex half-spectrum has `2 * bins - 2` free reals, which is exactly `N`:
    full rank, nothing redundant, no arbitrary tie between channels.

    `z'` is as long as the layer's output dimension - `out_features` for a Linear, `out_channels` for
    a Conv2d - so it is one number per output channel, broadcast over the tokens of a sequence or over
    the spatial extent of a feature map. That is what makes the reference contribution free to be per
    sample - every image of the batch gets its own `z'` at the cost of one 1d FFT per layer - and it
    is also the shape of its limit: there is no per-position channel, so the path cannot draw a mask
    or hold an edge. It is a global instruction, the same category as sdxl's crop_coords micro
    conditioning, which is also a spatially uniform per-channel bias and does move an image's framing.
    Coarse placement and size are reachable; the edge accuracy of a ControlNet is not. How much of
    *where* reaches the model at all is `pool_grid`'s doing - see its help and the README.

    Nothing about the base layer is touched. The path is attached as a forward hook, so the module
    tree, the state dict and the peft adapter of the model being trained stay exactly what they
    were, and the trainable weights all live in the FourierControl object.
    """
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names of the layers to "
                "control. For example, ['to_q', 'to_v'] or '.*attn.*(to_q|to_v)$'. When None, every "
                "Linear and Conv2d of the model is controlled, which is what FourierFT does."
            )
        },
    )

    adapter_name: str = field(
        default='default',
        metadata={
            'help': (
                'Name of the peft adapter a run trains next to this path, when it trains one. The '
                'control path itself does not read or touch any adapter - it hooks the layer, '
                'whatever that layer turned out to be - this is only the name the callers hand to '
                'peft.'
            )
        }
    )

    # --- the reference encoder -------------------------------------------------------------------

    reference_model: str = field(
        default='facebook/dinov2-base',
        metadata={'help': 'Frozen image encoder used to extract the reference features.'}
    )

    reference_dim: int = field(
        default=768,
        metadata={'help': 'Hidden size of the reference encoder (768 for dinov2-base).'}
    )

    patch_size: int = field(
        default=14,
        metadata={'help': 'Patch size of the reference encoder. Reference images are resized to a multiple of it.'}
    )

    pixel_budget: int = field(
        default=518,
        metadata={
            'help': (
                'The reference image is resized so that it holds about pixel_budget**2 pixels while '
                'keeping its aspect ratio. 518 = 37x14, the native dinov2 resolution.'
            )
        }
    )

    min_patches_per_side: int = field(
        default=8,
        metadata={'help': 'Lower bound on the patch grid, so that very elongated references keep some detail.'}
    )

    # --- the shared conditioner ------------------------------------------------------------------

    cond_dim: int = field(
        default=256,
        metadata={
            'help': (
                'Width of the reference summary vector the conditioner produces once per step. It '
                'is the whole channel the reference has into the model: every controlled layer '
                'reads it through its own cond_dim x n_reference projection, so widening it costs '
                'cond_dim * n_reference per layer.'
            )
        }
    )

    pool_grid: int = field(
        default=2,
        metadata={
            'help': (
                'The reference patch field is average pooled onto a pool_grid x pool_grid grid '
                'before being projected into that summary vector, so a coarse layout survives the '
                'pooling instead of only a global mean. 1 is the plain mean. It cannot give the '
                'control path spatial placement - z is per channel and uniform over the frame - '
                'only more to say about the reference.'
            )
        }
    )

    # --- the spectrum of one controlled layer ----------------------------------------------------

    n_frequency: Optional[int] = field(
        default=None,
        metadata={
            'help': (
                'Number of learned bins, the same for every controlled layer. Each bin is one complex '
                'coefficient, so it costs two parameters. When None, frequency_ratio of the layer bin '
                'count is used instead, which is what keeps a 320 channel conv and a 3072 wide linear '
                'both in range.'
            )
        }
    )

    frequency_ratio: float = field(
        default=0.5,
        metadata={
            'help': (
                'Fraction of the layer bins spent on learned coefficients. Read when n_frequency is '
                'None. A fraction f of the bins is 2 * f * bins parameters, which is f * out_features, '
                'so this reads as the same size it always did - what changed is that every one of '
                'those parameters is now independent.'
            )
        }
    )

    n_reference: Optional[int] = field(
        default=None,
        metadata={'help': 'Number of bins the reference projection fills. When None, reference_ratio is used.'}
    )

    reference_ratio: float = field(
        default=0.25,
        metadata={'help': 'Fraction of the layer bins the reference fills. Read when n_reference is None.'}
    )

    spectrum_scaling: float = field(
        default=1.0,
        metadata={
            'help': (
                'Gain the learned coefficients are written into z at. 1.0, not the 150 FourierFT '
                "uses. With norm='ortho' on a 1d transform there is nothing to compensate for: the 1/sqrt(N) the transform applies is cancelled by the ~N/2 bins that contribute, so std(z') comes out at roughly std(parameter) whatever N is. FourierFT's 150 belongs to its own setting - a 2d transform where ~1000 entries are spread over out*in ~ 1e7, a fill ratio near 1e-3 that does attenuate hugely. Ours is 0.5 to 0.75 and attenuates nothing. A gain of 150 here is 150x too hot: at lr 5e-4 one optimizer step "
                'already moves a layer by 69% of its own output, and over a few hundred hooked '
                'layers that is noise. It is a plain float on the layer, so it doubles as an '
                'inference dial.'
            )
        }
    )

    reference_scaling: float = field(
        default=1.0,
        metadata={
            'help': (
                'Gain the reference projection is written into z at, and the value '
                'FourierControl.strength comes up at. 0.0 zeroes the reference bins, which is '
                'exactly the model with no reference image.'
            )
        }
    )

    ifft_norm: str = field(
        default='ortho',
        metadata={
            'help': (
                "Normalization of the inverse FFT: 'ortho' (1/sqrt(n)), 'backward' (1/n, torch's "
                "default) or 'forward' (none). 'ortho' is used here so that the scale of z' depends "
                "on the number of written bins rather than on how wide the layer happens to be, "
                "which is what lets one spectrum_scaling cover every layer of a model."
            )
        }
    )

    random_loc_seed: int = field(
        default=777,
        metadata={
            'help': (
                'Seed the per-layer bin placement is drawn from. Both sets are scattered over '
                'individual bins of one permutation, the way FourierFT picks its frequencies, so each '
                'of them reaches every scale of the spectrum instead of occupying one band of '
                'neighbouring frequencies. The placements are saved as buffers as well, so a '
                'checkpoint does not depend on this staying the same.'
            )
        }
    )

    dtype: str = field(
        default='float32',
        metadata={'help': 'Dtype of the trainable control parameters.'}
    )

    def __post_init__(self):
        if self.cond_dim <= 0:
            raise ValueError(f'cond_dim must be positive, got {self.cond_dim}')
        if self.pool_grid < 1:
            raise ValueError(f'pool_grid must be at least 1, got {self.pool_grid}')
        if self.ifft_norm not in ('ortho', 'backward', 'forward'):
            raise ValueError(f"ifft_norm must be 'ortho', 'backward' or 'forward', got {self.ifft_norm!r}")
        for name in ('frequency_ratio', 'reference_ratio'):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be in [0, 1], got {value}')
        # the two sets of bins are disjoint, so together they cannot ask for more than the
        # half-spectrum has. At exactly 1 the whole of it is written, which is 2 * bins - 2 =
        # out_features free reals: the most this parameterization reaches, still with no redundancy
        if self.frequency_ratio + self.reference_ratio > 1.0:
            raise ValueError('frequency_ratio + reference_ratio must be at most 1, got '
                             f'{self.frequency_ratio} + {self.reference_ratio}')

    def bins_for(self, out_features: int) -> int:
        """How many bins the half-spectrum of a layer this wide has."""
        return out_features // 2 + 1

    def slots_for(self, out_features: int) -> tuple[int, int]:
        """The (learned, reference) bin counts of a layer with this output dimension.

        Counted in bins, not in parameters: each bin is one complex coefficient and so two reals.

        Both are clamped so that they stay disjoint inside z and neither disappears: a layer narrow
        enough that the ratios round to nothing still gets one bin of each, and a layer that the
        absolute n_frequency / n_reference would overrun is cut down to fit rather than raising,
        since one config is applied to every width in the model.
        """
        if out_features < 2:
            raise ValueError(f'a controlled layer needs an output dimension of at least 2, got {out_features}')
        bins = self.bins_for(out_features)
        learned = self.n_frequency if self.n_frequency is not None else round(self.frequency_ratio * bins)
        reference = self.n_reference if self.n_reference is not None else round(self.reference_ratio * bins)
        learned = max(1, min(int(learned), bins - 1))
        reference = max(1, min(int(reference), bins - learned))
        return learned, reference

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'fourier_control_config.json'), 'w') as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        path = os.path.join(save_directory, 'fourier_control_config.json')
        if not os.path.exists(path):
            legacy = os.path.join(save_directory, 'reference_control_config.json')
            if os.path.exists(legacy):
                raise ValueError(
                    f'{save_directory} holds a reference_control_config.json, which is a checkpoint '
                    'of the cross attention architecture this replaced. Its weights describe modules '
                    'that no longer exist and cannot be loaded; retrain, or check out the commit it '
                    'was written by.')
            raise FileNotFoundError(path)
        with open(path, 'r') as file:
            stored = json.load(file)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(stored) - known)
        if unknown:
            warnings.warn(f'ignoring unknown keys in {save_directory}: {", ".join(unknown)}',
                          RuntimeWarning)
        return cls(**{key: value for key, value in stored.items() if key in known})

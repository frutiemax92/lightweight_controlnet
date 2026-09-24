import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Union


@dataclass
class ReferenceControlConfig:
    """
    The code style is inspired from PEFT.
    This config indicates how the reference image features are injected inside
    the rank-r bottleneck of the LoRA adapters.
    """
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names of the LoRA layers to "
                "modulate. For example, ['to_q', 'to_v'] or '.*attn.*(to_q|to_v)$'. "
                "When None, every LoRA layer found in the model is modulated."
            )
        },
    )

    adapter_name: str = field(
        default='default',
        metadata={'help': 'Name of the peft adapter whose rank tensor gets modulated.'}
    )

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

    cond_dim: int = field(
        default=256,
        metadata={'help': 'Width of the shared conditioning trunk, i.e. the keys/values fed to every adapter.'}
    )

    num_latents: int = field(
        default=64,
        metadata={
            'help': (
                'Number of latent tokens produced by the shared resampler. 0 keeps the raw patch tokens, '
                'which is more precise but makes the per-layer attention proportional to the patch count.'
            )
        }
    )

    num_heads: int = field(
        default=4,
        metadata={'help': 'Number of attention heads of the shared resampler.'}
    )

    use_film: bool = field(
        default=True,
        metadata={'help': 'Global (aspect-ratio aware) FiLM modulation of the rank tensor.'}
    )

    use_cross_attention: bool = field(
        default=True,
        metadata={'help': 'Spatial cross attention between the rank tensor and the reference tokens.'}
    )

    dtype: str = field(
        default='float32',
        metadata={'help': 'Dtype of the trainable modulation parameters.'}
    )

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'reference_control_config.json'), 'w') as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        with open(os.path.join(save_directory, 'reference_control_config.json'), 'r') as file:
            return cls(**json.load(file))

"""
A training step of a reference controlled FourierFT path, written against a dummy transformer so it
runs anywhere. Swap DummyTransformer for the real transformer and the random latents for the batch of
the trainer, the rest is what a real loop looks like.

    python lightweight_controlnet/example_usage.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import LoraConfig, get_peft_model

from lightweight_controlnet import FourierControlConfig, ReferenceFeatures, apply_fourier_control

REFERENCE_DROPOUT = 0.1


class DummyTransformer(torch.nn.Module):
    def __init__(self, width=128, depth=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.ModuleDict({
            'to_q': torch.nn.Linear(width, width),
            'to_k': torch.nn.Linear(width, width),
            'to_v': torch.nn.Linear(width, width),
            'to_out': torch.nn.Linear(width, width),
            'norm': torch.nn.LayerNorm(width),
        }) for _ in range(depth)])

    def forward(self, hidden_states):
        for block in self.blocks:
            residual = hidden_states
            hidden_states = block['norm'](hidden_states)
            attention = torch.nn.functional.scaled_dot_product_attention(
                block['to_q'](hidden_states), block['to_k'](hidden_states), block['to_v'](hidden_states))
            hidden_states = residual + block['to_out'](attention)
        return hidden_states


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    # 1. the frozen base model, with a LoRA adapter if the run wants one. The control path does not
    #    need it - it hooks the layer, whatever that layer turned out to be - but hooking the layers
    #    peft has replaced puts the offset on the layer's whole output, the adapter included.
    transformer = DummyTransformer().to(device)
    transformer = get_peft_model(transformer, LoraConfig(r=16, lora_alpha=32,
                                                        target_modules=['to_q', 'to_k', 'to_v']))

    # 2. the control path. Pass encoder=ReferenceEncoder(config) to run dinov2 inline; here the
    #    features are precomputed, which is the low vram path: no encoder resident during training.
    config = FourierControlConfig(target_modules=['to_q', 'to_k', 'to_v', 'to_out'], cond_dim=256)
    control = apply_fourier_control(transformer, config, device=device)

    # every weight of the control path lives in this object - the layers are hooked, not rewritten -
    # so it is a parameter group of its own next to whatever the base model has left trainable
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([{'params': trainable, 'lr': 1e-4},
                                   {'params': list(control.parameters()), 'lr': 1e-4}])

    print(f'lora tensors {len(trainable)} '
          f'({sum(parameter.numel() for parameter in trainable)} parameters)')
    print(f'control path {control.num_trainable_parameters()} parameters over '
          f'{len(control.layers)} layers plus the shared conditioner')
    for name, layer in list(control.layers.items())[:4]:
        print(f'  {name}: {layer.bins} bins for {layer.out_features} channels, '
              f'{layer.n_frequency} learned and {layer.n_reference} from the reference '
              f'({2 * layer.n_frequency} + {2 * layer.n_reference} reals)')

    # 3. the training loop
    for step in range(3):
        latents = torch.randn(2, 64, 128, device=device)
        target = torch.randn_like(latents)

        # what the dataloader would hand over, read back from the precomputed shards
        features = ReferenceFeatures(patch_tokens=torch.randn(2, 28 * 49, 768, device=device),
                                     cls_token=torch.randn(2, 768, device=device),
                                     grid=(28, 49),
                                     aspect_ratio=torch.full((2,), 49 / 28, device=device))

        # reference dropout, so the model also learns the path with the reference block of z at
        # zero, which is what guidance is guided against
        reference = None if random.random() < REFERENCE_DROPOUT else features

        # backward inside the block: a gradient checkpointed model replays the forward of its
        # blocks from inside backward, and the hooks read the conditioning as they go. Nothing about
        # the resolution has to be declared - z is as long as the layer's output dimension, so the
        # offset it writes is per channel and the sequence layout never enters
        with control.reference(reference):
            prediction = transformer(latents)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f'step {step}: loss {loss.item():.4f}, reference '
              f'{"dropped" if reference is None else "on"}')

    # 4. checkpointing: the peft adapter stays a normal peft checkpoint - the control path never
    #    replaced anything in the model, so there is nothing to undo first - and the control path is
    #    saved next to it
    # transformer.save_pretrained('out/fourier_control')
    # control.save_pretrained('out/fourier_control')

    # 5. the inference dials, both plain floats on every layer and neither of them saved
    control.strength = 0.0            # exactly the model with no reference image
    control.spectrum_strength = 0.0   # and now the base model plus the lora
    print('done')


if __name__ == '__main__':
    main()

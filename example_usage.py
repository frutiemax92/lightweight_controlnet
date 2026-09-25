"""
A training step of a reference controlled LoRA, written against a dummy transformer so it runs
anywhere. Swap DummyTransformer for the real transformer and the random latents for the batch of
the trainer, the rest is what a real loop looks like.

    python lightweight_controlnet/example_usage.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import LoraConfig, get_peft_model

from lightweight_controlnet import ReferenceControlConfig, ReferenceFeatures, apply_reference_control

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

    # 1. the frozen base model with its LoRA adapter, exactly as usual
    transformer = DummyTransformer().to(device)
    transformer = get_peft_model(transformer, LoraConfig(r=16, lora_alpha=32,
                                                         target_modules=['to_q', 'to_k', 'to_v']))

    # 2. the reference path. Pass encoder=ReferenceEncoder(config) to run dinov2 inline; here the
    #    features are precomputed, which is the low vram path: no encoder resident during training.
    config = ReferenceControlConfig(target_modules=None, cond_dim=256, num_latents=64, num_heads=4)
    control = apply_reference_control(transformer, config, device=device)

    # the modulators live inside the base model, so model.parameters() already returns them; only
    # the shared conditioner has to be added to the optimizer
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    control_parameters = list(control.shared_parameters())
    optimizer = torch.optim.AdamW([{'params': trainable, 'lr': 1e-4},
                                   {'params': control_parameters, 'lr': 1e-4}])

    print(f'lora + modulator tensors {len(trainable)} '
          f'({sum(parameter.numel() for parameter in trainable)} parameters)')
    print(f'reference path {control.num_trainable_parameters()} parameters over '
          f'{len(control.modulators)} modulators plus the shared conditioner')

    # 3. the training loop
    for step in range(3):
        latents = torch.randn(2, 64, 128, device=device)
        target = torch.randn_like(latents)

        # what the dataloader would hand over, read back from the precomputed shards
        features = ReferenceFeatures(patch_tokens=torch.randn(2, 28 * 49, 768, device=device),
                                     cls_token=torch.randn(2, 768, device=device),
                                     grid=(28, 49),
                                     aspect_ratio=torch.full((2,), 49 / 28, device=device))

        # reference dropout, so the model also learns the unconditional path for guidance
        reference = None if random.random() < REFERENCE_DROPOUT else features

        # backward inside the block: a gradient checkpointed model replays the forward of its
        # blocks from inside backward, and the modulators read the conditioning as they go
        with control.reference(reference):
            prediction = transformer(latents)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f'step {step}: loss {loss.item():.4f}, reference '
              f'{"dropped" if reference is None else "on"}')

    # 4. checkpointing: the peft adapter stays a normal peft checkpoint, the reference path is
    #    saved next to it
    # transformer.save_pretrained('out/reference_lora')
    # control.save_pretrained('out/reference_lora')
    print('done')


if __name__ == '__main__':
    main()

import sys
import types
from types import SimpleNamespace

import torch

from mopadi_genomic_crossattn.model import MultiStageCrossAttentionUNetWrapper


def test_cond_emb_layers_patched():
    # Inject a fake mopadi.model.blocks.ResBlock so the wrapper's _patch_block
    # logic recognizes and patches our dummy ResBlock instances.
    mod = types.ModuleType("mopadi.model.blocks")

    class ResBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # minimal config expected by _patch_block
            self.conf = SimpleNamespace()
            # provide both attributes that code may inspect
            self.conf.out_channels = 8
            self.conf.channels = 8
            self.conf.two_cond = False
            # placeholder for cond_emb_layers that wrapper will replace
            self.cond_emb_layers = None

        def forward(self, x, **kwargs):
            return x

    mod.ResBlock = ResBlock
    sys.modules["mopadi.model.blocks"] = mod

    # Build a minimal base_unet-like object with attributes the wrapper uses
    class FakeUNet:
        def __init__(self):
            self.conf = SimpleNamespace()
            self.conf.channel_mult = (1, 2)
            self.conf.model_channels = 8
            self.conf.input_channel_mult = (1, 2)

            # simple lists of ResBlock instances
            self.input_blocks = [ResBlock(), ResBlock()]
            self.middle_block = ResBlock()
            self.output_blocks = [ResBlock(), ResBlock()]

            # numeric block counts (not used in __init__, but safe to include)
            self.input_num_blocks = [len(self.input_blocks)]
            self.output_num_blocks = [len(self.output_blocks)]

            # out projection expected by wrapper (callable)
            self.out = lambda h: h

    base = FakeUNet()

    wrapper = MultiStageCrossAttentionUNetWrapper(base_unet=base, cond_dim=512, pool_size=4, heads=2, dim_head=16)

    # After wrapping, each ResBlock should have been set to two_cond=True
    # and should contain a cond_emb_layers mapping to 2*out_ch
    for block in base.input_blocks + [base.middle_block] + base.output_blocks:
        assert hasattr(block.conf, "two_cond") and block.conf.two_cond is True
        assert hasattr(block, "cond_emb_layers") and block.cond_emb_layers is not None
        # cond_emb_layers should be a small Sequential with a Linear producing 2*out_channels
        seq = block.cond_emb_layers
        assert isinstance(seq, torch.nn.Sequential)
        lin = None
        for layer in seq:
            if isinstance(layer, torch.nn.Linear):
                lin = layer
                break
        assert lin is not None
        assert lin.out_features == 2 * int(block.conf.out_channels)

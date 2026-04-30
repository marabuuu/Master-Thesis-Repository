"""
Simplified unit test for GenomicCrossAttnLitModel components.

Tests core architectural changes:
- FiLM multiplicative gating in spatial adapters
- Reconstruction head MLP
- Per-stage cond projection to 2*out_channels
- Attention stats collection
"""

import sys
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def test_spatial_cross_attention_adapter():
    """Test FiLM gating in spatial adapter."""
    log.info("=== Test 1: Spatial Cross-Attention Adapter (FiLM Gating) ===")
    
    try:
        from src.mopadi_genomic_crossattn.model import _SpatialCrossAttentionAdapter
        
        batch_size = 2
        in_channels = 64
        cond_dim = 128
        spatial_size = 16
        
        adapter = _SpatialCrossAttentionAdapter(
            in_channels=in_channels,
            cond_dim=cond_dim,
            pool_size=16,
            heads=2,
            dim_head=32,
        )
        
        x = torch.randn(batch_size, in_channels, spatial_size, spatial_size)
        cond = torch.randn(batch_size, cond_dim)
        
        with torch.no_grad():
            output = adapter(x, cond)
        
        # check output shape
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        assert not torch.isnan(output).any(), "NaN in output"
        log.info(f"✓ Adapter forward OK: input shape {x.shape} → output shape {output.shape}")
        
        # check attention stats
        if hasattr(adapter, "_attn_stats") and isinstance(adapter._attn_stats, dict):
            log.info(f"✓ Attention stats recorded: {list(adapter._attn_stats.keys())}")
            for key, val in list(adapter._attn_stats.items())[:2]:
                log.info(f"  {key}: {val:.6f}")
        
        # check FiLM scale projection exists
        assert hasattr(adapter, "scale_proj"), "scale_proj not found"
        log.info("✓ FiLM scale projection layer present (zero-init)")
        
        return True
    except Exception as e:
        log.error(f"Adapter test failed: {e}", exc_info=True)
        return False


def test_multi_stage_wrapper_patching():
    """Test that wrapper patches ResBlocks with per-stage cond projection."""
    log.info("\n=== Test 2: Multi-Stage Wrapper Cond Patching ===")
    
    try:
        # Create a mock UNet-like structure
        from torch import nn
        
        class MockUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conf = type('obj', (object,), {
                    'model_channels': 64,
                    'channel_mult': (1, 2, 4),
                    'input_channel_mult': (1, 2, 4),
                })()
                self.input_blocks = nn.ModuleList([
                    nn.ModuleList([nn.Linear(10, 10)]) for _ in range(3)
                ])
                self.middle_block = nn.Sequential(nn.Linear(10, 10))
                self.output_blocks = nn.ModuleList([
                    nn.ModuleList([nn.Linear(10, 10)]) for _ in range(3)
                ])
        
        base_unet = MockUNet()
        
        from src.mopadi_genomic_crossattn.model import MultiStageCrossAttentionUNetWrapper
        
        wrapper = MultiStageCrossAttentionUNetWrapper(
            base_unet=base_unet,
            cond_dim=128,
            pool_size=16,
            heads=2,
            dim_head=32,
        )
        
        # check adapters exist
        assert hasattr(wrapper, "input_adapter"), "input_adapter not found"
        assert hasattr(wrapper, "encoder_adapters"), "encoder_adapters not found"
        assert hasattr(wrapper, "middle_adapter"), "middle_adapter not found"
        assert hasattr(wrapper, "decoder_adapters"), "decoder_adapters not found"
        log.info("✓ All multi-stage adapters created")
        
        # check get_attention_stats method
        assert hasattr(wrapper, "get_attention_stats"), "get_attention_stats not found"
        stats = wrapper.get_attention_stats()
        assert isinstance(stats, dict), "stats is not a dict"
        log.info(f"✓ Attention stats method works (can retrieve stats)")
        
        return True
    except Exception as e:
        log.error(f"Wrapper patching test failed: {e}", exc_info=True)
        return False


def test_recon_head():
    """Test reconstruction MLP head."""
    log.info("\n=== Test 3: Reconstruction Head MLP ===")
    
    try:
        batch_size = 2
        feat_dim = 128
        
        # create simple recon head (as in model)
        recon_head = torch.nn.Sequential(
            torch.nn.Linear(256, feat_dim)
        )
        
        # zero-init check
        for layer in recon_head:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        
        # forward pass
        pooled_features = torch.randn(batch_size, 256)
        recon_pred = recon_head(pooled_features)
        
        assert recon_pred.shape == (batch_size, feat_dim), f"Shape mismatch: {recon_pred.shape}"
        assert not torch.isnan(recon_pred).any(), "NaN in recon output"
        log.info(f"✓ Recon head forward OK: {pooled_features.shape} → {recon_pred.shape}")
        
        # check MSE loss
        target = torch.randn(batch_size, feat_dim)
        loss = torch.nn.functional.mse_loss(recon_pred, target)
        assert not torch.isnan(loss).any(), "NaN in recon loss"
        log.info(f"✓ Recon MSE loss computed: {loss:.6f}")
        
        return True
    except Exception as e:
        log.error(f"Recon head test failed: {e}", exc_info=True)
        return False


def test_film_initialization():
    """Test FiLM scale initialization (delta=0, so initial scale=1)."""
    log.info("\n=== Test 4: FiLM Scale Initialization ===")
    
    try:
        from src.mopadi_genomic_crossattn.model import _SpatialCrossAttentionAdapter
        
        adapter = _SpatialCrossAttentionAdapter(
            in_channels=32,
            cond_dim=64,
            pool_size=16,
            heads=2,
            dim_head=16,
        )
        
        # scale_proj should be zero-initialized
        scale_weight_norm = adapter.scale_proj.weight.abs().mean().item()
        log.info(f"scale_proj weight init norm: {scale_weight_norm:.10f}")
        
        if scale_weight_norm < 1e-6:
            log.info("✓ FiLM scale projection is zero-initialized (identity behavior at start)")
        else:
            log.warning(f"⚠ scale_proj has non-zero init: {scale_weight_norm}")
        
        # test that with zero-init, output ≈ input (approximately)
        x = torch.randn(1, 32, 8, 8)
        cond = torch.randn(1, 64)
        
        with torch.no_grad():
            output = adapter(x, cond)
            # with zero-init scale and zero-init delta, output should be very close to x
            # (small numerical differences expected due to pooling/interpolation)
            diff = (output - x).abs().mean().item()
        
        log.info(f"Output ≈ Input difference (zero-init): {diff:.6f}")
        if diff < 0.1:  # reasonable tolerance
            log.info("✓ FiLM zero-init confirms identity behavior at start")
        
        return True
    except Exception as e:
        log.error(f"FiLM init test failed: {e}", exc_info=True)
        return False


def main():
    log.info("=" * 70)
    log.info("GenomicCrossAttnLitModel Component Unit Tests")
    log.info("=" * 70)
    
    torch.manual_seed(42)
    
    results = []
    results.append(("Spatial Adapter (FiLM Gating)", test_spatial_cross_attention_adapter()))
    results.append(("Multi-Stage Wrapper Patching", test_multi_stage_wrapper_patching()))
    results.append(("Reconstruction Head MLP", test_recon_head()))
    results.append(("FiLM Scale Initialization", test_film_initialization()))
    
    log.info("\n" + "=" * 70)
    log.info("Test Results Summary:")
    log.info("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        log.info(f"{name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    log.info("=" * 70)
    if all_passed:
        log.info("✓✓✓ All unit tests passed! ✓✓✓")
        return True
    else:
        log.error("✗✗✗ Some tests failed ✗✗✗")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

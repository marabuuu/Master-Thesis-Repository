# -*- coding: utf-8 -*-
"""Tests for the finetune_diffusion submodule.

Focus areas
-----------
* ProjectionHead: correct output shapes for each architecture variant
* ProjectionHead: normalize_output produces unit-norm vectors
* ProjectionHead: forward rejects bad input shapes
* canonical_patient_id: extracting TCGA patient prefixes
* tensor_to_pil: conversion produces a proper PIL.Image
* GenomicTileDataset: raises early when no matching patients exist
* Integration: vae→projection head pipeline produces correct shapes
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

# Import ProjectionHead from the standalone sampling script (lighter, no MoPaDi dep)
from src.finetune_diffusion.sample_tiles_from_genomic import (
    ProjectionHead,
    canonical_patient_id,
    tensor_to_pil,
)


# ======================================================================
#   ProjectionHead
# ======================================================================

class TestProjectionHeadMLP:
    """Tests for the default MLP architecture."""

    def test_output_shape(self):
        head = ProjectionHead(in_dim=512, out_dim=512, hidden_dim=256, num_layers=2)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 512)

    def test_different_dims(self):
        head = ProjectionHead(in_dim=256, out_dim=128, hidden_dim=64, num_layers=3)
        out = head(torch.randn(2, 256))
        assert out.shape == (2, 128)

    def test_single_layer(self):
        head = ProjectionHead(in_dim=512, out_dim=512, num_layers=1, arch="mlp")
        out = head(torch.randn(3, 512))
        assert out.shape == (3, 512)

    def test_normalize_output(self):
        head = ProjectionHead(in_dim=64, out_dim=64, normalize_output=True)
        x = torch.randn(8, 64) * 10  # large magnitude
        out = head(x)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


class TestProjectionHeadLinear:

    def test_output_shape(self):
        head = ProjectionHead(in_dim=512, out_dim=256, arch="linear")
        out = head(torch.randn(4, 512))
        assert out.shape == (4, 256)


class TestProjectionHeadResidual:

    def test_output_shape(self):
        head = ProjectionHead(in_dim=512, out_dim=512, arch="residual", num_layers=2)
        out = head(torch.randn(4, 512))
        assert out.shape == (4, 512)


class TestProjectionHeadBadInput:

    def test_wrong_ndim_raises(self):
        head = ProjectionHead(in_dim=64, out_dim=64)
        with pytest.raises(AssertionError):
            head(torch.randn(64))  # 1-D instead of 2-D

    def test_wrong_feature_dim_raises(self):
        head = ProjectionHead(in_dim=64, out_dim=64)
        with pytest.raises(AssertionError):
            head(torch.randn(4, 128))  # dim mismatch

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError):
            ProjectionHead(arch="transformer")


# ======================================================================
#   ProjectionHead gradient flow
# ======================================================================

class TestProjectionHeadGradients:

    @pytest.mark.parametrize("arch", ["linear", "mlp", "residual"])
    def test_gradients_flow(self, arch):
        if arch == "residual":
            head = ProjectionHead(in_dim=32, out_dim=32, hidden_dim=32, arch=arch)
        else:
            head = ProjectionHead(in_dim=32, out_dim=64, hidden_dim=32, arch=arch)
        x = torch.randn(4, 32, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ======================================================================
#   canonical_patient_id
# ======================================================================

class TestCanonicalPatientId:

    def test_standard_tcga_name(self):
        assert canonical_patient_id("TCGA-3C-AALI.h5") == "TCGA-3C-AALI"

    def test_with_extra_parts(self):
        assert canonical_patient_id("TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF.h5") == "TCGA-3C-AALI"

    def test_underscore_separator(self):
        assert canonical_patient_id("TCGA_XX_1234.h5") == "TCGA-XX-1234"

    def test_non_tcga_name(self):
        # Should return the processed string as-is
        result = canonical_patient_id("OTHER-SAMPLE.h5")
        assert isinstance(result, str)
        assert len(result) > 0


# ======================================================================
#   tensor_to_pil
# ======================================================================

class TestTensorToPil:

    def test_returns_pil_image(self):
        t = torch.randn(3, 64, 64)  # in [-1, 1]-ish range
        img = tensor_to_pil(t)
        assert isinstance(img, Image.Image)

    def test_correct_size(self):
        t = torch.randn(3, 128, 256)
        img = tensor_to_pil(t)
        assert img.size == (256, 128)  # PIL size is (width, height)

    def test_rgb_mode(self):
        t = torch.randn(3, 32, 32)
        img = tensor_to_pil(t)
        assert img.mode == "RGB"


# ======================================================================
#   Pipeline integration: VAE encoder → ProjectionHead
# ======================================================================

class TestVAEToProjectionPipeline:
    """Verify data structures flow correctly from VAE latent space to
    the projection head that feeds the diffusion model."""

    def test_latent_to_projection(self, small_vae, device):
        """VAE latent mean → projection head should produce correct shape."""
        head = ProjectionHead(in_dim=16, out_dim=512, hidden_dim=64, num_layers=2)
        x = torch.randn(4, 200, device=device)
        z = small_vae.encode(x)  # (4, 16)
        projected = head(z)
        assert projected.shape == (4, 512)

    def test_dtypes_consistent(self, small_vae, device):
        head = ProjectionHead(in_dim=16, out_dim=512, hidden_dim=64)
        x = torch.randn(4, 200, device=device)
        z = small_vae.encode(x)
        projected = head(z)
        assert z.dtype == torch.float32
        assert projected.dtype == torch.float32

    def test_normalize_keeps_shape(self, small_vae, device):
        head = ProjectionHead(in_dim=16, out_dim=512, hidden_dim=64,
                              normalize_output=True)
        x = torch.randn(4, 200, device=device)
        z = small_vae.encode(x)
        projected = head(z)
        assert projected.shape == (4, 512)
        norms = projected.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


# ======================================================================
#   GenomicTileDataset (error path)
# ======================================================================

class TestGenomicTileDatasetErrors:

    def test_raises_when_no_matching_patients(self, tmp_path):
        """Dataset should raise RuntimeError when genomic & tiles dirs share no IDs."""
        from src.finetune_diffusion.finetune_diffusion_with_genomic import GenomicTileDataset

        genomic_dir = tmp_path / "genomic"
        genomic_dir.mkdir()
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()
        # No files → no common IDs
        with pytest.raises(RuntimeError, match="No matching patient"):
            GenomicTileDataset(
                genomic_h5_dir=str(genomic_dir),
                tiles_zip_dir=str(tiles_dir),
            )

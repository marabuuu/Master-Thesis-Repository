"""Smoke tests for the finetune_diffusion submodule.

These tests intentionally cover only a few high-value import and shape
checks. The heavy training behaviors are exercised elsewhere.
"""

import pytest
import torch

from src.finetune_diffusion.sample_tiles_from_genomic import ProjectionHead, canonical_patient_id


def test_projection_head_smoke():
    head = ProjectionHead(in_dim=512, out_dim=512, hidden_dim=256, num_layers=2)
    out = head(torch.randn(4, 512))
    assert out.shape == (4, 512)


def test_projection_head_backward_smoke():
    head = ProjectionHead(in_dim=32, out_dim=64, hidden_dim=32, num_layers=2)
    x = torch.randn(4, 32, requires_grad=True)
    out = head(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_canonical_patient_id_smoke():
    assert canonical_patient_id("TCGA-3C-AALI.h5") == "TCGA-3C-AALI"
    assert canonical_patient_id("TCGA_XX_1234.h5") == "TCGA-XX-1234"


def test_genomic_tile_dataset_requires_matching_inputs(tmp_path):
    from src.finetune_diffusion.finetune_diffusion_with_genomic import GenomicTileDataset

    genomic_dir = tmp_path / "genomic"
    tiles_dir = tmp_path / "tiles"
    genomic_dir.mkdir()
    tiles_dir.mkdir()

    with pytest.raises(RuntimeError, match="No matching patients"):
        GenomicTileDataset(str(genomic_dir), str(tiles_dir))

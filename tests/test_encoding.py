# -*- coding: utf-8 -*-
"""Tests for the encoding submodule.

Focus areas
-----------
* VAE forward pass produces correct tensor shapes (x_hat, z)
* Encoder outputs (mean, log_var) are the right shape / type
* Decoder output matches expected output dimension
* Loss function returns scalar tensors with expected components
* MMD / kernel computations return correct types
* Config dataclasses round-trip (to_dict / from_dict / YAML save-load)
* FullyConnectedLayer builds with different activation / norm options
* `make_unique_ids` and `save_h5` helper functions work correctly
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from src.encoding.architecture import (
    ProbabilisticEncoder,
    ProbabilisticDecoder,
    VAE,
    FullyConnectedLayer,
    compute_mmd,
    compute_kernel,
    MMDLoss,
)
from src.encoding.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    OutputConfig,
    EncodingConfig,
    get_default_config,
)
from src.encoding.train import make_unique_ids, save_h5


# ======================================================================
#   FullyConnectedLayer
# ======================================================================

class TestFullyConnectedLayer:

    def test_output_shape(self):
        layer = FullyConnectedLayer(100, 50)
        x = torch.randn(8, 100)
        out = layer(x)
        assert out.shape == (8, 50)

    def test_no_activation(self):
        layer = FullyConnectedLayer(100, 50, activation=False)
        x = torch.randn(4, 100)
        out = layer(x)
        assert out.shape == (4, 50)

    def test_no_normalization(self):
        layer = FullyConnectedLayer(100, 50, normalization=False)
        x = torch.randn(4, 100)
        out = layer(x)
        assert out.shape == (4, 50)

    def test_invalid_activation_name_raises(self):
        with pytest.raises(NotImplementedError):
            FullyConnectedLayer(10, 5, activation_name="nonexistent_act")

    @pytest.mark.parametrize("act_name", ["relu", "sigmoid", "LeakyReLU", "tanh"])
    def test_various_activations(self, act_name):
        layer = FullyConnectedLayer(10, 5, activation_name=act_name)
        out = layer(torch.randn(2, 10))
        assert out.shape == (2, 5)


# ======================================================================
#   Encoder
# ======================================================================

class TestProbabilisticEncoder:

    def test_output_types(self):
        enc = ProbabilisticEncoder(200, [64, 32], 16)
        x = torch.randn(8, 200)
        mean, log_var = enc(x)
        assert isinstance(mean, torch.Tensor)
        assert isinstance(log_var, torch.Tensor)

    def test_output_shapes(self):
        enc = ProbabilisticEncoder(200, [64, 32], 16)
        x = torch.randn(8, 200)
        mean, log_var = enc(x)
        assert mean.shape == (8, 16)
        assert log_var.shape == (8, 16)

    def test_single_hidden_layer(self):
        enc = ProbabilisticEncoder(100, [64], 8)
        mean, log_var = enc(torch.randn(4, 100))
        assert mean.shape == (4, 8)

    def test_dropout_param(self):
        enc = ProbabilisticEncoder(100, [64, 32], 8, dropout=0.5)
        mean, _ = enc(torch.randn(4, 100))
        assert mean.shape == (4, 8)


# ======================================================================
#   Decoder
# ======================================================================

class TestProbabilisticDecoder:

    def test_output_shape(self):
        dec = ProbabilisticDecoder(16, [32, 64], 200)
        z = torch.randn(8, 16)
        x_hat = dec(z)
        assert x_hat.shape == (8, 200)

    def test_single_hidden_layer(self):
        dec = ProbabilisticDecoder(8, [64], 100)
        x_hat = dec(torch.randn(4, 8))
        assert x_hat.shape == (4, 100)


# ======================================================================
#   VAE
# ======================================================================

class TestVAE:

    def test_forward_shapes(self, small_vae, device):
        x = torch.randn(8, 200, device=device)
        x_hat, z = small_vae(x)
        assert x_hat.shape == x.shape, "Reconstruction should match input shape"
        assert z.shape == (8, 16), "Latent dim should be 16"

    def test_loss_returns_three_tensors(self, small_vae, device):
        x = torch.randn(8, 200, device=device)
        total, recon, mmd = small_vae.loss_components(x, beta=1.0)
        for t in (total, recon, mmd):
            assert isinstance(t, torch.Tensor)
            assert t.dim() == 0, "Loss should be a scalar"

    def test_loss_is_finite(self, small_vae, device):
        x = torch.randn(8, 200, device=device)
        total, recon, mmd = small_vae.loss_components(x, beta=1.0)
        assert torch.isfinite(total), "Total loss should be finite"
        assert torch.isfinite(recon), "Recon loss should be finite"
        assert torch.isfinite(mmd), "MMD loss should be finite"

    def test_loss_beta_zero_removes_mmd(self, small_vae, device):
        """With beta=0, the total loss should equal reconstruction loss."""
        x = torch.randn(8, 200, device=device)
        total, recon, mmd = small_vae.loss_components(x, beta=0.0)
        assert torch.allclose(total, recon, atol=1e-6)

    def test_encode_returns_mean_only(self, small_vae, device):
        x = torch.randn(8, 200, device=device)
        z = small_vae.encode(x)
        assert isinstance(z, torch.Tensor)
        assert z.shape == (8, 16)

    def test_reparameterization_shape(self, small_vae, device):
        mean = torch.randn(4, 16, device=device)
        std = torch.ones(4, 16, device=device)
        z = small_vae.reparameterization(mean, std)
        assert z.shape == mean.shape

    def test_gradient_flow_through_vae(self, small_vae, device):
        x = torch.randn(8, 200, device=device)
        loss = small_vae.loss(x, beta=1.0)
        loss.backward()
        # At least one parameter should have a gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in small_vae.parameters())
        assert has_grad, "Gradients should flow through VAE"


# ======================================================================
#   MMD / Kernel
# ======================================================================

class TestMMD:

    def test_compute_kernel_shape(self):
        x = torch.randn(10, 16)
        y = torch.randn(8, 16)
        k = compute_kernel(x, y)
        assert k.shape == (10, 8)

    def test_compute_kernel_values_positive(self):
        x = torch.randn(5, 4)
        y = torch.randn(5, 4)
        k = compute_kernel(x, y)
        assert (k >= 0).all(), "RBF kernel values should be non-negative"
        assert (k <= 1).all(), "RBF kernel values should be ≤ 1"

    def test_compute_mmd_same_distribution(self):
        """MMD between identical samples should be near zero."""
        x = torch.randn(50, 16)
        mmd = compute_mmd(x, x)
        assert mmd.item() < 0.01

    def test_compute_mmd_different_distributions(self):
        """MMD between very different distributions should be larger."""
        x = torch.randn(100, 16)
        y = torch.randn(100, 16) + 10.0  # shifted
        mmd = compute_mmd(x, y)
        assert mmd.item() > 0.1

    def test_mmd_is_scalar(self):
        x = torch.randn(10, 8)
        y = torch.randn(10, 8)
        mmd = compute_mmd(x, y)
        assert mmd.dim() == 0

    def test_mmd_loss_module(self):
        loss_fn = MMDLoss()
        x = torch.randn(10, 8)
        y = torch.randn(10, 8)
        loss = loss_fn(x, y)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


# ======================================================================
#   Config dataclasses
# ======================================================================

class TestEncodingConfig:

    def test_default_config(self):
        cfg = get_default_config()
        assert isinstance(cfg, EncodingConfig)
        assert cfg.model.latent_dim == 512
        assert cfg.training.epochs == 100

    def test_to_dict_roundtrip(self):
        cfg = get_default_config()
        d = cfg.to_dict()
        cfg2 = EncodingConfig.from_dict(d)
        assert cfg2.model.latent_dim == cfg.model.latent_dim
        assert cfg2.training.batch_size == cfg.training.batch_size

    def test_yaml_save_load(self, tmp_path):
        cfg = get_default_config()
        cfg.model.latent_dim = 256
        path = str(tmp_path / "config.yml")
        cfg.save(path)
        cfg2 = EncodingConfig.load(path)
        assert cfg2.model.latent_dim == 256

    def test_custom_model_config(self):
        mc = ModelConfig(latent_dim=128, hidden_dim=[1024])
        assert mc.latent_dim == 128
        assert mc.hidden_dim == [1024]

    def test_custom_training_config(self):
        tc = TrainingConfig(epochs=50, batch_size=32, learning_rate=5e-4)
        assert tc.epochs == 50
        assert tc.batch_size == 32


# ======================================================================
#   Training helpers
# ======================================================================

class TestTrainingHelpers:

    def test_make_unique_ids_no_duplicates(self):
        ids = ["A", "B", "C"]
        new_ids, orig_map = make_unique_ids(ids)
        assert new_ids == ["A", "B", "C"]
        assert orig_map["A"] == "A"

    def test_make_unique_ids_with_duplicates(self):
        ids = ["A", "A", "B"]
        new_ids, orig_map = make_unique_ids(ids)
        assert new_ids[0] == "A"
        assert new_ids[1] == "A-DX2"
        assert new_ids[2] == "B"
        assert orig_map["A-DX2"] == "A"

    def test_save_h5(self, tmp_path):
        arr = np.random.randn(1, 16).astype(np.float32)
        path = save_h5("test_patient", arr, str(tmp_path))
        assert os.path.exists(path)
        import h5py
        with h5py.File(path, "r") as f:
            assert "feats" in f
            loaded = f["feats"][:]
            assert loaded.shape == (1, 16)
            np.testing.assert_array_almost_equal(loaded, arr)

    def test_save_h5_1d_input(self, tmp_path):
        """1D array should be reshaped to (1, dim) in the H5 file."""
        arr = np.random.randn(16).astype(np.float32)
        path = save_h5("patient1d", arr, str(tmp_path))
        import h5py
        with h5py.File(path, "r") as f:
            assert f["feats"].shape == (1, 16)

# -*- coding: utf-8 -*-
"""Tests for the statistics submodule.

Focus areas
-----------
* FID computation: correct formula on known distributions
* compute_statistics: returns correct shapes for mean / covariance
* Log-file parsing: extracts epoch-loss tuples from different formats
* Checkpoint parsing: round-trip with known .pt files
* plot_training_curves: returns a matplotlib Figure (no display)
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# guard heavy imports
torch = pytest.importorskip("torch")
plt = pytest.importorskip("matplotlib.pyplot")
scipy_linalg = pytest.importorskip("scipy.linalg")

from src.statistics.fid_score import (
    compute_fid,
    compute_statistics,
)
from src.statistics.training_curves import (
    parse_log_file,
    parse_checkpoints,
    plot_training_curves,
)


# ======================================================================
#   compute_statistics
# ======================================================================

class TestComputeStatistics:

    def test_returns_mean_and_covariance(self):
        features = np.random.randn(100, 2048)
        mu, sigma = compute_statistics(features)
        assert mu.shape == (2048,)
        assert sigma.shape == (2048, 2048)

    def test_mean_is_close_for_standard_normal(self):
        np.random.seed(42)
        features = np.random.randn(5000, 64)
        mu, _ = compute_statistics(features)
        np.testing.assert_array_almost_equal(mu, 0.0, decimal=1)

    def test_covariance_approximately_identity(self):
        np.random.seed(42)
        features = np.random.randn(5000, 16)
        _, sigma = compute_statistics(features)
        np.testing.assert_array_almost_equal(sigma, np.eye(16), decimal=1)

    def test_single_feature_dim(self):
        features = np.random.randn(50, 1)
        mu, sigma = compute_statistics(features)
        assert mu.shape == (1,)
        # sigma is 2D even for 1-D features
        assert sigma.ndim == 2


# ======================================================================
#   compute_fid
# ======================================================================

class TestComputeFID:

    def test_fid_identical_distributions_is_zero(self):
        np.random.seed(0)
        features = np.random.randn(500, 64)
        mu, sigma = compute_statistics(features)
        fid = compute_fid(mu, sigma, mu, sigma)
        assert abs(fid) < 1e-4, "FID of identical distributions should be ≈ 0"

    def test_fid_shifted_distribution(self):
        np.random.seed(0)
        f1 = np.random.randn(500, 64)
        f2 = np.random.randn(500, 64) + 5.0  # shift
        mu1, s1 = compute_statistics(f1)
        mu2, s2 = compute_statistics(f2)
        fid = compute_fid(mu1, s1, mu2, s2)
        assert fid > 0, "Shifted distributions should have positive FID"

    def test_fid_is_float(self):
        f = np.random.randn(100, 16)
        mu, sigma = compute_statistics(f)
        fid = compute_fid(mu, sigma, mu, sigma)
        assert isinstance(fid, float)

    def test_fid_symmetric(self):
        """FID(A, B) should equal FID(B, A)."""
        np.random.seed(1)
        f1 = np.random.randn(200, 32)
        f2 = np.random.randn(200, 32) + 2.0
        mu1, s1 = compute_statistics(f1)
        mu2, s2 = compute_statistics(f2)
        fid_ab = compute_fid(mu1, s1, mu2, s2)
        fid_ba = compute_fid(mu2, s2, mu1, s1)
        assert abs(fid_ab - fid_ba) < 1e-2

    def test_fid_non_negative(self):
        """FID should be non-negative for realistic inputs."""
        np.random.seed(2)
        f1 = np.random.randn(300, 32)
        f2 = np.random.randn(300, 32) + 1.0
        mu1, s1 = compute_statistics(f1)
        mu2, s2 = compute_statistics(f2)
        fid = compute_fid(mu1, s1, mu2, s2)
        assert fid >= -1e-6, "FID should be non-negative"

    def test_larger_shift_gives_higher_fid(self):
        """A bigger distribution shift should produce a larger FID."""
        np.random.seed(3)
        base = np.random.randn(300, 32)
        shifted_small = base + 1.0
        shifted_big = base + 10.0
        mu0, s0 = compute_statistics(base)
        mu_s, s_s = compute_statistics(shifted_small)
        mu_b, s_b = compute_statistics(shifted_big)
        fid_small = compute_fid(mu0, s0, mu_s, s_s)
        fid_big = compute_fid(mu0, s0, mu_b, s_b)
        assert fid_big > fid_small


# ======================================================================
#   parse_log_file
# ======================================================================

class TestParseLogFile:

    def _write_log(self, tmp_path, content: str) -> Path:
        path = tmp_path / "train.log"
        path.write_text(content)
        return path

    def test_parses_epoch_loss(self, tmp_path):
        log_content = (
            "Epoch 1/10 | Time: 5.2s | Loss: 0.500000\n"
            "Epoch 2/10 | Time: 5.1s | Loss: 0.300000\n"
            "Epoch 3/10 | Time: 5.0s | Loss: 0.200000\n"
        )
        path = self._write_log(tmp_path, log_content)
        history = parse_log_file(path)
        assert "loss" in history
        assert len(history["loss"]) == 3
        epochs, losses = zip(*history["loss"])
        assert list(epochs) == [1, 2, 3]
        assert abs(losses[0] - 0.5) < 1e-6

    def test_parses_component_losses(self, tmp_path):
        log_content = (
            "Epoch 1/50 | total=0.026, mean=0.012, var=0.008, diversity=0.006\n"
            "Epoch 2/50 | total=0.020, mean=0.010, var=0.006, diversity=0.004\n"
        )
        path = self._write_log(tmp_path, log_content)
        history = parse_log_file(path)
        assert "loss" in history
        assert "mean" in history
        assert "var" in history
        assert "diversity" in history

    def test_empty_log_returns_empty(self, tmp_path):
        path = self._write_log(tmp_path, "no relevant data here\n")
        history = parse_log_file(path)
        assert isinstance(history, dict)
        assert len(history) == 0 or all(len(v) == 0 for v in history.values())

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_log_file(tmp_path / "nonexistent.log")


# ======================================================================
#   parse_checkpoints
# ======================================================================

class TestParseCheckpoints:

    def test_parses_epoch_and_loss(self, tmp_path):
        for i in range(3):
            ckpt = {"epoch": i + 1, "loss": 0.5 - i * 0.1}
            torch.save(ckpt, tmp_path / f"epoch{i + 1:03d}.pt")
        history = parse_checkpoints(tmp_path)
        assert "loss" in history
        assert len(history["loss"]) == 3
        # sorted by epoch
        epochs = [e for e, _ in history["loss"]]
        assert epochs == sorted(epochs)

    def test_no_pt_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_checkpoints(tmp_path)

    def test_missing_epoch_in_ckpt_still_works(self, tmp_path):
        """If epoch is missing but filename has pattern, we still extract."""
        ckpt = {"loss": 0.42}
        torch.save(ckpt, tmp_path / "epoch005.pt")
        history = parse_checkpoints(tmp_path)
        assert "loss" in history
        epochs = [e for e, _ in history["loss"]]
        assert 5 in epochs


# ======================================================================
#   plot_training_curves
# ======================================================================

class TestPlotTrainingCurves:

    def test_returns_figure(self):
        history = {"loss": [(1, 0.5), (2, 0.3), (3, 0.2)]}
        fig = plot_training_curves([history], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_runs(self):
        h1 = {"loss": [(1, 0.5), (2, 0.3)]}
        h2 = {"loss": [(1, 0.4), (2, 0.2)]}
        fig = plot_training_curves([h1, h2], labels=["Run 1", "Run 2"], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        history = {"loss": [(1, 0.5), (2, 0.3)]}
        out_path = str(tmp_path / "curves.png")
        fig = plot_training_curves([history], output_path=out_path, show=False)
        assert os.path.exists(out_path)
        plt.close(fig)

    def test_multiple_metrics(self):
        history = {
            "loss": [(1, 0.5), (2, 0.3)],
            "mean": [(1, 0.2), (2, 0.1)],
            "var": [(1, 0.1), (2, 0.08)],
        }
        fig = plot_training_curves([history], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_history(self):
        """Supplying an empty history dict should not crash."""
        fig = plot_training_curves([{}], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

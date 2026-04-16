# -*- coding: utf-8 -*-
"""Tests for the statistics submodule.

Focus areas
-----------
* FID computation: correct formula on known distributions
* compute_statistics: returns correct shapes for mean / covariance
* Log-file parsing: extracts epoch-loss tuples from current run formats
* Experiment parsing: merges stdout/stderr logs into a single run dict
* plotting: current training_plots API returns matplotlib figures
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
    parse_lightning_log,
    parse_stderr_log,
    parse_experiment_dir,
)
from src.visualization.training_plots import plot_loss_curves, plot_training_summary


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
        # Single-feature covariance may collapse to a scalar / 0-D array.
        assert sigma.ndim in (0, 2)


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
#   parse_lightning_log / parse_stderr_log
# ======================================================================

class TestParseLogFile:

    def _write_log(self, tmp_path, content: str) -> Path:
        path = tmp_path / "train.log"
        path.write_text(content)
        return path

    def test_parses_epoch_loss(self, tmp_path):
        log_content = (
            "Epoch 0: 100%|██████████| 10/10 [00:01<00:00, loss_step=0.500000, loss_epoch=0.500000, val_loss=0.600000]\n"
            "Epoch 1: 100%|██████████| 10/10 [00:01<00:00, loss_step=0.300000, loss_epoch=0.300000, val_loss=0.400000]\n"
            "Epoch 2: 100%|██████████| 10/10 [00:01<00:00, loss_step=0.200000, loss_epoch=0.200000, val_loss=0.300000]\n"
        )
        path = self._write_log(tmp_path, log_content)
        history = parse_lightning_log(path)
        assert history["epochs"] == [0, 1, 2]
        assert history["loss_epoch"] == [0.5, 0.3, 0.2]
        assert history["val_loss"] == [0.6, 0.4, 0.3]
        assert history["loss_step"] == [0.5, 0.3, 0.2]

    def test_parses_component_losses(self, tmp_path):
        log_content = (
            "Epoch 0: 100%|██| 10/10 [00:01<00:00, loss_step=0.026, loss_epoch=0.026, val_loss=0.030]\n"
            "Epoch 1: 100%|██| 10/10 [00:01<00:00, loss_step=0.020, loss_epoch=0.020, val_loss=0.025]\n"
        )
        path = self._write_log(tmp_path, log_content)
        history = parse_lightning_log(path)
        assert history["loss_epoch"] == [0.026, 0.02]
        assert history["val_loss"] == [0.03, 0.025]

    def test_empty_log_returns_empty(self, tmp_path):
        path = self._write_log(tmp_path, "no relevant data here\n")
        history = parse_lightning_log(path)
        assert isinstance(history, dict)
        assert len(history.get("loss_epoch", [])) == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_lightning_log(tmp_path / "nonexistent.log")


# ======================================================================
#   parse_experiment_dir / parse_stderr_log
# ======================================================================

class TestParseCheckpoints:

    def test_parse_experiment_dir_merges_logs(self, tmp_path):
        out_path = tmp_path / "run.out"
        err_path = tmp_path / "run.err"
        out_path.write_text(
            "Epoch 0: 100%|██| 10/10 [00:01<00:00, loss_step=0.50, loss_epoch=0.50, val_loss=0.60]\n"
            "Epoch 1: 100%|██| 10/10 [00:01<00:00, loss_step=0.30, loss_epoch=0.30, val_loss=0.40]\n"
        )
        err_path.write_text(
            "Metric val_loss improved. New best score: 0.60\n"
            "Metric val_loss improved. New best score: 0.40\n"
        )
        history = parse_experiment_dir(tmp_path)
        assert history["loss_epoch"] == [0.5, 0.3]
        assert history["val_loss"] == [0.6, 0.4]
        assert history["best_val_loss"] == 0.4
        assert history["best_val_epoch"] == 1


# ======================================================================
#   plot_training_curves
# ======================================================================

class TestPlotTrainingCurves:

    def test_returns_figure(self):
        history = {"epochs": [0, 1, 2], "loss_epoch": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.3]}
        fig = plot_loss_curves(history, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_runs(self):
        h1 = {"epochs": [0, 1], "loss_epoch": [0.5, 0.3], "val_loss": [0.6, 0.4]}
        h2 = {"epochs": [0, 1], "loss_epoch": [0.4, 0.2], "val_loss": [0.5, 0.35]}
        fig = plot_loss_curves([h1, h2], labels=["Run 1", "Run 2"], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path):
        history = {"epochs": [0, 1], "loss_epoch": [0.5, 0.3], "val_loss": [0.6, 0.4]}
        out_path = str(tmp_path / "curves.png")
        fig = plot_loss_curves(history, save_path=out_path, show=False)
        assert os.path.exists(out_path)
        plt.close(fig)

    def test_multiple_metrics(self):
        history = {"epochs": [0, 1], "loss_epoch": [0.5, 0.3], "val_loss": [0.6, 0.4], "loss_step": [0.5, 0.45, 0.3]}
        fig = plot_training_summary(history, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_history(self):
        """Supplying an empty history dict should not crash."""
        fig = plot_loss_curves([{}], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

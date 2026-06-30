# -*- coding: utf-8 -*-
"""Tests for the statistics/visualization submodules.

Focus areas
-----------
* plotting: current training_plots API returns matplotlib figures
"""

import os
from pathlib import Path

import numpy as np
import pytest

# guard heavy imports
plt = pytest.importorskip("matplotlib.pyplot")

from src.visualization.training_plots import plot_loss_curves, plot_training_summary


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

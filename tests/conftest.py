# -*- coding: utf-8 -*-
"""Shared fixtures for the test suite."""

import os
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# Ensure the workspace-local mopadi source tree is importable during tests.
_repo_root = Path(__file__).resolve().parents[1]
_mopadi_src = (_repo_root.parent / "mopadi" / "src").resolve()
if _mopadi_src.exists() and str(_mopadi_src) not in sys.path:
    sys.path.insert(0, str(_mopadi_src))


# ---------------------------------------------------------------------------
#   Gene-expression related fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_gene_expression_df():
    """A small DataFrame that mimics raw RNA-seq count data (samples × genes)."""
    np.random.seed(0)
    n_samples, n_genes = 30, 200
    counts = np.random.poisson(lam=5, size=(n_samples, n_genes)).astype(float)
    sample_ids = [f"TCGA-XX-{i:04d}" for i in range(n_samples)]
    gene_names = [f"GENE_{j}" for j in range(n_genes)]
    return pd.DataFrame(counts, index=sample_ids, columns=gene_names)


@pytest.fixture
def zscore_gene_expression_df(raw_gene_expression_df):
    """Gene-expression data that already looks z-scored (mean ≈ 0, var ≈ 1)."""
    from src.preprocessing.utils import preprocess_log1p_zscore

    return preprocess_log1p_zscore(raw_gene_expression_df)


@pytest.fixture
def gene_expression_csv(raw_gene_expression_df, tmp_path):
    """Write raw gene-expression data to a CSV and return the path."""
    csv_path = tmp_path / "gene_expression.csv"
    raw_gene_expression_df.to_csv(csv_path)
    return str(csv_path)


@pytest.fixture
def gene_expression_csv_with_label(raw_gene_expression_df, tmp_path):
    """CSV that includes a label column (e.g. PAM50 subtype)."""
    df = raw_gene_expression_df.copy()
    labels = np.random.choice(["Basal", "LumA", "LumB"], size=len(df))
    df.insert(0, "Majority_Subtype_mRNA", labels)
    csv_path = tmp_path / "gene_expression_labeled.csv"
    df.to_csv(csv_path)
    return str(csv_path)


# ---------------------------------------------------------------------------
#   Torch / model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Return a CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_vae(device):
    """A tiny VAE instance suitable for fast unit tests."""
    from src.encoding.architecture import ProbabilisticEncoder, ProbabilisticDecoder, VAE

    input_dim = 200
    hidden = [64, 32]
    latent_dim = 16

    enc = ProbabilisticEncoder(input_dim, hidden, latent_dim)
    dec = ProbabilisticDecoder(latent_dim, list(reversed(hidden)), input_dim)
    vae = VAE(enc, dec, device)
    return vae


# ---------------------------------------------------------------------------
#   Temporary directory helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_outdir(tmp_path):
    """A temporary output directory for tests that write files."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)

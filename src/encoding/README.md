# Genomic Feature Encoding

This module encodes high-dimensional genomic data (e.g., gene expression from CSV files) into compact latent feature vectors using a Variational Autoencoder (VAE) with MMD regularization.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        GENOMIC ENCODING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐      ┌───────────────────┐      ┌─────────────────────┐
    │  Gene Expression │      │   Preprocessing   │      │        VAE          │
    │   CSV File       │─────▶│  log1p + z-score  │─────▶│   Encoder/Decoder   │
    │  (~20K genes)    │      │                   │      │                     │
    └─────────────────┘      └───────────────────┘      └──────────┬──────────┘
                                                                    │
                                                                    ▼
                                                        ┌─────────────────────┐
                                                        │  Latent Features    │
                                                        │     (512-dim)       │
                                                        └──────────┬──────────┘
                                                                    │
                              ┌─────────────────────────────────────┼─────────────┐
                              ▼                                     ▼             ▼
                    ┌─────────────────┐              ┌─────────────────┐   ┌──────────────┐
                    │   H5 Features   │              │  Classification │   │  Diffusion   │
                    │  (per patient)  │              │    Subtyping    │   │ Fine-tuning  │
                    └─────────────────┘              └─────────────────┘   └──────────────┘
```

## Module Structure

```
encoding/
├── __init__.py           # Module exports
├── config.py             # Configuration dataclasses
├── train.py              # Main training script
├── architecture/
│   ├── __init__.py       # Architecture exports
│   ├── layers.py         # FullyConnectedLayer building block
│   ├── encoder.py        # Probabilistic encoder (input → mean, log_var)
│   ├── decoder.py        # Probabilistic decoder (latent → reconstruction)
│   ├── vae.py            # Variational Autoencoder with MMD regularization
│   └── loss.py           # MMD loss functions
```

## Purpose

Transform patient-level genomic measurements into compact, meaningful feature vectors that can be used for:

1. **Conditional Image Generation**: Feed latent vectors to diffusion models to generate histopathology tiles reflecting genomic characteristics

2. **Patient Stratification**: Cluster patients based on encoded features for subtype discovery

3. **Downstream Classification**: Train classifiers on the learned representations

## Usage

### Basic Training

```bash
python -m src.encoding.train \
    --csv /path/to/gene_expression.csv \
    --out-dir ./output \
    --latent-dim 512 \
    --epochs 100
```

### With Custom Architecture

```bash
python -m src.encoding.train \
    --csv /path/to/gene_expression.csv \
    --out-dir ./output \
    --hidden-dim 2048,1024 \
    --latent-dim 512 \
    --batch-size 64 \
    --epochs 100
```

### With External Checkpoint Directory

```bash
python -m src.encoding.train \
    --csv /path/to/gene_expression.csv \
    --out-dir ./output \
    --checkpoint-dir /external/path/checkpoints \
    --epochs 100
```

## Input Format

**Gene Expression CSV:**
```
Patient_ID,Gene1,Gene2,...,GeneN,Majority_Subtype_mRNA
TCGA-12-3456,10.2,9.1,...,6.8,LumA
...
```

- First column: Patient identifiers
- Middle columns: Gene expression values
- Last column (optional): Subtype labels for stratified splitting

## Output

```
output/
├── encoded_mean.npy          # Latent features (N_patients, latent_dim)
├── encoded_mean.csv          # Same as CSV with patient IDs
├── id_mapping.json           # Unique ID → original patient ID
├── encoder.pth               # Trained encoder weights
├── decoder.pth               # Trained decoder weights
└── mopadi_features/
    ├── train/
    │   ├── TCGA-3C-AAAU.h5   # Per-patient H5 files for diffusion
    │   └── ...
    ├── test/
    │   └── ...
    ├── norm_state.pth        # Normalization statistics (conds_mean, conds_std)
    └── clinical_table.csv    # Patient metadata with train/test split
```

## Architecture Details

### VAE with MMD Regularization

The VAE uses Maximum Mean Discrepancy (MMD) instead of KL divergence for regularization. This provides more stable training for high-dimensional genomic data.

**Loss Function:**
```
L = L_reconstruction + β · L_MMD
  = MSE(x, x_hat) + β · MMD(z, N(0,I))
```

### MMD (Maximum Mean Discrepancy)

MMD measures the distance between two distributions using an RBF kernel:

```
MMD(P, Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
```

where k is the Gaussian kernel: `k(x,y) = exp(-||x-y||² / d)`

### Default Architecture

```
Input (20K genes)
    ↓
FullyConnectedLayer(20000 → 2048) + BatchNorm + LeakyReLU
    ↓
FullyConnectedLayer(2048 → 1024) + BatchNorm + LeakyReLU
    ↓
├── Mean Layer (1024 → 512)
└── LogVar Layer (1024 → 512)
    ↓
Reparameterization: z = μ + σ·ε
    ↓
FullyConnectedLayer(512 → 1024) + BatchNorm + LeakyReLU
    ↓
FullyConnectedLayer(1024 → 2048) + BatchNorm + LeakyReLU
    ↓
FullyConnectedLayer(2048 → 20000)
    ↓
Output (reconstruction)
```

## Extending the Module

To add a new encoder architecture (e.g., Transformer-based):

1. Create `architecture/transformer_encoder.py`
2. Implement the same interface: `forward(x) → (mean, log_var)`
3. Export in `architecture/__init__.py`
4. Use in training with `--encoder-type transformer`

The modular design allows swapping components while keeping the overall pipeline intact.

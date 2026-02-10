# Quality Assurance Module

This module provides tools for evaluating the quality of diffusion model reconstructions conditioned on genomic vectors.

## Metrics

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **MSE** | Mean Squared Error between pixel intensities | [0, ∞) | Lower = better (0 = perfect) |
| **PSNR** | Peak Signal-to-Noise Ratio | [0, ∞) dB | Higher = better (>40dB excellent, 30-40 good, <20 poor) |
| **SSIM** | Structural Similarity Index | [-1, 1] | Higher = better (>0.95 excellent, 0.85-0.95 good) |

## Installation

The module requires the following dependencies (most are likely already installed):

```bash
pip install numpy pillow scikit-image matplotlib seaborn
```

## Usage

### Command Line

Evaluate reconstruction quality for all matching patients:

```bash
python -m quality_assurance.evaluate_reconstruction \
    --original-zip-dir /path/to/original_tiles \
    --reconstructed-zip-dir /path/to/reconstructed_tiles \
    --output-dir ./evaluation_results \
    --plot-dir ./plots
```

For specific patients:

```bash
python -m quality_assurance.evaluate_reconstruction \
    --original-zip-dir /path/to/original_tiles \
    --reconstructed-zip-dir /path/to/reconstructed_tiles \
    --patient-ids patient-id-number \
    --output-dir ./evaluation_results
```

## Output Files

When running evaluation, the following files are generated:

| File | Description |
|------|-------------|
| `tile_metrics.csv` | Per-tile metrics (patient_id, tile_name, mse, psnr, ssim) |
| `patient_summary.csv` | Per-patient aggregated metrics |
| `evaluation_summary.json` | Overall summary and statistics |

### Plots (if --plot-dir specified)

| File | Description |
|------|-------------|
| `metrics_distribution.png` | Histograms and box plots of all metrics |
| `per_patient_metrics.png` | Bar charts comparing metrics across patients |
| `tile_comparison.png` | Side-by-side comparison of sample tiles |

## File Matching

The evaluation script matches original and reconstructed tiles by:

1. **Patient matching**: Extract TCGA patient ID (e.g., `TCGA-BH-A0AU`) from zip filenames
2. **Tile matching**: Match tiles by their basename (e.g., `tile_(12032.517, 8960.385).png`)

For this to work correctly, ensure your sampling script preserves original tile names (the `sample_tiles_from_genomic.py` script has been updated to do this in encode-decode mode).

## Module Structure

```
quality_assurance/
├── __init__.py              # Package exports
├── metrics.py               # MSE, PSNR, SSIM implementations
├── evaluate_reconstruction.py  # Main evaluation script
├── visualization.py         # Plotting functions
└── README.md               # This file
```

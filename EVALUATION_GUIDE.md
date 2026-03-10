# Reconstruction Quality Evaluation Guide

This guide explains how to evaluate genomic-conditioned diffusion autoencoder reconstructions using the configuration-based evaluation system.

## Quick Start

### 1. Configure `config.yaml`

Add or modify the `evaluation` section in your `src/config.yaml`:

```yaml
evaluation:
  # Required directories
  original_zip_dir: /path/to/original/tiles
  reconstructed_zip_dir: /path/to/reconstructed/tiles
  output_dir: /path/to/evaluation/results
  
  # Optional: where to save plots
  plot_dir: /path/to/plots
  
  # Optional: specific patients to evaluate (null = all)
  patient_ids: null  # or [TCGA-A1-A0SM, TCGA-5L-AAT0]
  
  # Results format
  save_csv: true      # per-tile and per-patient summaries
  save_json: true     # JSON summary with statistics
  
  # Visualization
  plot_types:
    - metrics_summary      # MSE/PSNR/SSIM distributions
    - per_patient_metrics  # bar charts per patient
    - comparison_grid      # Original | Reconstructed | SSIM-diff
    - metric_correlation   # scatter plots of metrics
  
  num_comparison_samples: 32
  include_diff_heatmap: true  # Add SSIM spatial maps with Crameri colormaps
```

### 2. Run Evaluation from Config

#### Option A: Unified Pipeline CLI (Recommended ✨)
```bash
cd /mnt/bulk-saturn/maralampert/genhist
source Master-Thesis-Repository/.venv/bin/activate

# Run evaluation only
python run_pipeline.py --config src/config.yaml --stage evaluation

# Run all pipeline stages in sequence
python run_pipeline.py --config src/config.yaml --stage all

# With quiet output
python run_pipeline.py --config src/config.yaml --stage evaluation --quiet
```

#### Option B: Direct Module Call
```bash
source Master-Thesis-Repository/.venv/bin/activate

# Run as a Python module
python -m quality_assurance --config src/config.yaml
```

#### Option C: Direct Script Call
```bash
source Master-Thesis-Repository/.venv/bin/activate

# Direct script invocation
python src/quality_assurance/run_evaluation.py --config src/config.yaml
```

### 3. Check Results

Results are saved to `output_dir`:
- **`tile_metrics.csv`** — Per-tile metrics (patient_id, tile_name, mse, psnr, ssim)
- **`patient_summary.csv`** — Summary statistics per patient
- **`evaluation_summary.json`** — Overall statistics and metadata

Plots are saved to `plot_dir`:
- **`metrics_distribution.png`** — 2×3 grid of histograms & box plots
- **`per_patient_metrics.png`** — Bar charts per patient
- **`tile_comparison.png`** — Original | Reconstructed | SSIM-diff triptychs (with Crameri colormaps)
- **`metric_correlation.png`** — Scatter plots of metric relationships

## Configuration Parameters

### Required
| Parameter | Type | Description |
|-----------|------|-------------|
| `original_zip_dir` | string | Path to original tile zip files |
| `reconstructed_zip_dir` | string | Path to reconstructed tile zips |
| `output_dir` | string | Directory for CSV/JSON results |

### Optional
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plot_dir` | string | null | Directory for generated plots (skip if null) |
| `patient_ids` | list | null | Specific patients to evaluate (null = all) |
| `save_csv` | bool | true | Save per-tile and per-patient CSV |
| `save_json` | bool | true | Save JSON evaluation summary |
| `plot_types` | list | all | Which plots to generate |
| `num_comparison_samples` | int | 16 | Max tile pairs in comparison grid |
| `include_diff_heatmap` | bool | true | Add SSIM spatial maps to comparisons |
| `cmap_diverging` | string | vik | Crameri colormap for diff maps |
| `cmap_heatmap` | string | lajolla | Fallback colormap for absolute diffs |

## All Stages via Unified CLI

The `run_pipeline.py` script supports all pipeline stages with a consistent interface:

```bash
# Run each stage individually
python run_pipeline.py --config src/config.yaml --stage preprocessing
python run_pipeline.py --config src/config.yaml --stage encoding
python run_pipeline.py --config src/config.yaml --stage training
python run_pipeline.py --config src/config.yaml --stage sampling
python run_pipeline.py --config src/config.yaml --stage evaluation

# Or run everything in order
python run_pipeline.py --config src/config.yaml --stage all
```

Stages not yet integrated into the unified CLI will display instructions for running them separately.

## Tile Matching Strategy

The evaluation automatically handles **three matching strategies** for genomic reconstructions, where original and reconstructed filenames differ:

1. **Exact basename match** — Original and reconstructed tiles with same filename
2. **Coordinate-based matching** (genomic case) — Extracts `(x, y)` from filenames like:
   - Original: `tile_(26386.952, 9222.624).png`
   - Reconstructed: `reconstructed_00005_tile_(26386.952,_9222.624).png`
3. **Ordered fallback** — Pair by file order (last resort)

## Usage from Python

You can also run evaluation programmatically:

```python
from quality_assurance import run_evaluation, load_config

config = load_config("src/config.yaml")
run_evaluation(config, verbose=True)
```

Or use the evaluator directly:

```python
from quality_assurance import ReconstructionEvaluator

evaluator = ReconstructionEvaluator(
    original_zip_dir="/path/to/original",
    reconstructed_zip_dir="/path/to/reconstructed",
)
results = evaluator.evaluate_all()
evaluator.save_results(output_dir="/path/to/results")
```

## Visualization Functions

The visualization module provides several plots with **Crameri scientific colormaps**:

```python
from visualization.reconstruction_eval import (
    plot_metrics_summary,
    plot_comparison_grid,
    plot_per_patient_metrics,
    plot_metric_correlation,
    plot_single_comparison,
)

# Single pair comparison with SSIM diff heatmap
plot_single_comparison(
    original=img1,
    reconstructed=img2,
    title="Tile TCGA-A1-A0SM",
    include_diff=True,  # Shows SSIM map with vik colormap
)

# Grid of comparisons (3 columns: Orig | Recon | SSIM)
plot_comparison_grid(
    tile_pairs=list_of_pairs,
    include_diff=True,  # Third column = spatial SSIM map
)
```

## Metrics Explained

- **MSE** — Mean Squared Error (lower = better)
- **PSNR** — Peak Signal-to-Noise Ratio in dB (higher = better)
- **SSIM** — Structural Similarity Index (higher = better, range [0, 1])

## Scientific Colormaps (Crameri)

All plots use Fabio Crameri's **perceptually uniform, color-vision-deficiency-friendly** scientific colormaps:

- **`batlowS`** — Categorical/qualitative colors
- **`batlow`** — Sequential (single-hue) heatmaps
- **`vik`** — Diverging (for SSIM difference maps)
- **`lajolla`** — Sequential heatmaps

Reference: Crameri, F. (2018). Scientific colour maps. Zenodo. https://doi.org/10.5281/zenodo.1243862

## Example: Full Workflow

```yaml
# config.yaml
evaluation:
  original_zip_dir: /mnt/bulk/genhist/data/BRCA-tumor-tiles-all
  reconstructed_zip_dir: /mnt/bulk/genhist/experiments/20260306_mopadi/reconstruct_1k/
  output_dir: /mnt/bulk/genhist/experiments/20260306_mopadi/evaluation/
  plot_dir: /mnt/bulk/genhist/experiments/20260306_mopadi/evaluation/plots/
  
  patient_ids: null  # evaluate all
  plot_types: [metrics_summary, per_patient_metrics, comparison_grid, metric_correlation]
  num_comparison_samples: 32
  include_diff_heatmap: true
```

```bash
python run_pipeline.py --config src/config.yaml --stage evaluation
```

Results:
```
✓ evaluation/tile_metrics.csv
✓ evaluation/patient_summary.csv
✓ evaluation/evaluation_summary.json
✓ evaluation/plots/metrics_distribution.png
✓ evaluation/plots/per_patient_metrics.png
✓ evaluation/plots/tile_comparison.png (+ SSIM diff heatmaps!)
✓ evaluation/plots/metric_correlation.png
```

## Troubleshooting

### Module not found when using `-m`
If you get `ModuleNotFoundError`, make sure you're running from the repo root and the src directory is in your Python path. The unified CLI (`run_pipeline.py`) is more reliable for this reason.

### Config file not found
Use absolute paths in config.yaml, or run from the directory where the config file is located.

### No tiles matched
Check that:
1. Original and reconstructed zip files exist and are readable
2. Both sets have overlapping patient IDs (first 12 characters: `TCGA-XX-XXXX`)
3. At least one matching strategy works (exact filename, coordinates, or ordered pairing)

### "No matching patients found"
Ensure the patient ID extraction logic works for your zip names. See **Tile Matching Strategy** section above.

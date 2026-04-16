# Tile Reconstruction from Genomic Features

Reconstruct histopathology tiles from bulk RNA-seq expression using the jointly-trained VAE-Diffusion model.

## Overview

This module provides tools to:
1. **Reconstruct tiles** from genomic conditioning (RNA-seq → VAE latent → Diffusion)
2. **Investigate noising steps** to verify proper encoding/decoding behavior

### Architecture

```
RNA-seq (19,000 genes)
    ↓
Genomic VAE Encoder
    ↓
VAE Latent (512 dim)
    ↓
Projection Head
    ↓
UNet Conditioning
    ↓
Diffusion Denoising (from random noise)
    ↓
Reconstructed Tile (512×512 RGB)
```

---

## Scripts

### 1. `reconstruct_tiles.py` — Main Reconstruction Pipeline

Reconstruct tiles for selected patients with full metrics computation.

#### Usage

```bash
# Image-guided reconstruction (condition on genomics)
python -m src.reconstruction.reconstruct_tiles \
  --checkpoint experiments/20260304_finetune_vae/last.ckpt \
  --config src/config.yaml \
  --patients TCGA-5L-AAT0 TCGA-5T-A9QA \
  --save-dir experiments/reconstructed_tiles/test_set \
  --n-tiles-per-patient 50

# Random noise mode (generate from scratch)
python -m src.reconstruction.reconstruct_tiles \
  --checkpoint experiments/20260304_finetune_vae/last.ckpt \
  --config src/config.yaml \
  --mode random_noise \
  --n-tiles-per-patient 100 \
  --save-dir experiments/reconstructed_tiles/random

# With investigation mode enabled
python -m src.reconstruction.reconstruct_tiles \
  --checkpoint experiments/20260304_finetune_vae/last.ckpt \
  --config src/config.yaml \
  --patients TCGA-A1-A0SK \
  --investigate \
  --n-tiles-per-patient 10
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Path to joint training checkpoint (`.ckpt`) |
| `--config` | str | **required** | Path to YAML config file |
| `--patients` | list[str] | None | Specific patient IDs (e.g., `TCGA-XX-XX`). If not provided, uses all patients in gene CSV |
| `--gene-csv` | str | auto | Path to gene expression CSV (auto-inferred from config if not provided) |
| `--tiles-dir` | str | auto | Path to tiles directory (auto-inferred from config if not provided) |
| `--save-dir` | str | `experiments/reconstructed_tiles` | Output directory |
| `--n-tiles-per-patient` | int | 20 | Number of tiles per patient to reconstruct |
| `--mode` | str | `image_guided` | Reconstruction mode: `image_guided` or `random_noise` |
| `--investigate` | flag | False | Save intermediate noising steps |
| `--device` | str | auto | Device (e.g., `cuda:0`, `cuda:1`, `cpu`) |

#### Output

```
experiments/reconstructed_tiles/
├── reconstruction_results.csv         # Metrics for all reconstructions
├── TCGA-5L-AAT0_tile_001.png
├── TCGA-5L-AAT0_tile_002.png
├── ...
├── investigation/                     # (if --investigate enabled)
│   ├── TCGA-5L-AAT0_tile_001/
│   │   ├── step_000_t1000.png
│   │   ├── step_001_t750.png
│   │   ├── ...
│   │   └── final_reconstruction.png
│   └── ...
```

#### Metrics

`reconstruction_results.csv` contains:

| Column | Description |
|--------|-------------|
| `patient_id` | Patient ID (TCGA-XX-XXXX) |
| `tile_name` | Original tile filename |
| `status` | `success` or error message |
| `ssim` | Structural Similarity Index (0-1, higher is better) |
| `mse` | Mean Squared Error (lower is better) |
| `psnr` | Peak Signal-to-Noise Ratio in dB (higher is better) |

---

### 2. `investigate_noising.py` — Separate Investigation Utility

Debug encoding/decoding and visualize diffusion trajectories independently.

#### Usage

```bash
# Visualize denoising trajectory (noise → final image)
python -m src.reconstruction.investigate_noising \
  --checkpoint experiments/20260304_finetune_vae/last.ckpt \
  --gene-expression dataframes/brca_gene_expression.csv \
  --patient TCGA-A1-A0SK \
  --output investigations/test_patient \
  --trajectory \
  --steps 15

# Investigate VAE encoder/decoder behavior
python -m src.reconstruction.investigate_noising \
  --checkpoint experiments/20260304_finetune_vae/last.ckpt \
  --gene-expression dataframes/brca_gene_expression.csv \
  --patient TCGA-A1-A0SK \
  --output investigations/test_patient \
  --encode-decode

# Both together
python -m src.reconstruction.investigate_noising \
  --checkpoint experiments/20260304_finetune_vae/last.ckpt \
  --gene-expression dataframes/brca_gene_expression.csv \
  --patient TCGA-A1-A0SK \
  --output investigations/test_patient \
  --trajectory \
  --encode-decode \
  --steps 20
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | **required** | Joint training checkpoint |
| `--gene-expression` | str | **required** | Gene expression CSV |
| `--patient` | str | **required** | Patient ID to investigate |
| `--output` | str | `investigations` | Output directory |
| `--config` | str | None | YAML config (optional) |
| `--steps` | int | 10 | Number of frames in trajectory |
| `--trajectory` | flag | False | Visualize denoising trajectory |
| `--encode-decode` | flag | False | Analyze VAE encoding/decoding |
| `--device` | str | auto | Device |

#### Output

**Trajectory Mode** (`--trajectory`):
```
investigations/test_patient/trajectory/
├── frame_000_noise_t1000.png          # Pure noise (t=T)
├── frame_001_alpha0.11.png            # Intermediate frames
├── frame_002_alpha0.22.png
├── ...
└── frame_final_reconstruction.png     # Final reconstructed tile
```

**Encode-Decode Mode** (`--encode-decode`):
```
investigations/test_patient/encode_decode/
└── vae_statistics.json                # Encoding/decoding statistics
```

Example `vae_statistics.json`:
```json
{
  "input_mean": 2.456,
  "input_std": 1.823,
  "latent_mean": 0.045,
  "latent_std": 0.987,
  "reconstruction_error": 0.123
}
```

---

## Workflow

### Step 1: Prepare Data

Ensure you have:
- **Gene Expression CSV**: One row per patient, ~19,000 gene columns
  ```
  Patient_ID,GENE1,GENE2,...,GENE19000
  TCGA-5L-AAT0,2.4,1.8,...
  TCGA-5T-A9QA,3.2,2.1,...
  ```

- **Tile Directories**: Organized by patient
  ```
  /path/to/tiles/
  ├── TCGA-5L-AAT0-01Z-00-DX1/
  │   ├── tile_001.jpg
  │   ├── tile_002.jpg
  │   └── ...
  ├── TCGA-5T-A9QA-01Z-00-DX1/
  │   ├── tile_001.jpg
  │   └── ...
  ```

### Step 2: Quick Investigation (Optional)

Before running full reconstruction, check that the model works:

```bash
python -m src.reconstruction.investigate_noising \
  --checkpoint <ckpt> \
  --gene-expression <csv> \
  --patient TCGA-A1-A0SK \
  --output debug_investigation \
  --trajectory --encode-decode
```

Review the saved images and JSON stats to verify:
- ✓ Noising trajectory shows clear progression (noise → image)
- ✓ VAE statistics are reasonable (latent centered ~0, std ~1)
- ✓ Reconstruction error is acceptable

### Step 3: Reconstruct Tiles

```bash
# Method 1: Command line directly
python -m src.reconstruction.reconstruct_tiles \
  --checkpoint <ckpt> \
  --config src/config.yaml \
  --save-dir experiments/reconstructed_tiles \
  --n-tiles-per-patient 50

# Method 2: Via SLURM
sbatch slurm/reconstruct_tiles.sh
```

### Step 4: Analyze Results

```bash
# View reconstruction statistics
cat experiments/reconstructed_tiles/reconstruction_results.csv

# Compute average metrics by patient
python -c "
import pandas as pd
df = pd.read_csv('experiments/reconstructed_tiles/reconstruction_results.csv')
print(df.groupby('patient_id')[['ssim', 'mse', 'psnr']].mean())
"
```

---

## Integration with Joint Training Config

The scripts auto-detect settings from your `src/config.yaml`:

```yaml
reconstruction:
  checkpoint_path: experiments/20260304_finetune_vae/last.ckpt
  csv_path: dataframes/brca_gene_expression.csv
  tiles_zip_dir: data/BRCA-tumor-tiles-all
  output_dir: experiments/reconstructed_tiles
  n_tiles_per_patient: 20
  mode: image_guided
  investigate: false
  patient_col: Patient_ID
```

You can override these with CLI arguments:
```bash
python -m src.reconstruction.reconstruct_tiles \
  --checkpoint <ckpt> \
  --config src/config.yaml \
  --gene-csv custom_genes.csv \
  --tiles-dir /custom/tiles/path \
  --save-dir custom_output
```

---

## SLURM Integration

### Submit Reconstruction Job

```bash
sbatch slurm/reconstruct_tiles.sh
```

The script reads all parameters from `src/config.yaml` under the `reconstruction` section.

### Submit Investigation Job

```bash
sbatch slurm/investigate_noising.sh --patient TCGA-A1-A0SK
```

---

## Common Issues & Solutions

### Issue: "Model missing expected attributes"
**Solution**: Ensure the checkpoint is from the `JointLitModel` (contains VAE + Projection). Verify the checkpoint was saved with `src/joint_training/train.py`, not mopadi's standard training.

### Issue: "Patient not found in gene CSV"
**Solution**: Check that patient IDs match between gene CSV and tile directories. The script uses canonical format `TCGA-XX-XXXX` (lowercase). Verify with:
```bash
python -c "
from pathlib import Path
import pandas as pd
df = pd.read_csv('dataframes/brca_gene_expression.csv')
print('Patients in CSV:', df['Patient_ID'].unique()[:5])
print('Tile folders:', list(Path('data/BRCA-tumor-tiles-all').glob('TCGA*'))[:5])
"
```

### Issue: CUDA out of memory during reconstruction
**Solution**: Reduce batch size or number of tiles:
```bash
python -m src.reconstruction.reconstruct_tiles \
  --checkpoint <ckpt> \
  --config src/config.yaml \
  --n-tiles-per-patient 10  # Reduce from default 20
  --device cuda:0
```

### Issue: Investigation mode produces blank images
**Solution**: This likely indicates the diffusion model is still learning during early checkpoints. Try a later checkpoint:
```bash
# Check available checkpoints
ls -lh experiments/20260304_finetune_vae/checkpoints/

# Use a later epoch
python -m src.reconstruction.investigate_noising \
  --checkpoint experiments/20260304_finetune_vae/checkpoints/epoch_50.ckpt \
  --gene-expression dataframes/brca_gene_expression.csv \
  --patient TCGA-A1-A0SK \
  --output debug \
  --trajectory
```

---

## Next Steps

### Enhanced Investigation
Future improvements could include:
- Step-by-step diffusion sampling visualization (not just interpolation)
- Attention map visualization showing genomic influence
- Comparison between different genomic profiles side-by-side

---

## References

- **Model Architecture**: See [model.py](../joint_training/model.py) `JointLitModel` class
- **Dataset Format**: See [dataset.py](../joint_training/dataset.py) `GenomicTileDataset` class

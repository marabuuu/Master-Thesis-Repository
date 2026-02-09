# Genomic-Conditioned Diffusion Fine-tuning

This module enables generating histopathology tile images conditioned on genomic feature vectors using a fine-tuned MoPaDi diffusion model.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GENOMIC → TILE GENERATION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                         TRAINING PHASE
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │   ┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐   │
  │   │  Genomic    │      │  Projection      │      │ Diffusion Model  │   │
  │   │  Features   │─────▶│  Head Training   │─────▶│  Fine-tuning     │   │
  │   │  (512-dim)  │      │  (Step 1)        │      │  (Step 2)        │   │
  │   └─────────────┘      └──────────────────┘      └──────────────────┘   │
  │                               │                          │              │
  │                               ▼                          ▼              │
  │                    projection_head_best.pt    diffusion_genomic_best.pt │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘

                         INFERENCE PHASE
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                                                                          │
  │   ┌─────────────┐      ┌──────────────────┐      ┌──────────────────┐   │
  │   │  Genomic    │      │  Projection      │      │  cDDIM Sampler   │   │
  │   │  Features   │─────▶│  Head            │─────▶│  (Denoising)     │──▶│ Tiles
  │   │  (512-dim)  │      │                  │      │                  │   │
  │   └─────────────┘      └──────────────────┘      └──────────────────┘   │
  │                                                         ▲               │
  │                                                         │               │
  │                              Random Noise x_T ──────────┘               │
  │                                   OR                                    │
  │                              Encoded Real Tile                          │
  │                                                                          │
  └──────────────────────────────────────────────────────────────────────────┘
```

## Module Purpose

Transform patient-level genomic feature vectors (e.g., RNA-seq embeddings) into synthetic histopathology tile images that reflect genomic characteristics. This creates a bridge between molecular and morphological data.

**Key Use Cases:**
- Generate tiles reflecting specific genomic subtypes (e.g., PAM50 subtypes in breast cancer)
- Visualize how genomic features might manifest morphologically
- Data augmentation with genomic-conditioned synthetic images
- Counterfactual image generation ("What would this patient's tissue look like with different genomics?")

---

## 1. Projection Head Training

**Script:** `projection_head_genomic.py`

### Purpose
Train a learnable MLP to map genomic feature vectors into the conditioning space expected by the pretrained MoPaDi diffusion model. This alignment ensures genomic features can effectively guide image generation.

### Training Objective
**Distribution Matching** (recommended): The projection head learns to transform genomic features so their distribution matches the diffusion model's expected conditioning distribution (conds_mean, conds_std).

Loss function:
```
L = L_mean + L_var + L_diversity
  = MSE(batch_mean, conds_mean) 
  + MSE(batch_std, conds_std)
  + ReLU(τ - mean_pairwise_distance)
```

### Input / Output

| Type | Description |
|------|-------------|
| **Input** | Genomic H5 files: `{patient_id}.h5` with `feats` dataset (512-dim vectors) |
| **Input** | Pretrained diffusion checkpoint (to extract target mean/std) |
| **Output** | `projection_head_best.pt` - trained projection head weights |
| **Output** | `projection_head_config.json` - architecture configuration |

### Example Commands

```bash
# Basic training with distribution matching
python projection_head_genomic.py \
    --mode distribution_matching \
    --genomic-h5-dir /path/to/genomic_features \
    --diffusion-ckpt ./diffusion_without_encoder.ckpt \
    --out-dir ./projection_head_output \
    --epochs 50 \
    --lr 1e-4

# With custom architecture
python projection_head_genomic.py \
    --mode distribution_matching \
    --genomic-h5-dir /path/to/genomic_features \
    --diffusion-ckpt ./diffusion_without_encoder.ckpt \
    --out-dir ./projection_head_output \
    --arch mlp \
    --hidden-dim 512 \
    --num-layers 2 \
    --epochs 100
```

### Expected Training Time
- ~5-10 minutes for 50 epochs on a single GPU
- Loss should converge to ~0.02-0.05

---

## 2. Diffusion Fine-tuning

**Script:** `finetune_diffusion_with_genomic.py`

### Purpose
Fine-tune the pretrained MoPaDi diffusion model to generate tiles conditioned on projected genomic features. This teaches the model the semantic relationship between genomic features and tissue morphology.

### Training Objective
Standard diffusion denoising loss with genomic conditioning:
```
L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t, cond)||² ]
```
where `cond = normalize(ProjectionHead(genomic))`.

### Input / Output

| Type | Description |
|------|-------------|
| **Input** | Trained projection head checkpoint (`projection_head_best.pt`) |
| **Input** | Pretrained diffusion checkpoint |
| **Input** | Genomic H5 files (512-dim feature vectors) |
| **Input** | Tile ZIP files (real tiles for training, matched by patient ID) |
| **Output** | `diffusion_genomic_best.pt` - combined checkpoint with model + projection head |
| **Output** | `diffusion_genomic_epoch_XX.pt` - epoch checkpoints |

### Example Commands

```bash
# Basic fine-tuning
python finetune_diffusion_with_genomic.py \
    --projection-head-ckpt ./projection_head_best.pt \
    --diffusion-ckpt ./diffusion_without_encoder.ckpt \
    --genomic-h5-dir /path/to/genomic_features \
    --tiles-zip-dir /path/to/tile_zips \
    --out-dir ./finetuned_diffusion \
    --epochs 50 \
    --lr 5e-6

# With more tiles per patient and gradient accumulation
python finetune_diffusion_with_genomic.py \
    --projection-head-ckpt ./projection_head_best.pt \
    --diffusion-ckpt ./diffusion_without_encoder.ckpt \
    --genomic-h5-dir /path/to/genomic_features \
    --tiles-zip-dir /path/to/tile_zips \
    --out-dir ./finetuned_diffusion \
    --epochs 50 \
    --lr 5e-6 \
    --tiles-per-patient 20 \
    --gradient-accumulation-steps 4
```

### Training Modes
- **Joint training** (default): Both UNet and projection head are trainable
- **Frozen projection head**: Add `--freeze-projection-head` to only train UNet

### Expected Training Time
- ~4-8 hours for 50 epochs on H100 (depends on dataset size)
- Loss should reach ~0.02-0.03

---

## 3. Sampling / Inference

**Script:** `sample_tiles_from_genomic.py`

### Purpose
Generate synthetic tile images from genomic feature vectors using the fine-tuned diffusion model. Supports two sampling modes for different use cases.

### Sampling Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `random` | Generate from pure random noise x_T | Fully synthetic diverse samples |
| `encode-decode` | Encode real tile → noise → decode with genomic conditioning | Preserve structure, apply genomic features |

### Input / Output

| Type | Description |
|------|-------------|
| **Input** | Fine-tuned checkpoint (`diffusion_genomic_best.pt`) |
| **Input** | Genomic H5 files for patients to sample |
| **Input** | (encode-decode only) Real tile ZIPs as starting points |
| **Output** | ZIP files per patient containing generated tile PNGs |

### Example Commands

```bash
# Mode 1: Random noise generation (fully synthetic)
python sample_tiles_from_genomic.py \
    --checkpoint ./diffusion_genomic_best.pt \
    --genomic-h5-dir /path/to/genomic_features \
    --output-dir ./generated_tiles \
    --mode random \
    --num-samples-per-patient 4

# Mode 2: Encode-decode (preserve structure from real tiles)
python sample_tiles_from_genomic.py \
    --checkpoint ./diffusion_genomic_best.pt \
    --genomic-h5-dir /path/to/genomic_features \
    --tiles-zip-dir /path/to/tile_zips \
    --output-dir ./generated_tiles \
    --mode encode-decode \
    --num-samples-per-patient 4 \
    --encode-steps 250

# Sample specific patients only
python sample_tiles_from_genomic.py \
    --checkpoint ./diffusion_genomic_best.pt \
    --genomic-h5-dir /path/to/genomic_features \
    --output-dir ./generated_tiles \
    --patient-ids TCGA-3C-AAAU TCGA-3C-AALJ \
    --num-samples-per-patient 8
```

### Encode Steps Parameter
Controls the trade-off between structure preservation and variation:
- **Lower values (100-250):** More structure from original tile preserved
- **Higher values (500+):** More noise, more variation, closer to random sampling
- **Default (full T):** Maximum noise, equivalent to random sampling with structure hint

### Output Format
```
output_dir/
├── TCGA-3C-AAAU.zip
│   ├── sample_00.png
│   ├── sample_01.png
│   ├── sample_02.png
│   └── sample_03.png
├── TCGA-3C-AALJ.zip
│   └── ...
└── ...
```
---

## File Format Requirements

### Genomic H5 Files
```
{patient_id}.h5
└── feats: (N, 512) or (512,) float32 array
```
If multiple vectors (N > 1), they are averaged to get a single patient-level representation.

### Tile ZIP Files
```
{patient_id}*.zip
├── tile_001.png
├── tile_002.jpg
└── ...
```
Tiles should be 512×512 (or will be resized/cropped).

---

# Genomic-Conditioned Diffusion Fine-tuning

This module enables generating histopathology tile images conditioned on genomic feature vectors using a fine-tuned MoPaDi diffusion model.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     GENOMIC → TILE GENERATION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────────┘

Step 1: Genomic features (512-dim vectors) → Projection Head → Conditioning Space
Step 2: Conditioning Space + Random Noise → Diffusion Model → Synthetic Tiles
Step 3 (Optional): Encode real tiles → Noise → Decode with Genomic Conditioning
```

## Module Purpose

Transform patient-level genomic feature vectors (e.g., RNA-seq embeddings) into synthetic histopathology tile images that reflect genomic characteristics. This creates a bridge between molecular and morphological data.

**Key Use Cases:**
- Generate tiles reflecting specific genomic subtypes (e.g., PAM50 subtypes in breast cancer)
- Visualize how genomic features might manifest morphologically
- Data augmentation with genomic-conditioned synthetic images
- Counterfactual image generation ("What would this patient's tissue look like with different genomics?")

---

## 1. Status and Scope

This module is maintained as a **legacy workflow**. The current repository keeps:
- `finetune_diffusion_with_genomic.py` for diffusion fine-tuning
- `sample_tiles_from_genomic.py` for sampling/inference

Older references to a standalone `projection_head_genomic.py` script are obsolete in this repo.

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
python -m src.finetune_diffusion.finetune_diffusion_with_genomic \
    --projection-head-ckpt ./projection_head_best.pt \
    --genomic-h5-dir ./genomic_features \
    --tiles-zip-dir ./tile_zips \
    --lr 5e-6

# With more tiles per patient and gradient accumulation
python -m src.finetune_diffusion.finetune_diffusion_with_genomic \
    --projection-head-ckpt ./projection_head_best.pt \
    --genomic-h5-dir ./genomic_features \
    --tiles-zip-dir ./tile_zips \
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
python -m src.finetune_diffusion.sample_tiles_from_genomic \
    --checkpoint ./diffusion_genomic_best.pt \
    --genomic-h5-dir ./genomic_features \
    --output-dir ./generated_tiles \
    --num-samples-per-patient 4

# Mode 2: Encode-decode (preserve structure from real tiles)
python -m src.finetune_diffusion.sample_tiles_from_genomic \
    --checkpoint ./diffusion_genomic_best.pt \
    --genomic-h5-dir ./genomic_features \
    --tiles-zip-dir ./tile_zips \
    --output-dir ./generated_tiles \
    --mode encode-decode \
    --num-samples-per-patient 4

# Sample specific patients only
python -m src.finetune_diffusion.sample_tiles_from_genomic \
    --checkpoint ./diffusion_genomic_best.pt \
    --genomic-h5-dir ./genomic_features \
    --output-dir ./generated_tiles \
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

# Joint Genomic VAE + Diffusion Training

## Overview

This system jointly trains two models in tandem:

1. **Genomic VAE**: Encodes bulk-RNA sequencing data (~19,000 genes) into a latent representation
2. **Diffusion UNet**: Generates histopathology tile images, conditioned on the VAE's latent encoding

The key innovation is that **the VAE latent representation directly conditions the diffusion model**, so the model learns to generate images that are consistent with genomic features.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONE TRAINING STEP                        │
└─────────────────────────────────────────────────────────────────┘

INPUT DATA
  ├─ Gene Expression CSV:    (n_patients, ~19000 genes)
  └─ Tile Images (ZIP):      (many tiles per patient, 512×512 RGB)

         │
         ├──────────────────────┬──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
     ┌─────────┐          ┌──────────┐          ┌─────────┐
     │ Genomic │          │ Tile     │          │ Patient │
     │ Vector  │          │ Image    │          │ ID      │
     │ (1,G)   │          │ (3,512,  │          │         │
     │         │          │  512)    │          │         │
     └────┬────┘          └────┬─────┘          └─────────┘
          │                    │
          │                    ▼
          │              ┌──────────────┐
          │              │ Normalize    │
          │              │ to [-1, 1]   │
          │              └──────┬───────┘
          │                     │
    ┌─────▼────────────────────┬┘
    │                          │
    │   ╔════════════════════╗ │
    │   ║  VAE ENCODING      ║ │
    │   ╠════════════════════╣ │
    │   ║ Encoder (linear)   ║ │
    │   ║ Input: genes       ║ │
    │   ║ Output: μ, σ       ║ │
    │   ║                    ║ │
    │   ║ Reparameterize:    ║ │
    │   ║ z ~ N(μ, σ²)      ║ │
    │   ╚────────┬───────────╝ │
    │            │             │
    │     ┌──────▼──────┐◄─────┘
    │     │   z_vae     │
    │     │ (512 dim)   │
    │     └──────┬──────┘
    │            │
    │     ┌──────▼──────────────────┐
    │     │ Projection Head (MLP)    │
    │     │ 512 → 512 (cond_dim)    │
    │     └──────┬───────────────────┘
    │            │
    │     ┌──────▼─────┐
    │     │ cond       │
    │     │ (512 dim)  │
    │     └──────┬─────┘
    │            │
    │     ┌──────▼──────────────────────┐
    │     │ DIFFUSION UNet FORWARD PASS │
    │     │ ─────────────────────────── │
    │     │ Input: noisy image + t      │
    │     │ Condition: cond via embed   │
    │     │ Output: predicted noise     │
    │     └──────┬──────────────────────┘
    │            │
    │     ┌──────▼────────┐        ┌────────────────────┐
    │     │ Diffusion Loss│        │ VAE Decoder Loss   │
    │     │ L_diff        │        │ L_recon + β*L_mmd  │
    │     └──────┬────────┘        └────────┬───────────┘
    │            │                         │
    │            └────────────┬────────────┘
    │                         │
    │              L_total = L_diff + λ*L_vae
    │                         │
    │            Backprop through both models
    │
    └─────────────────────────────────────────┘
```

---

## Architecture Diagram Walkthrough

Let me break down each key step in the diagram above:

### **INPUT DATA** (Top)
```
├─ Gene Expression CSV:    (n_patients, ~19000 genes)
└─ Tile Images (ZIP):      (many tiles per patient, 512×512 RGB)
```
We start with two completely different modalities:
- **Gene expression**: One vector per patient, capturing their molecular profile
- **Tile images**: Multiple patches from the same patient's histology slides

The key insight is that these two data types describe the **same biological sample** but from different angles.

---

### **DATA BATCHING** (First split)
```
     ┌─────────┐          ┌──────────┐          ┌─────────┐
     │ Genomic │          │ Tile     │          │ Patient │
     │ Vector  │          │ Image    │          │ ID      │
     │ (1,G)   │          │ (3,512,  │          │         │
     │         │          │  512)    │          │         │
     └────┬────┘          └────┬─────┘          └─────────┘
```
**Why three inputs?**
- **Genomic vector**: Raw gene expression (~19k dimensions) for that patient
- **Tile image**: One noisy/random tile from that patient's ZIP
- **Patient ID**: Metadata to track which sample everything came from (helps with debugging, logging)

The genomic vector and tile image come from the **same patient** — that's the coupling that makes joint training work.

---

### **IMAGE NORMALIZATION**
```
          │
          ▼
     ┌──────────────┐
     │ Normalize    │
     │ to [-1, 1]   │
     └──────┬───────┘
```
**Why normalize to [-1, 1]?**  
Diffusion models are trained on this range because:
- It's symmetric around zero (good for noise prediction)
- Allows the model to easily flip between real images and noise
- Standard practice in generative modeling

The genomic vector is also normalized (log1p + z-score) separately before the VAE encoder.

---

### **VAE ENCODER** (The heart of genomic processing)
```
   ╔════════════════════╗
   ║  VAE ENCODING      ║
   ╠════════════════════╣
   ║ Encoder (linear)   ║
   ║ Input: genes       ║
   ║ Output: μ, σ       ║
   ║                    ║
   ║ Reparameterize:    ║
   ║ z ~ N(μ, σ²)      ║
   ╚────────┬───────────╝
```
**What's happening here?**

1. **Encoder network**: Maps 19,000 genes → 512 latent dimensions
   - `19000 → 2048 → 1024 → 512`
   - Each layer compresses the information, learning abstract genomic features
   - Output is **two vectors**: mean μ and log-variance log_σ

2. **Reparameterization trick**: Convert probabilistic output to a concrete sample
   - Instead of just outputting μ, we output a distribution: `z ~ N(μ, σ²)`
   - This allows the VAE to learn uncertainty in the latent space
   - At inference time, we use the mean (deterministic): `z = μ`

**Why not just use μ directly?** The randomness (σ) helps regularize the model — forces the VAE to not collapse to a single point, keeping the latent space smooth.

---

### **z_vae: The Bottleneck Latent**
```
     ┌──────▼──────┐
     │   z_vae     │
     │ (512 dim)   │
     └──────┬──────┘
```
**This is the most important intermediate representation!**

- **512 dimensions**: Highly compressed summary of 19,000 genes
- **Learned by the VAE**: Optimized to reconstruct genes accurately
- **Represents genomic similarity**: Similar genes → similar z vectors → can be used for clustering patients

This latent vector is the **bridge** between genomics (left side) and microscopy (right side).

---

### **PROJECTION HEAD** (The adapter)
```
     ┌──────▼──────────────────┐
     │ Projection Head (MLP)    │
     │ 512 → 512 (cond_dim)    │
     └──────┬───────────────────┘
```
**Why do we need this if input and output are both 512-dim?**

It's **not about changing dimensions** — it's about learning the **right transformation** for the diffusion model:

- `z_vae` is optimized for gene reconstruction (what the VAE decoder sees)
- `cond` is optimized for modifying image generation (what the UNet sees)

**Analogy**: Imagine a translator standing between a doctor (VAE, thinking about genes) and an artist (UNet, thinking about visual features). The translator speaks both languages fluently but must learn how to map the doctor's concepts into visual instructions that the artist can use.

**The MLP learns:**
- Which genomic features matter for image generation
- How to amplify important features and suppress noise
- How to combine features in ways the UNet can use effectively

---

### **cond: The Conditioning Signal**
```
     ┌──────▼─────┐
     │ cond       │
     │ (512 dim)  │
     └──────┬─────┘
```
**This vector controls image generation.**

It gets passed to the diffusion UNet, which learns:
- "If cond says 'high HER2 expression', add more dense tissue patterns"
- "If cond says 'basal subtype', shift the color palette toward blue hues"
- Etc.

The UNet doesn't learn explicit rules — it learns implicit patterns from the data.

---

### **DIFFUSION UNet FORWARD PASS**
```
     ┌──────▼──────────────────────┐
     │ DIFFUSION UNet FORWARD PASS │
     │ ─────────────────────────── │
     │ Input: noisy image + t      │
     │ Condition: cond via embed   │
     │ Output: predicted noise     │
     └──────┬──────────────────────┘
```
**What's the UNet doing?**

1. **Takes three inputs**:
   - `x_t`: The image at diffusion step `t` (progressively noisier → cleaner during generation)
   - `t`: The timestep (tells model how much noise to expect)
   - `cond`: Genomic conditioning signal

2. **Predicts the noise** that was added at this step
   - This is the core of diffusion: learn to denoise iteratively
   - By doing this correctly for all noise levels, it learns the distribution of clean images

3. **Uses conditioning via `TimeStyleSeperateEmbed`**:
   - Special module that mixes the conditioning signal into each residual block
   - Acts like "style" embedding (think: what overall "look" should this image have?)
   - The UNet learns to respect the conditioning signal while still denoising effectively

---

### **LOSS COMPUTATION** (Training signal)
```
     ┌──────▼────────┐        ┌────────────────────┐
     │ Diffusion Loss│        │ VAE Decoder Loss   │
     │ L_diff        │        │ L_recon + β*L_mmd  │
     └──────┬────────┘        └────────┬───────────┘
            │                         │
            └────────────┬────────────┘
                         │
              L_total = L_diff + λ*L_vae
```

**Two independent loss signals working together:**

1. **Diffusion Loss (L_diff)**:
   - "How well did you predict the noise?"
   - Measures: Does the UNet learn realistic image patterns?
   - Provides **visual feedback**: "Your generated tiles don't look like real histology"

2. **VAE Loss** (`L_recon + β*L_mmd`):
   - `L_recon`: "Can you reconstruct the original genes from the latent?"
     - Measures: Does the VAE latent capture gene information?
   - `L_mmd`: "Is your latent space smooth and well-distributed?"
     - Prevents the VAE from collapsing into a few modes
   - Provides **genomic feedback**: "Your latent doesn't capture gene variation well"

**How they combine:**
```
L_total = L_diff + λ * (L_recon + β * L_mmd)
```
- `λ` controls the trade-off (default 1.0 = equal weight)
- If `λ` is too high: Model prioritizes genes over realistic images
- If `λ` is too low: Model ignores genes, just generates pretty pictures

---

### **BACKPROP THROUGH BOTH MODELS**
```
            │
Backprop through both models
```
This is the magic: **both models improve simultaneously**.

- UNet improves: "Learn better patterns that respect the genomic signal"
- VAE improves: "Learn better latents that help the UNet generate realistic images"

It's a **mutually beneficial** collaboration — neither model can ignore the other.

---

## Summary of Key Characteristics

| Component | Dimension | Purpose | Key Learning |
|-----------|-----------|---------|--------------|
| **Genomic Input** | 19,000 | Raw expression data | Which genes matter? |
| **VAE Encoder** | 19k → 512 | Compress genes to latent | Learn genomic manifold |
| **z_vae (latent)** | 512 | Bridge representation | Genomic similarity metric |
| **Projection Head** | 512 → 512 | Adapt for images | Map genes to visual features |
| **cond signal** | 512 | Image control | Condition generation |
| **UNet** | 138M params | Generate images | Realistic morphology + genomic control |
| **L_diff** | Scalar | Denoising accuracy | Visual quality |
| **L_vae** | Scalar | Reconstruction + smoothness | Genomic quality |

---

## The Training Dance

Think of it like a **dance between two partners**:

🧬 **VAE** (the genomics partner):  
"Here's what I learned about genes in this latent space"  
↓  
🎨 **UNet** (the image partner):  
"Thanks, but I need the genomic info in this specific format to control images effectively"  
↓  
🧬 **VAE** (learns from feedback):  
"Ah, I see! Let me adjust my latent to be more useful for image generation"  
↓  
🎨 **UNet** (learns from feedback):  
"Better! Now I can generate images that respect both gene patterns and look realistic"  
↓  
*Repeat for 200M samples*

Neither model can succeed without the other. That's what makes **joint training** powerful.



```
BATCH = {
  "img":       (batch_size, 3, 512, 512)  → normalized to [-1, 1]
  "genomic":   (batch_size, n_genes)      → log1p + z-score normalized
  "patient_id": patient TCGA IDs
}
```

### Forward Pass

1. **VAE Encoder**: `genomic` → `(μ, log_var)`
   - 2 hidden layers: 19000 → 2048 → 1024 → 512
   - Outputs mean and log-variance of latent distribution

2. **Reparameterization**: `(μ, log_var)` → `z`
   - Deterministic at inference: `z = μ`

3. **Projection Head**: `z` → `cond`
   - 512 → 512 (same as UNet conditioning dimension)
   - MLP with LayerNorm, GELU, dropout

4. **VAE Decoder**: `z` → `gene_recon`
   - Mirrors encoder: 512 → 1024 → 2048 → 19000
   - Reconstruction loss: MSE(gene_recon, genomic)

5. **Diffusion UNet**: `(img, noise_level, cond)` → `noise_prediction`
   - 138M parameters (BeatGANsAutoencModel)
   - `cond` injected via TimeStyleSeperateEmbed
   - Diffusion loss: MSE(predicted_noise, sampled_noise)

### Loss Computation

```
L_recon  = MSE(decoder(z), genomic)
L_mmd    = MMD(z, N(0,I))  # Maximum Mean Discrepancy regularization
L_vae    = L_recon + β * L_mmd

L_diff   = MSE(u_net(x_t, t, cond), ε)

L_total  = L_diff + λ * L_vae
```

Where:
- `λ` (lambda_vae): weight of VAE loss vs diffusion loss (default: 1.0)
- `β` (beta_mmd): weight of MMD regularization (default: 1.0)

---

## Training Configuration

```yaml
Joint Genomic VAE + Diffusion Training:

  Data:
    - CSV with gene expression (~19k genes, 981 patients)
    - ZIP archives with histopathology tiles (~1048 zips, 145k+ tiles)
    - Patient-level train/val/test splits (80/10/10)
      → Each patient appears in only ONE split
      → Test patients completely held out

  Models:
    - VAE:           87.6M params (encoder: 19k→2048→1024→512, decoder: mirrors)
    - Diffusion UNet: 138M params (BeatGANsAutoencModel)
    - Projection:    526K params (512→512 MLP)
    - Total:         226M trainable, 364M total

  Training:
    - Batch Size:     2
    - Learning Rates: UNet 1e-5, VAE 1e-3, Projection 3e-4
    - Optimizer:      Adam with weight decay
    - Warmup:         500 steps → Cosine annealing
    - EMA:            0.9999 decay on UNet
    - Duration:       200M samples (∞ cap via epochs)

  Checkpointing:
    - Every 200k samples
    - Sample visualization every 50k samples
    - Auto-resume from last.ckpt
```

---

## Patient-Level Train/Val/Test Splits

```python
Total matched patients: 94
├─ Train:  75 patients (1,105,865 tiles)
├─ Val:    10 patients (145,610 tiles)
└─ Test:    9 patients (completely held out, never seen during training)

# Splits saved to: experiments/joint_training/patient_splits.json
{
  "train": {"patients": ["TCGA-XX-XXXX", ...], "n_patients": 75},
  "val":   {"patients": ["TCGA-YY-YYYY", ...], "n_patients": 10},
  "test":  {"patients": ["TCGA-ZZ-ZZZZ", ...], "n_patients": 9}
}
```

This ensures **no patient leakage** — each patient's data is used in only one split.

---

## Key Implementation Details

### 1. Subclasses mopadi's LitModel
Instead of reimplementing training infrastructure, we inherit:
- **EMA updates** for UNet (exponential moving average of weights)
- **Gradient clipping** and optimizer scheduling
- **Sample visualization** to TensorBoard
- **Checkpointing** and auto-resume
- **DDP support** for multi-GPU training

### 2. VAE Device Management
VAE is created on CPU but moved to GPU in `on_fit_start()` and verified every batch in `on_train_batch_start()` to prevent CUDA device mismatch errors.

### 3. Joint Loss Backprop
Both models are updated every step with a weighted combination of losses:
```python
loss = diff_loss + λ_vae * (recon_loss + β_mmd * mmd_loss)
loss.backward()  # Updates both UNet, VAE, and Projection params
```

### 4. Conditioning Signal
The genomic-derived `cond` vector is passed to the diffusion UNet's `TimeStyleSeperateEmbed`, which adds it as a learned style embedding to each residual block.

---

## Outputs & Monitoring

### During Training
- **TensorBoard logs**: `experiments/joint_training/joint/`
  - `loss`: total loss
  - `loss/diff`: diffusion component
  - `loss/vae_recon`: VAE reconstruction
  - `loss/vae_mmd`: VAE regularization
  - Generated sample images every 50k samples

### Checkpoints
- **Last checkpoint**: `experiments/joint_training/joint/last.ckpt`
- **Resumable**: Resubmit the job to continue training automatically

### After Training

```bash
# Extract VAE latent features (per-patient h5 files)
python -m src.joint_training.train --config src/config.yaml \
  --extract-latents --split all

# Files: experiments/joint_training/latents/<patient_id>.h5
# Each contains: z, mu, log_var, recon, cond, genes
```

---

## Running the Training

```bash
# Single GPU (auto-resumes)
sbatch slurm/joint_training.sh

# Or directly:
python run_pipeline.py --config src/config.yaml --stage joint_training

# Monitor with TensorBoard
tensorboard --logdir experiments/joint_training/joint --port 6006
```

---

## Why This Matters

By jointly training the VAE and diffusion model:

✅ **Genomic consistency**: Generated tiles are constrained by bulk-RNA features  
✅ **Multi-modal learning**: Model learns the relationship between gene expression and morphology  
✅ **Reusable encodings**: VAE latents can be used for downstream genomics analysis  
✅ **Conditioned generation**: Sample realistic tiles given a patient's genomic profile  

This bridges **genomic data** (expression profiles) and **histopathology data** (tile images) in a unified learned representation.

# Joint Genomic Encoder + Diffusion Training

## Overview

This system jointly trains three components end-to-end:

1. **Genomic Encoder** (ProbabilisticEncoder): Encodes bulk-RNA sequencing data into a meaningful latent representation
2. **Projection Head** (Linear MLP): Maps encoder latents to the UNet conditioning space
3. **Diffusion UNet**: Generates histopathology tile images, conditioned on projected genomic features

**Key Innovation**: All three components learn simultaneously from a **pure diffusion loss**. The encoder learns a biologically meaningful latent space through diffusion gradients, without explicit reconstruction objectives.

---

## Architecture Overview

```
RNA-seq Data                   Image Data
     ↓                              ↓
[Encoder] ──→ [Latent Space]    [Images]
     ↓                              ↓
[Project] ──→ [Conditioning Vector] ──→ [UNet Diffusion Model]
                                         ↓
                                    [MSE Diffusion Loss]
                                         ↓
                                    [Backprop → All 3 components]
```

### Loss Computation (Single, Unweighted)

```
loss = MSE(UNet_predicted_noise - actual_noise)
```

That's it. **No VAE reconstruction loss, no MMD regularization.** The same proven loss used in mopadi.

---

## Comparison with mopadi

### Similarities ✅

| Aspect | Implementation |
|--------|---|
| **Loss function** | MSE diffusion loss only (unweighted) |
| **Model architecture** | UNet with timestep and conditioning |
| **EMA updates** | Applied to UNet every batch |
| **Training mode** | Noise prediction at random timesteps |
| **Optimizer** | Adam or AdamW with learning rate scheduling |

### Key Differences 🔄

| Component | mopadi | Joint Training |
|-----------|--------|---|
| **Feature source** | Pre-computed from frozen extractor (CONCH/Virchow) | Dynamically encoded from RNA-seq |
| **Feature extraction** | Frozen, cached to disk | Learned, computed per batch |
| **Trained modules** | UNet only | **UNet + Encoder + Projection** |
| **Conditioning** | Static load from `.h5` files | Dynamic computation during training |
| **Validation** | Optional | **Implemented** |
| **Early stopping** | Not built-in | Can be added (6-line change) |

---

## Loss Computation Details

### Step-by-Step Process

1. **Encode genomics** → latent vector
   ```python
   mean, log_var = self.encoder(genomic)
   z = mean + exp(0.5 * log_var) * randn_like(log_var)  # Reparameterization
   cond = self.projection(z)
   ```

2. **Sample timestep and corruption**
   ```python
   t = sample_timestep(batch_size)  # Random t ∈ [0, 1000)
   noise = randn_like(imgs)
   x_t = sqrt(α_cumprod[t]) * imgs + sqrt(1 - α_cumprod[t]) * noise
   ```

3. **Predict noise with UNet**
   ```python
   pred_noise = self.model(x_t, t, cond=cond)
   ```

4. **Compute loss**
   ```python
   loss = MSE(pred_noise, noise)  # No weighting, no additional terms
   ```

5. **Backpropagation**
   ```python
   loss.backward()
   # Gradients flow to: Encoder → Projection → UNet
   ```

### Why This Architecture Works

- **Encoder learns through diffusion signals**: Upstream (UNet) gradients teach the encoder to produce conditioning vectors that improve image generation
- **Simpler than VAE**: No need for explicit reconstruction loss or distribution matching (MMD)
- **Proven by mopadi**: Same loss used in production diffusion model training
- **Biologically justified**: The encoder latent space is shaped by image generation quality, not arbitrary reconstruction

---

## Training & Validation Logging

### Metrics Logged to TensorBoard

```
loss              ← Per-step training loss (every step)
loss_epoch        ← Epoch-averaged training loss
val_loss          ← Epoch-averaged validation loss
lr-*              ← Learning rates (from LearningRateMonitor callback)
```

### Viewing Results

```bash
tensorboard --logdir experiments/joint_training/joint/
```

Then navigate to **Scalars** tab to see all metrics.

### Logging Implementation

**Training step** (model.py, line 293-324):
```python
def training_step(self, batch, batch_idx):
    # Compute loss
    loss = self.sampler.training_losses(...).mean()
    
    # PyTorch Lightning logging (epoch aggregation + progress bar)
    self.log('loss_epoch', loss, on_step=False, on_epoch=True, ...)
    self.log('loss_step', loss, on_step=True, on_epoch=False, ...)
    
    # TensorBoard direct logging (per-step)
    self.logger.experiment.add_scalar('loss', loss.item(), self.num_samples)
    
    return {'loss': loss}
```

**Validation step** (model.py, line 326-343):
```python
def validation_step(self, batch, batch_idx):
    loss = self.sampler.training_losses(...).mean()
    self.log('val_loss', loss, on_step=False, on_epoch=True, ...)
    return {'val_loss': loss}
```

---

## Early Stopping

### Current Status: ❌ NOT Implemented

The trainer only uses checkpoint saving and learning rate monitoring:
```python
callbacks=[checkpoint, LearningRateMonitor()]
```

### To Add Early Stopping (6-line addition to train.py)

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=int(joint_cfg.get("early_stopping_patience", 5)),
    mode='min'
)

trainer = pl.Trainer(
    ...,
    callbacks=[checkpoint, LearningRateMonitor(), early_stopping],
)
```

Then add to `config.yaml`:
```yaml
joint_training:
  early_stopping_patience: 5  # Stop if val_loss doesn't improve for 5 epochs
```

---

## Key Implementation Details

### Reparameterization Trick

The encoder outputs `mean` and `log_var` for a Gaussian distribution:

```python
def encode_genomic(self, genomic):
    mean, log_var = self.encoder(genomic)
    # Stochastic sampling during training
    z = mean + exp(0.5 * log_var) * randn_like(log_var)
    cond = self.projection(z)
    return cond
```

**Why stochastic?** During training, the encoder learns a distribution over latent codes. This prevents mode collapse and encourages the latent space to be well-structured.

### Device Management

```python
def on_train_batch_start(self, batch, batch_idx):
    """Ensure encoder and projection are on correct device."""
    encoder_device = next(self.encoder.parameters()).device
    if encoder_device != self.device:
        self.encoder = self.encoder.to(self.device)
    proj_device = next(self.projection.parameters()).device
    if proj_device != self.device:
        self.projection = self.projection.to(self.device)
```

This is necessary because:
- Multi-GPU training can unexpectedly move modules
- Genomic encoder is not part of mopadi's standard setup
- Projection head must match UNet device for forward passes

### EMA (Exponential Moving Average)

```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    if self.is_last_accum(batch_idx):
        ema(self.model, self.ema_model, self.conf.ema_decay)
```

- **What**: Maintains an exponential running average of UNet weights
- **Why**: EMA model produces higher-quality samples during inference
- **Applied to**: UNet only (not encoder/projection, to save compute)
- **Decay**: 0.9999 (very slow update, mostly uses historical weights)

### Sample Visualization

During training, the model periodically generates sample images and logs them to TensorBoard:

```python
def on_train_batch_end(self, outputs, batch, batch_idx):
    # ... EMA update ...
    with torch.no_grad():
        genomic = batch['genomic'].to(self.device, dtype=torch.float32)
        cond = self.encode_genomic(genomic)
    self.log_sample(x_start=batch['img'], cond=cond)
    self.evaluate_scores()
```

**Frequency**: Every `reconstruct_every_samples` samples (default: 50,000)

---

## Configuration Parameters

Key parameters in `config.yaml` under `joint_training` section:

### Encoder & Projection
```yaml
latent_dim: 512                 # Latent space dimensionality
vae_hidden_dims: [2048, 1024]   # Hidden layer sizes (encoder only)
vae_dropout: 0.2                # Dropout rate in encoder
cond_dim: 512                   # UNet conditioning dimension (must match net_beatgans_embed_channels)
proj_hidden_dim: 512            # Projection head hidden size
proj_layers: 2                  # Number of projection layers
proj_dropout: 0.1               # Projection dropout
encoder_ckpt: null              # Optional: path to pre-trained encoder checkpoint
```

### Diffusion UNet
```yaml
img_size: 512                   # Image resolution
net_ch: 128                     # Base channel count
net_ch_mult: [1, 1, 2, 2, 4, 4] # Channel multipliers per layer
net_num_res_blocks: 2           # ResBlocks per resolution level
T: 1000                         # Diffusion timesteps
T_eval: 20                      # DDIM steps for sampling
sample_size: 8                  # Batch size for TensorBoard samples
diffusion_ckpt: null            # Optional: pre-trained UNet checkpoint
```

### Training
```yaml
batch_size: 2                   # Batch size
lr: 1e-4                        # Default learning rate
encoder_lr: 1e-4                # Encoder-specific learning rate
unet_lr: 1e-5                   # UNet learning rate (low if using pre-trained)
proj_lr: 3e-4                   # Projection head learning rate
optimizer: adam                 # adam | adamw
weight_decay: 0.0               # L2 regularization
warmup_steps: 500               # Learning rate warmup
grad_clip: 1.0                  # Gradient clipping threshold
ema_decay: 0.9999               # EMA decay for UNet
accumulate_grad_batches: 1      # Gradient accumulation steps
fp16: false                     # Mixed precision training
```

### Data & Scheduling
```yaml
csv_path: /path/to/gene_expression.csv
tiles_zip_dir: /path/to/image/tiles
val_fraction: 0.1               # Validation split
test_fraction: 0.1              # Held-out test split
epochs: 100                     # Max epochs
total_samples: 200000000        # Training duration (in samples)
steps_per_epoch: 5000           # Batches per epoch
save_every_samples: 200000      # Checkpoint frequency
reconstruct_every_samples: 50000 # Sample visualization frequency
```

---

## Training Workflow

```
Epoch Loop (max 100 epochs)
├─ Train Loop (5000 steps per epoch)
│  └─ For each batch:
│     ├─ Load batch (img, genomic)
│     ├─ encode_genomic(genomic) → cond
│     ├─ Sample timestep t and noise
│     ├─ Corrupt: x_t = √α·img + √(1-α)·noise
│     ├─ Predict: pred_noise = UNet(x_t, t, cond)
│     ├─ Loss: MSE(pred_noise, noise)
│     ├─ Backward: loss.backward() → updates all params
│     ├─ Log to TensorBoard: loss
│     └─ EMA update (if last accumulation batch)
│
├─ Validation Loop (after training epoch)
│  └─ For each val batch:
│     ├─ encode_genomic(genomic) → cond
│     ├─ Compute MSE diffusion loss
│     └─ Accumulate val_loss
│  └─ Log to TensorBoard: val_loss (epoch-averaged)
│
└─ End of epoch:
   ├─ Check val_loss trend (if early stopping enabled)
   ├─ Save checkpoint if time (every save_every_samples)
   └─ Generate & visualize samples if time (every reconstruct_every_samples)
```

---

## Checkpoint Management

### Saving Checkpoints

Checkpoints are saved to `experiments/joint_training/joint/` by default:

```
last.ckpt              ← Latest checkpoint (always updated)
epoch=5-step=25000.ckpt ← Saved every save_every_samples
```

### Loading Checkpoints

**Resume training**:
```python
trainer.fit(model, ckpt_path="experiments/joint_training/joint/last.ckpt")
```

**Load for inference**:
```python
from src.joint_training.model import JointLitModel
model = JointLitModel.load_from_checkpoint(
    "path/to/checkpoint.ckpt",
    conf=conf,
    joint_cfg=joint_cfg,
    n_genes=n_genes
)
model.eval()
```

### Checkpoint Contents

Each checkpoint contains:
- UNet weights
- Encoder weights
- Projection head weights
- EMA model weights
- Optimizer state
- Learning rate scheduler state
- Hyperparameters (logged via `save_hyperparameters`)

---

## Inference & Feature Extraction

### Generate Images from Genomics

```python
with torch.no_grad():
    genomic = torch.tensor([...])  # (1, n_genes)
    cond = model.encode(genomic)   # (1, cond_dim)
    noise = torch.randn(1, 3, 512, 512)
    generated_tiles = model.generate(genomic)  # (1, 3, 512, 512)
```

### Extract Latent Features

```python
from src.joint_training.train import extract_latents

latent_dir = extract_latents(
    joint_cfg,
    ckpt_path="experiments/joint_training/joint/last.ckpt",
    split="all"
)
```

This saves one `.h5` file per patient with encoder latent features.

---

## Recent Changes (March 2026)

### VAE Decoder Removal

The initial implementation used a full VAE with encoder + decoder and explicit reconstruction loss:

```python
# OLD: VAE with reconstruction loss
loss = diffusion_loss + lambda_vae * (recon_loss + beta_mmd * mmd_loss)
```

This was refactored to align with mopadi's proven approach:

```python
# NEW: Pure diffusion loss only
loss = diffusion_loss
```

**Benefits**:
- ✅ Simpler training (fewer hyperparameters)
- ✅ Proven to work (matches mopadi)
- ✅ Encoder learns through diffusion signal (no explicit reconstruction needed)

**Breaking Change**: Old checkpoints with VAE decoder cannot be loaded directly. To migrate:
1. Load old model with old code
2. Extract encoder: `old_model.vae.encoder.state_dict()`
3. Load into new model: `new_model.encoder.load_state_dict(...)`

---

## Files

- **model.py**: `JointLitModel` class, loss computation, training/validation steps
- **train.py**: Training loop, checkpoint management, latent feature extraction
- **dataset.py**: `GenomicTileDataset` for loading RNA-seq + image pairs
- **config.yaml**: All hyperparameter configuration (in parent directory)

---

## Quick Start

```bash
# Training
python run_pipeline.py --config src/config.yaml --stage joint_training

# Or directly
python -m src.joint_training.train --config src/config.yaml

# Monitor training
tensorboard --logdir experiments/joint_training/joint/

# Extract latents
python -c "
from src.joint_training.train import extract_latents
extract_latents({'csv_path': '...', 'tiles_zip_dir': '...'})
"
```

---

## Summary

✅ **Pure diffusion loss** (same as mopadi, no VAE reconstruction)  
✅ **All 3 components trained jointly** (Encoder, Projection, UNet)  
✅ **Logging to TensorBoard** (training loss, validation loss, learning rates)  
✅ **Validation implemented** (val_loss computed each epoch)  
✅ **EMA updates** (for improved sample quality)  
✅ **Sample visualization** (periodic generation to TensorBoard)  
❌ **Early stopping** (not built-in, but takes 6 lines to add)  

The encoder learns a **meaningful latent space through diffusion gradients**, not through explicit reconstruction objectives. This is simpler, more stable, and proven by mopadi's success in production.


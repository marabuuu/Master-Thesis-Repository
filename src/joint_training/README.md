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
```

For full architecture details and training dynamics, see [JOINT_TRAINING_SUMMARY.md](JOINT_TRAINING_SUMMARY.md) in the src/joint_training directory.

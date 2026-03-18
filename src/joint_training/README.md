# Joint Genomic VAE + Diffusion Training

## Overview

This system jointly trains two models in tandem:

1. **Genomic VAE**: Encodes bulk-RNA sequencing data (~19,000 genes) into a latent representation
2. **Diffusion UNet**: Generates histopathology tile images, conditioned on the VAE's latent encoding

The key innovation is that **the VAE latent representation directly conditions the diffusion model**, so the model learns to generate images that are consistent with genomic features.

---


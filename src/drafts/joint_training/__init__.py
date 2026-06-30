"""
Joint Genomic VAE + Diffusion Training Module

Trains a Variational Autoencoder on bulk-RNA sequencing data simultaneously
with a diffusion model on histopathology tiles. The VAE encodes high-dimensional
gene expression (~19,000 genes) into a compact latent representation that
conditions the diffusion model, replacing image-based feature extractors.

Architecture:
    RNA-seq (N_genes) → VAE Encoder → z (latent_dim) → ProjectionHead → cond (512)
    Tile + noise + cond → UNet → predicted noise
    Loss = L_diffusion + λ_vae * (L_recon + β * L_mmd)

Main components:
    - dataset: GenomicTileDataset (raw CSV + tile ZIPs)
    - model: JointGenomicDiffusionLit (PyTorch Lightning module)
    - train: Training entry point
"""

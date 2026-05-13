"""
Genomic cross-attention variant of MoPaDi training.

Differences from mopadi_genomic_crossattn (AdaGN-FiLM + CFL):
  1. Bottleneck cross-attention injected into the UNet after middle_block.
  2. Prediction-gap hinge loss — non-zero gradient at step 0, bootstraps conditioning.
  3. ema_decay=0.999 — corrected for 4-GPU global batch=16 (avoids ~700k-sample EMA warmup).
"""

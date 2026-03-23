# Cross-Attention Joint Training — Implementation Plan

Goal: variant of `joint_training` that strengthens genomic ↔ image coupling via multi-level cross-attention and robustness tricks (noise/conditioning dropout), while reusing mopadi/joint_training components by import (no copy-paste).

## 1) Directory & entrypoints
- Folder: `src/cross_attention_joint_training/`
- Reuse configs/checkpoint loading from `joint_training` via imports.
- New files: `model.py`, `train.py`, `dataset.py` (thin wrappers over existing ones), `config.py` (if needed for defaults), `__init__.py`.

## 2) Model changes (UNet cross-attention)
- Start from `joint_training.model` but swap UNet for a cross-attention-enabled variant that accepts conditioning at multiple scales.
- Introduce a `CrossAttentionUNetWrapper` that:
  - Accepts `cond` projected to multiple resolutions (e.g., 512 → {512, 256, 128}) via small MLPs.
  - Inserts cross-attn blocks at down, middle, and up blocks (matching mopadi net layout). Keep interfaces compatible with `sampler`.
  - Uses the same `TrainConfig` style; prefer minimal API change (e.g., `model.forward(x, t, cond_multi)`).
- Conditioning projection:
  - Keep existing `ProjectionHead` to 512, then derive per-level embeddings via 1-2 layer MLPs.
  - Option: positional tags per level (level id embedding) added to cond before cross-attn.

## 3) Robustness / information forcing
- **x_T dropout**: during training, randomly replace a fraction of batch images with pure noise before encoding; flag passed through to loss so model must rely on conditioning when image info is absent.
- **Genomic dropout**: randomly zero or mask parts of `cond` (or full cond) per step to prevent overreliance on image pathway.
- **Semantic noise emphasis**: optionally upweight loss on masked regions (drop patches) to force model to use conditioning to fill in.

## 4) Training loop adjustments
- Reuse `LitModel` training step, but:
  - Accept `cond_multi` and `cond_dropout_mask` in `model_kwargs` to UNet.
  - Hook for `x_T_dropout`: before forward noising, replace subset of `x_start` with `torch.randn_like`.
  - Hook for `cond_dropout`: randomly zero cond or apply feature dropout.
- Ensure EMA, optimizer, scheduler remain identical to mopadi for stability.

## 5) Dataloading
- Reuse `joint_training.dataset`; add optional flags for:
  - Patient-level genomic dropout probability.
  - Patch dropout masks (if spatial masking used later).

## 6) Config additions (minimal deltas)
- `cross_attention: { num_levels: 3, heads: 4, dim_per_head: 64, cond_dims: [512,256,128] }`
- `xT_dropout_prob: 0.1` (probability to replace x_start with noise)
- `cond_dropout_prob: 0.1` (probability to zero conditioning)
- `cond_feature_dropout: 0.05` (per-feature dropout inside cond)
- Optional: `mask_loss_weight` if adding masked region loss.

## 7) Inference behavior
- Default: no dropouts; pass full cond.
- Allow ablation flag to zero cond at inference for debugging reliance.

## 8) Tasks to implement
- [ ] Add configs and CLI flags for dropout/cross-attn params.
- [ ] Implement `CrossAttentionUNetWrapper` adapting existing UNet (import, wrap forward to inject cross-attn blocks).
- [ ] Extend projection head to multi-level cond heads.
- [ ] Add x_T dropout + cond dropout hooks in training_step.
- [ ] Wire model_kwargs to sampler so UNet receives cond per level.
- [ ] Update dataset/config plumbs; reuse existing loaders.
- [ ] Add minimal tests: forward pass with/without dropouts; zero-cond ablation produces different outputs than full-cond.

Notes:
- Keep imports from `joint_training`/`mopadi` to avoid duplication.
- Preserve optimizer/EMA/scheduler from mopadi to stay close to baseline.
- Cross-attn insertion should mirror mopadi block structure to minimize risk.

## 9) Risks and mitigations
- UNet wiring mismatch: start by inserting cross-attn only at mid + one down block; expand after verifying shapes/VRAM. Mirror mopadi ch_mult/num_res_blocks to avoid silent shape bugs.
- Sampler interface drift: keep UNet signature compatible; pass `cond_multi` via `model_kwargs` in a wrapper without changing sampler internals.
- Training instability: keep conservative defaults (x_T_dropout_prob≈0.05–0.1, cond_dropout_prob≈0.05 with occasional full-zero batches). Optionally warm-up dropout after a few epochs.
- Over-regularization collapse: avoid stacking strong x_T and cond dropout simultaneously; add a cap or schedule. Monitor recon diversity early.
- Memory/runtime: cross-attn per level increases VRAM; begin with fewer heads/dims and limited levels; profile before scaling.
- Checkpoint metadata: ensure save_hyperparameters logs new config keys so no patching is needed later.
- Validation hooks: keep zero-cond inference toggle to confirm conditioning is actually used; add a quick unit test for forward with/without cond.

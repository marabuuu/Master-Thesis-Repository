# Gene-Token + Cross-Attention Joint Training (Implemented Behavior)

This folder contains the implemented hybrid training variant used in the project.

- `model.py`: defines `GeneTokenCrossAttentionJointLitModel`
- `train.py`: defines training entrypoint and config merge behavior
- `__init__.py`: exports the model class and config builder

The implementation is intentionally thin and reuses existing modules:

- gene-token encoder path from `gene_token_transformer_joint_training`
- cross-attention UNet wrapper from `cross_attention_joint_training`
- diffusion training/scaffolding inherited from the existing joint model stack

---

## What the model actually is

`GeneTokenCrossAttentionJointLitModel` inherits from `GeneTokenTransformerJointLitModel`.

That means this hybrid keeps the full gene-token genomic encoding path from the parent class and then only changes the image-side conditioning interface.

In `__init__(conf, joint_cfg, n_genes)`:

1. Read cross-attention parameters from `joint_cfg["cross_attention"]`:
  - `heads` (default `4`)
  - `dim_per_head` (default `64`)
  - `cond_dims` (default `[512, 256, 128]`)

2. Call `super().__init__(...)` first:
  - builds the gene-token transformer encoder + projection path
  - initializes diffusion/sampler state via inherited joint model code

3. Validate conditioning shape compatibility:
  - `cond_dims[0]` must equal `cond_dim` (`joint_cfg["cond_dim"]` or `conf.feat_dim`)
  - raises `ValueError` on mismatch

4. Build `self.cross_cfg` with training-time forcing knobs:
  - `xT_dropout_prob`
  - `cond_dropout_prob`
  - `cond_feature_dropout`

5. Replace the UNet with a wrapper:
  - `self.model = CrossAttentionUNetWrapper(self.model, ...)`

6. Save extra checkpoint metadata:
  - `cross_cfg`
  - `joint_variant = "gene_token_cross_attention_joint_training"`

---

## Exact training-step behavior

`training_step` in this module overrides parent behavior.

For each batch:

1. Load tensors:
  - `imgs = batch["img"]`
  - `genomic = batch["genomic"]`

2. Encode genomics using inherited gene-token path:
  - `cond = self.encode_genomic(genomic)`

3. Apply conditioning dropout (`cond_dropout_prob`):
  - randomly zero whole conditioning vectors per sample

4. Apply feature dropout (`cond_feature_dropout`):
  - `F.dropout` over conditioning features

5. Build multi-scale conditioning for cross-attention:
  - `cond_multi = self.model.make_cond_multi(cond)`

6. Apply `xT` dropout (`xT_dropout_prob`):
  - randomly replace selected `x_start` samples with Gaussian noise

7. Sample timesteps and compute diffusion loss:
  - `t, _ = self.T_sampler.sample(...)`
  - call `self.sampler.training_losses(...)`
  - pass `model_kwargs={"cond": cond, "cond_multi": cond_multi}`
  - optimize `loss = losses["loss"].mean()`

8. Logging:
  - Lightning logs: `loss_epoch`, `loss_step`
  - TensorBoard scalar: `loss` at `self.num_samples` (rank 0 only)

---

## EMA update detail (important)

Because `self.model` is wrapped, EMA cannot be updated against the wrapper object directly.

`on_train_batch_end` therefore calls:

- `ema(self.model.base_unet, self.ema_model, self.conf.ema_decay)`

and only when `self.is_last_accum(batch_idx)` is true.

This mirrors cross-attention behavior and keeps EMA weights aligned with the underlying UNet.

---

## Config builder used by this module

`build_gene_token_cross_attention_conf(joint_cfg)` simply forwards to:

- `build_gene_token_transformer_conf(joint_cfg)`

So diffusion/training defaults come from the gene-token configuration builder; this module does not define a separate diffusion config class.

---

## Training entrypoint behavior (`train.py`)

`run_gene_token_cross_attention_training(joint_cfg, verbose=True)` does the following:

1. Seed everything with `joint_cfg.get("seed", 42)`.
2. Build `conf` via `build_gene_token_cross_attention_conf(joint_cfg)`.
3. Infer `n_genes` via `_count_genes(joint_cfg)`.
4. Instantiate `GeneTokenCrossAttentionJointLitModel(conf, joint_cfg, n_genes)`.
5. Configure checkpointing:
  - `save_last=True`
  - `save_top_k` from config (default `3`)
  - filename pattern `epoch{epoch:03d}-step{step:08d}`
  - step interval from `conf.save_every_samples // conf.batch_size_effective` (clamped at `>=1`)
6. Resume automatically from `last.ckpt` if present in `conf.logdir`.
7. Create TensorBoard logger writing directly into `conf.logdir`.
8. Use DDP strategy with `find_unused_parameters=True` if multi-GPU.
9. Build Lightning trainer with:
  - `check_val_every_n_epoch`
  - `limit_val_batches`
  - `num_sanity_val_steps=0`
  - mixed precision if `conf.fp16`
10. Call `trainer.fit(model, ckpt_path=ckpt_path)`.

---

## How config is merged in `main()`

`main()` loads YAML and merges two sections:

- base: `gene_token_transformer_joint_training` (fallback to `joint_training`)
- overrides: `gene_token_cross_attention_joint_training`

Merge is recursive via `_deep_update(base, overrides)`.

If override section does not provide `out_dir`, it auto-sets:

- `out_dir = f"{base_out_dir}_cross"`

This means you can keep the hybrid section minimal and only set cross-attention-specific differences.

---

## What this module does not add

This module does not introduce a new reconstruction/inversion pipeline by itself.

- It changes how RNA conditioning is encoded + injected during training.
- Reconstruction behavior is handled in the reconstruction scripts.

For inference/reconstruction, checkpoints from this module are identifiable via `joint_variant = "gene_token_cross_attention_joint_training"` in checkpoint hyperparameters.

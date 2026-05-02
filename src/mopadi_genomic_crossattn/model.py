"""
GenomicCrossAttnLitModel — MoPaDi genomic training with patchified cross-attention.

Extends GenomicLitModel with two additions:

1. CrossAttentionUNetWrapper: wraps self.model so that, before each UNet
   forward pass, image patches attend to the genomic conditioning vector as a
   single K/V token and a zero-initialised residual is added to the input.
   Style conditioning (AdaGN) is unchanged and provides the global genomic
   signal as before; the cross-attention layer adds spatial specificity on top.

2. Genomic-guided high-t loss: in addition to the standard diffusion loss
   (L1, computed by the parent at uniformly sampled t), a second forward pass
   at t ∈ [high_t_frac·T, T) is computed and added as λ·L2.  At these high
   timesteps x_t ≈ N(0,I) so the model cannot rely on image content and must
   use the genomic conditioning to predict the noise direction.  This creates
   an explicit genomic-guided learning regime.

Total loss = L1 + λ·L2, with λ = conf.genomic_guided_loss_weight.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F
from typing import Any, NamedTuple, TYPE_CHECKING, cast, Optional

if TYPE_CHECKING:
    from mopadi_genomic_crossattn.genomic_train import GenomicLitModel
    from mopadi_genomic_crossattn.config import GenomicCrossAttnConfig
    from mopadi.model.nn import timestep_embedding
    from mopadi.model.unet import BeatGANsUNetModel
else:
    from mopadi_genomic_crossattn.genomic_train import GenomicLitModel
    from mopadi_genomic_crossattn.config import GenomicCrossAttnConfig
    from mopadi.model.nn import timestep_embedding

log = logging.getLogger(__name__)


class Return(NamedTuple):
    pred: torch.Tensor
    features: Optional[torch.Tensor] = None


class _SpatialCrossAttentionAdapter(torch.nn.Module):
    """Inject genomic conditioning into a single UNet stage."""

    def __init__(self, in_channels: int, cond_dim: int, pool_size: int, heads: int, dim_head: int):
        super().__init__()
        embed_dim = heads * dim_head
        self.pool_size = int(pool_size)
        self.embed_dim = embed_dim

        self.in_proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.k_proj = torch.nn.Linear(cond_dim, embed_dim)
        self.v_proj = torch.nn.Linear(cond_dim, embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)
        self.out_proj = torch.nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        torch.nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            torch.nn.init.zeros_(self.out_proj.bias)
        self.heads = heads
        self._attn_stats = {}  # for logging attention statistics
        # multiplicative FiLM branch: predict per-channel scale delta
        self.scale_proj = torch.nn.Conv2d(embed_dim, in_channels, kernel_size=1)
        torch.nn.init.zeros_(self.scale_proj.weight)
        if self.scale_proj.bias is not None:
            torch.nn.init.zeros_(self.scale_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        cond = cond.to(device=x.device, dtype=x.dtype)

        pooled_h = min(h, self.pool_size)
        pooled_w = min(w, self.pool_size)
        pooled = x if (pooled_h == h and pooled_w == w) else F.adaptive_avg_pool2d(x, (pooled_h, pooled_w))

        tokens = self.in_proj(pooled).flatten(2).transpose(1, 2)
        k = self.k_proj(cond).unsqueeze(1)
        v = self.v_proj(cond).unsqueeze(1)
        # Capture attention output and attention weights for diagnostics.
        # attn_weights shape depends on PyTorch version and `batch_first`:
        # typically (B, tgt_len, src_len). With a single K/V token src_len==1.
        attn_out, attn_weights = self.attn(tokens, k, v)

        attn_map = attn_out.transpose(1, 2).reshape(b, self.embed_dim, pooled_h, pooled_w)
        # additive residual (spatial)
        delta = self.out_proj(attn_map)
        # multiplicative FiLM: predict a small per-channel delta initialized to zero
        scale_delta = self.scale_proj(attn_map)
        
        # Log attention output and FiLM modulation stats (no grad).
        # Note: with a single K/V token (the genomic vector), PyTorch's
        # MultiheadAttention always assigns weight 1.0 to that token — softmax
        # of a single score is always 1.  The informative quantities are the
        # magnitude of attn_out (= V_proj(cond)) and the resulting FiLM
        # residuals delta / scale_delta that actually modulate the UNet features.
        with torch.no_grad():
            # attn_out: (B, N_patches, embed_dim) — spatially uniform (same V at each patch)
            attn_norm = attn_out.norm(dim=-1)          # (B, N_patches)
            stats = {
                "attn_out_norm_mean": attn_norm.mean().item(),
                "attn_out_norm_std": attn_norm.std().item() if attn_norm.numel() > 1 else 0.0,
                "scale_delta_abs_mean": scale_delta.abs().mean().item(),
                "scale_delta_abs_max": scale_delta.abs().max().item(),
                "delta_abs_mean": delta.abs().mean().item() if delta.shape[2:] == (h, w) else 0.0,
                "delta_abs_max": delta.abs().max().item() if delta.shape[2:] == (h, w) else 0.0,
            }

            # If attention weights are available, aggregate simple diagnostics.
            try:
                if attn_weights is not None:
                    # attn_weights may be (B, tgt_len, src_len) or (B, num_heads, tgt_len, src_len)
                    aw = attn_weights
                    if aw.dim() == 4:
                        # (B, heads, tgt, src) -> merge heads
                        aw = aw.mean(dim=1)
                    # now (B, tgt, src)
                    stats["attn_weights_mean"] = aw.mean().item()
                    stats["attn_weights_std"] = aw.std().item() if aw.numel() > 1 else 0.0
                    # record max weight (useful to see concentration)
                    stats["attn_weights_max"] = aw.max().item()
            except Exception:
                # Be conservative: don't fail forward on diagnostics
                pass

            self._attn_stats = stats
        
        # ensure same spatial resolution as input
        if delta.shape[2:] != (h, w):
            delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)
        if scale_delta.shape[2:] != (h, w):
            scale_delta = F.interpolate(scale_delta, size=(h, w), mode="bilinear", align_corners=False)
        # apply multiplicative gating: x * (1 + scale_delta) + additive residual
        return x * (1.0 + scale_delta) + delta


class MultiStageCrossAttentionUNetWrapper(torch.nn.Module):
    """Wrap the base UNet with cross-attention at multiple depths."""

    def __init__(
        self,
        base_unet: torch.nn.Module,
        cond_dim: int,
        pool_size: int = 16,
        heads: int = 4,
        dim_head: int = 64,
    ):
        super().__init__()
        self.base_unet: torch.nn.Module = base_unet
        self.pool_size = int(pool_size)

        conf = getattr(base_unet, "conf", None)
        if conf is None:
            raise ValueError("Base UNet must expose a `conf` attribute for multi-stage cross-attention.")

        channel_mult_raw = getattr(conf, "channel_mult", None)
        if channel_mult_raw is None:
            channel_mult_raw = getattr(conf, "net_ch_mult", None)
        if channel_mult_raw is None:
            raise ValueError("Base UNet config must define `channel_mult` or `net_ch_mult`.")

        channel_mult = tuple(channel_mult_raw)
        if not channel_mult:
            raise ValueError("Base UNet config must define a non-empty channel multiplier list.")

        input_mult = tuple(getattr(conf, "input_channel_mult", None) or channel_mult)
        if len(input_mult) != len(channel_mult):
            raise ValueError(
                "input_channel_mult and channel_mult/net_ch_mult must have the same length for multi-stage cross-attention."
            )

        self.input_adapter = _SpatialCrossAttentionAdapter(
            in_channels=3,
            cond_dim=cond_dim,
            pool_size=self.pool_size,
            heads=heads,
            dim_head=dim_head,
        )

        self.encoder_adapters = torch.nn.ModuleList([
            _SpatialCrossAttentionAdapter(
                in_channels=int(mult * conf.model_channels),
                cond_dim=cond_dim,
                pool_size=self.pool_size,
                heads=heads,
                dim_head=dim_head,
            )
            for mult in input_mult
        ])

        bottleneck_channels = int(channel_mult[-1] * conf.model_channels)
        self.middle_adapter = _SpatialCrossAttentionAdapter(
            in_channels=bottleneck_channels,
            cond_dim=cond_dim,
            pool_size=self.pool_size,
            heads=heads,
            dim_head=dim_head,
        )

        self.decoder_adapters = torch.nn.ModuleList([
            _SpatialCrossAttentionAdapter(
                in_channels=int(mult * conf.model_channels),
                cond_dim=cond_dim,
                pool_size=self.pool_size,
                heads=heads,
                dim_head=dim_head,
            )
            for mult in list(channel_mult)[::-1]
        ])

        # --- Local per-stage cond projection (scale+shift) ---
        # For each ResBlock we replace or attach a cond_emb_layers that maps
        # the raw `cond` vector -> 2*out_channels so the condition can act
        # as scale+shift (FiLM). We zero-init the linear so initial behaviour
        # is identity (scale_delta=0, shift=0).
        try:
            from mopadi.model.blocks import ResBlock
        except Exception:
            ResBlock = None

        def _patch_block(block, cond_dim: int):
            # block may be a TimestepEmbedSequential or ResBlock
            if hasattr(block, "__iter__") and not isinstance(block, torch.nn.ModuleList):
                for layer in block:
                    _patch_block(layer, cond_dim)
                return
            if ResBlock is not None and isinstance(block, ResBlock):
                # ensure two_cond path is enabled so cond_emb_layers is actually called
                try:
                    block.conf.two_cond = True
                except Exception:
                    log.warning(
                        "Could not set two_cond=True on ResBlock (frozen config?); "
                        "cond_emb_layers will be replaced but NOT called during forward."
                    )
                out_ch = int(getattr(block.conf, "out_channels", block.conf.channels))
                # create new cond_emb_layers mapping cond_dim -> 2*out_ch
                linear_layer = torch.nn.Linear(cond_dim, 2 * out_ch)
                torch.nn.init.zeros_(linear_layer.weight)
                if linear_layer.bias is not None:
                    torch.nn.init.zeros_(linear_layer.bias)
                block.cond_emb_layers = torch.nn.Sequential(torch.nn.SiLU(), linear_layer)

        # Patch input_blocks, middle_block, and output_blocks in-place
        # and count how many ResBlocks were successfully patched.
        patched_count = 0
        
        def count_patched_resblocks(module):
            nonlocal patched_count
            if ResBlock is not None and isinstance(module, ResBlock):
                if hasattr(module, "cond_emb_layers") and module.cond_emb_layers is not None:
                    patched_count += 1
            elif hasattr(module, "__iter__") and not isinstance(module, torch.nn.ModuleList):
                for layer in module:
                    count_patched_resblocks(layer)
        
        for module in list(getattr(base_unet, "input_blocks", [])):
            _patch_block(module, cond_dim)
        if hasattr(base_unet, "middle_block"):
            _patch_block(base_unet.middle_block, cond_dim)
        for module in list(getattr(base_unet, "output_blocks", [])):
            _patch_block(module, cond_dim)
        
        # Verify patching succeeded by counting patched blocks
        for module in list(getattr(base_unet, "input_blocks", [])):
            count_patched_resblocks(module)
        if hasattr(base_unet, "middle_block"):
            count_patched_resblocks(base_unet.middle_block)
        for module in list(getattr(base_unet, "output_blocks", [])):
            count_patched_resblocks(module)
        
        if patched_count == 0:
            log.warning(
                "No ResBlocks were patched with cond_emb_layers! "
                "Check that mopadi.model.blocks.ResBlock is available and FiLM conditioning may not be active."
            )
        else:
            log.info(f"Successfully patched {patched_count} ResBlocks with 2*out_channels cond_emb_layers.")

    def wrapper_parameters(self):
        yield from self.input_adapter.parameters()
        yield from self.encoder_adapters.parameters()
        yield from self.middle_adapter.parameters()
        yield from self.decoder_adapters.parameters()

    def get_attention_stats(self):
        """Collect attention statistics from all adapters for logging."""
        stats = {}
        for i, adapter in enumerate(self.encoder_adapters):
            if hasattr(adapter, "_attn_stats") and isinstance(adapter._attn_stats, dict):
                for key, val in adapter._attn_stats.items():
                    stats[f"adapter_enc_{i}_{key}"] = val
        if hasattr(self.middle_adapter, "_attn_stats") and isinstance(self.middle_adapter._attn_stats, dict):
            for key, val in self.middle_adapter._attn_stats.items():
                stats[f"adapter_mid_{key}"] = val
        for i, adapter in enumerate(self.decoder_adapters):
            if hasattr(adapter, "_attn_stats") and isinstance(adapter._attn_stats, dict):
                for key, val in adapter._attn_stats.items():
                    stats[f"adapter_dec_{i}_{key}"] = val
        return stats

    def forward(self, x: torch.Tensor, t: torch.Tensor, *, cond: torch.Tensor, **kwargs):
        # optional flag: when True return pooled middle features for recon losses
        return_features = bool(kwargs.pop("return_features", False))
        base_unet = cast("BeatGANsUNetModel", self.base_unet)
        time_emb = timestep_embedding(t, base_unet.time_emb_channels)
        time_embed_module = base_unet.time_embed
        if hasattr(time_embed_module, "forward") and not isinstance(time_embed_module, torch.nn.Sequential):
            time_res = time_embed_module.forward(time_emb=time_emb, cond=cond, time_cond_emb=time_emb)
            emb = time_res.time_emb if getattr(time_res, "time_emb", None) is not None else time_res.emb
            cond_emb = time_res.emb if getattr(time_res, "emb", None) is not None else cond
        else:
            emb = time_embed_module(time_emb)
            cond_emb = cond

        h = x.to(dtype=getattr(base_unet, "dtype", x.dtype))
        h = self.input_adapter(h, cond)

        input_num_blocks = list(base_unet.input_num_blocks)
        output_num_blocks = list(base_unet.output_num_blocks)
        input_blocks = list(base_unet.input_blocks)
        output_blocks = list(base_unet.output_blocks)

        hs: list[list[torch.Tensor]] = [[] for _ in range(len(input_num_blocks))]
        input_block_idx = 0
        for level in range(len(input_num_blocks)):
            for _ in range(input_num_blocks[level]):
                input_block = cast(Any, input_blocks[input_block_idx])
                h = input_block(h, emb=emb, cond=cond_emb)
                hs[level].append(h)
                input_block_idx += 1
            h = self.encoder_adapters[level](h, cond)

        middle_block = cast(Any, base_unet.middle_block)
        h = middle_block(h, emb=emb, cond=cond_emb)
        features = None
        if return_features:
            # global-pool spatial middle features -> (B, C)
            features = F.adaptive_avg_pool2d(h, (1, 1)).view(h.shape[0], -1)
        h = self.middle_adapter(h, cond)

        output_block_idx = 0
        for level in range(len(output_num_blocks)):
            for _ in range(output_num_blocks[level]):
                try:
                    lateral = hs[-level - 1].pop()
                except IndexError:
                    lateral = None
                output_block = cast(Any, output_blocks[output_block_idx])
                h = output_block(h, emb=emb, cond=cond_emb, lateral=lateral)
                output_block_idx += 1
            h = self.decoder_adapters[level](h, cond)

        h = h.type(x.dtype)
        pred = base_unet.out(h)
        return Return(pred=pred, features=features)


class GenomicCrossAttnLitModel(GenomicLitModel):
    """MoPaDi genomic diffusion model with patchified cross-attention + dual loss.

    Inherits from GenomicLitModel:
      - setup(): creates ZipTilesWithGenomicFeatures datasets (unchanged)
      - val_dataloader(): validation loader with batch cap (unchanged)
      - validation_step(): logs loss/val, loss/val_shuffled, cond/gap (unchanged)
      - on_validation_epoch_end(): per-epoch validation summary (unchanged)
      - on_fit_start(): sanity check on ZIP dataset (unchanged)
      - evaluate_scores(): no-op (unchanged)
      - log_sample(): skip at num_samples==0 (unchanged)

    Overrides:
      - __init__: wraps self.model with CrossAttentionUNetWrapper, re-inits EMA
      - configure_optimizers: splits params into UNet / cross-attn LR groups
      - training_step: parent L1 + genomic-guided high-t L2
    """

    model: Any

    def __init__(self, conf: GenomicCrossAttnConfig):
        super().__init__(conf)
        self.conf = conf

        # Wrap the base UNet with multi-stage cross-attention.  Each stage is
        # zero-initialised so training starts from the identity and the genomic
        # path can be learned gradually without destabilising the UNet.
        self.model = MultiStageCrossAttentionUNetWrapper(
            base_unet=self.model,
            cond_dim=conf.style_ch,
            pool_size=conf.cross_attn_patch_size,
            heads=conf.cross_attn_heads,
            dim_head=conf.cross_attn_dim_per_head,
        )

        # Re-initialise EMA to track the full wrapped model (base_unet +
        # cross-attn layers).  Without this, EMA would point to the unwrapped
        # model and the on_train_batch_end EMA update would fail silently.
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        # Exclude EMA from DDP's initial parameter broadcast and gradient buckets.
        # EMA starts as a deepcopy of self.model, which DDP already syncs from
        # rank 0 — re-broadcasting 139 M extra params is redundant and triggers
        # a CUDA illegal-memory-access crash during _sync_params_and_buffers on
        # multi-GPU setups.  ema_model remains a registered submodule so PL
        # still includes it in checkpoints automatically.
        self._ddp_params_and_buffers_to_ignore = [
            f"ema_model.{n}" for n, _ in self.ema_model.named_parameters()
        ] + [
            f"ema_model.{n}" for n, _ in self.ema_model.named_buffers()
        ]

        # Reconstruction head: map pooled middle features -> genomic feats
        # Uses the wrapped model's middle adapter input channel size to
        # determine the pooled feature dimension.
        try:
            recon_in = int(self.model.middle_adapter.in_proj.in_channels)
        except Exception:
            # fallback to style_ch if inspection fails
            recon_in = int(getattr(conf, "style_ch", getattr(conf, "feat_dim", 512)))
        self.recon_head = torch.nn.Sequential(
            torch.nn.Linear(recon_in, int(getattr(conf, "feat_dim", conf.style_ch)))
        )

        log.info(
            "MultiStageCrossAttentionUNetWrapper applied: pool_size=%d, heads=%d, dim_per_head=%d",
            conf.cross_attn_patch_size,
            conf.cross_attn_heads,
            conf.cross_attn_dim_per_head,
        )

    # ------------------------------------------------------------------
    # Optimizer: separate LR groups for UNet and cross-attention wrapper
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Split parameters into two LR groups.

        UNet params: trained at unet_lr.  When training from scratch this
        equals conf.lr; reduce it only when warm-starting from a pre-trained
        diffusion checkpoint.

        Wrapper params (patch_proj / k_proj / v_proj / attn / out_proj):
        trained at cross_attn_lr.  These layers are zero-initialised and need
        a comparable or higher LR to learn meaningful attention patterns within
        a reasonable number of steps.
        """
        unet_lr = float(getattr(self.conf, "unet_lr", self.conf.lr))
        cross_attn_lr = float(getattr(self.conf, "cross_attn_lr", self.conf.lr))

        param_groups = [
            {"params": list(self.model.base_unet.parameters()) + list(self.recon_head.parameters()), "lr": unet_lr},
            {"params": list(self.model.wrapper_parameters()), "lr": cross_attn_lr},
        ]

        optim = torch.optim.Adam(param_groups, weight_decay=self.conf.weight_decay)
        return {"optimizer": optim}

    def _compute_genomic_guided_loss(
        self,
        model: Any,
        imgs: torch.Tensor,
        feats: torch.Tensor,
        bag_n: int = 1,
    ) -> torch.Tensor:
        """Compute the high-t genomic-guided denoising loss."""
        high_t_frac = float(getattr(self.conf, "genomic_guided_high_t_frac", 0.8))
        T = self.conf.T
        t_lo = int(high_t_frac * T)
        # imgs is expected to be flat (B*N, C, H, W). If bag_n > 1,
        # aggregate the per-tile losses into per-bag means before averaging
        # to compute the genomic-guided loss at the bag level.
        total_tiles = imgs.shape[0]
        high_t = torch.randint(t_lo, T, (total_tiles,), device=imgs.device)

        losses_high_t = self.sampler.training_losses(
            model=model,
            x_start=imgs,
            cond=feats,
            t=high_t,
            model_kwargs={"cond": feats},
        )
        per_tile = losses_high_t["loss"]  # (B*N,)

        if bag_n is None or bag_n <= 1:
            return per_tile.mean()

        # bag_n > 1: reshape to (B, N) where B = total_tiles // bag_n
        if total_tiles % bag_n != 0:
            # fall back to flat mean if shapes don't align
            return per_tile.mean()
        B = total_tiles // bag_n
        per_bag = per_tile.view(B, bag_n).mean(dim=1)
        return per_bag.mean()

    # ------------------------------------------------------------------
    # Training step: L1 (standard diffusion) + L2 (genomic-guided high-t)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """Compute dual loss: standard diffusion (L1) + high-t genomic (L2).

        L1 is computed by GenomicLitModel.training_step at uniformly sampled t,
        logged as loss/train.  L2 is an additional forward pass at t ∈ [high_t, T)
        where x_t ≈ N(0,I) — the model cannot exploit image structure and must
        use the genomic conditioning to predict noise.  Logged as loss/genomic_guided.

        Total backprop loss = L1 + λ·L2, with λ = genomic_guided_loss_weight.
        """
        # Flatten bag batches before any processing so that super() sees
        # (B*N, C, H, W) images. GenomicLitModel.training_step does the same
        # check, but doing it here first avoids double-flattening issues.
        batch, _bag_n = self._flatten_bag_batch(batch)

        # L1: standard diffusion loss at uniform t (image-guided regime)
        out = super().training_step(batch, batch_idx)

        genomic_weight = float(getattr(self.conf, "genomic_guided_loss_weight", 0.0))
        if genomic_weight <= 0.0:
            return out

        # Always extract imgs and feats for reconstruction loss (even if high-t is skipped)
        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)
        loss_l1 = out["loss"] if isinstance(out, dict) else out
        total_loss = loss_l1

        # L2: denoising at high t (genomic-guided regime).
        # Compute both conditioned and shuffled high-t losses using the
        # same timesteps and noise so we can form a counterfactual gap at
        # high t. Aggregate per-bag when bag mode is active.
        # Skip during training if compute_high_t_loss_during_training=False (default).
        compute_high_t = bool(getattr(self.conf, "compute_high_t_loss_during_training", False))
        
        if compute_high_t:
            # draw high timesteps and shared noise for fair comparison
            high_t_frac = float(getattr(self.conf, "genomic_guided_high_t_frac", 0.8))
            T = self.conf.T
            t_lo = int(high_t_frac * T)
            total_tiles = imgs.shape[0]
            high_t = torch.randint(t_lo, T, (total_tiles,), device=imgs.device)
            shared_noise = torch.randn_like(imgs)

            losses_cond = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=feats,
                t=high_t,
                noise=shared_noise,
                model_kwargs={"cond": feats},
            )
            per_tile_cond = losses_cond["loss"]  # (B*N,)

            # Shuffle feats at patient (bag) level for comparison
            if _bag_n > 1:
                B_bags = feats.shape[0] // _bag_n
                bag_perm = self._non_identity_permutation(B_bags, feats.device)
                feats_shuffled = (
                    feats.reshape(B_bags, _bag_n, -1)[bag_perm]
                    .reshape(B_bags * _bag_n, -1)
                )
            else:
                perm = self._non_identity_permutation(feats.size(0), feats.device)
                feats_shuffled = feats[perm]

            losses_shuffled = self.sampler.training_losses(
                model=self.model,
                x_start=imgs,
                cond=feats_shuffled,
                t=high_t,
                noise=shared_noise,
                model_kwargs={"cond": feats_shuffled},
            )
            per_tile_shuffled = losses_shuffled["loss"]

            # Aggregate per-bag if requested
            def _aggregate_per_bag(per_tile_loss: torch.Tensor, bag_n: int) -> torch.Tensor:
                if bag_n is None or bag_n <= 1:
                    return per_tile_loss.mean()
                if per_tile_loss.numel() % bag_n != 0:
                    return per_tile_loss.mean()
                B = per_tile_loss.numel() // bag_n
                return per_tile_loss.view(B, bag_n).mean(dim=1).mean()

            genomic_loss = _aggregate_per_bag(per_tile_cond, _bag_n)
            genomic_loss_shuffled = _aggregate_per_bag(per_tile_shuffled, _bag_n)

            # Counterfactual gap at high-t (encourages using cond at high noise)
            cf_weight = float(getattr(self.conf, "counterfactual_loss_weight", 0.0))
            cf_temperature = max(1e-6, float(getattr(self.conf, "counterfactual_temperature", 0.05)))
            gap = genomic_loss_shuffled - genomic_loss
            cf_penalty_high_t = F.softplus(-gap / cf_temperature)

            total_loss = loss_l1 + genomic_weight * genomic_loss + cf_weight * cf_penalty_high_t

            if isinstance(out, dict):
                out["loss"] = total_loss
            else:
                out = total_loss

            self.log(
                "loss/genomic_guided",
                genomic_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "loss/genomic_train",
                genomic_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "loss/genomic_guided_gap",
                gap,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "loss/genomic_cf_high_t",
                cf_penalty_high_t,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        # --- Optional genomic reconstruction loss (from pooled middle features)
        recon_weight = float(getattr(self.conf, "genomic_recon_weight", 0.0))
        if recon_weight > 0.0:
            # forward pass with return_features to get pooled middle features
            zero_t = torch.zeros(len(imgs), dtype=torch.long, device=imgs.device)
            try:
                scaled_zero_t = self.sampler._scale_timesteps(zero_t)
            except Exception:
                scaled_zero_t = zero_t
            model_forward = self.model.forward(x=imgs, t=scaled_zero_t, cond=feats, return_features=True)
            features = model_forward.features
            if features is not None:
                recon_pred = self.recon_head(features)
                recon_loss = F.mse_loss(recon_pred, feats)
                total_loss = total_loss + recon_weight * recon_loss
                if isinstance(out, dict):
                    out["loss"] = total_loss
                else:
                    out = total_loss

                self.log(
                    "loss/genomic_recon",
                    recon_loss,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                )
                self.log(
                    "loss/genomic_recon_train",
                    recon_loss,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                )

        # --- Log attention statistics periodically (every ~100 steps)
        if self.global_step % 100 == 0:
            attn_stats = self.model.get_attention_stats()
            if self.global_rank == 0:
                logger = getattr(self, "logger", None)
                if logger is not None and hasattr(logger, "experiment"):
                    for key, val in attn_stats.items():
                        logger.experiment.add_scalar(f"attn/{key}", val, self.num_samples)
            for key, val in attn_stats.items():
                self.log(f"attn/{key}", val, on_step=True, on_epoch=False, prog_bar=False, logger=False)

        return out

    def validation_step(self, batch, batch_idx):
        """Extend validation with an explicit genomic-guided validation loss."""
        # Val dataset always uses n_tiles_per_bag=1, but guard for safety.
        batch, _bag_n = self._flatten_bag_batch(batch)
        loss_cond = super().validation_step(batch, batch_idx)
        if loss_cond is None or self.trainer.sanity_checking:
            return loss_cond

        imgs = batch["img"].to(self.device)
        feats = batch["feat"].to(self.device, dtype=torch.float32)
        with torch.no_grad():
            genomic_val_loss = self._compute_genomic_guided_loss(self.ema_model, imgs, feats, bag_n=_bag_n)

        # genomic reconstruction on EMA model
        recon_val_loss = None
        recon_weight = float(getattr(self.conf, "genomic_recon_weight", 0.0))
        if recon_weight > 0.0:
            zero_t = torch.zeros(len(imgs), dtype=torch.long, device=imgs.device)
            try:
                scaled_zero_t = self.sampler._scale_timesteps(zero_t)
            except Exception:
                scaled_zero_t = zero_t
            with torch.no_grad():
                mf = self.ema_model.forward(x=imgs, t=scaled_zero_t, cond=feats, return_features=True)
                features = mf.features
                if features is not None:
                    recon_pred = self.recon_head(features)
                    recon_val_loss = F.mse_loss(recon_pred, feats)

        if self.global_rank == 0:
            logger = getattr(self, "logger", None)
            if logger is not None and hasattr(logger, "experiment"):
                logger.experiment.add_scalar("loss/genomic_val", genomic_val_loss.item(), self.num_samples)
                if recon_val_loss is not None:
                    logger.experiment.add_scalar("loss/genomic_recon_val", recon_val_loss.item(), self.num_samples)

        self.log(
            "loss/genomic_val",
            genomic_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        if recon_val_loss is not None:
            self.log(
                "loss/genomic_recon_val",
                recon_val_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
        return loss_cond

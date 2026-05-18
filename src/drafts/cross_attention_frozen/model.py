"""
FrozenBackboneCaLitModel — frozen backbone + fresh cross-attention (v12).

Motivation
----------
In every previous run the backbone's reconstruction gradient (≈50× stronger
than any conditioning gradient) trained the backbone to suppress CA output.
guidance_delta peaked at 5.7e-4 then fell 98.6 % to 8.1e-6 over 2.5M samples.

Root cause: the backbone CAN suppress CA because its weights are updated by
reconstruction on the same step that CA output is applied.  The fix: freeze
the backbone entirely so it cannot compensate.

Setup
-----
1. Load the v11 final checkpoint (well-trained unconditional denoiser).
2. Freeze ALL backbone parameters (backbone.requires_grad_(False)).
3. Attach a FRESH CA block (re-initialised, not biased by 2.5M steps of
   suppression training).  Only CA parameters train.
4. CFG dropout = 30 % (vs 15 % in v11) — stronger incentive for CA on the
   70 % real-feats batches.

Why the frozen backbone approach can work
-----------------------------------------
With the backbone frozen, the reconstruction gradient of the 70 % conditional
batches has one path: through CA parameters only.  If CA produces a subtype-
specific h_mid modification that helps the frozen backbone reconstruct the
tile (possible because Basal/LumA have distinct morphology), the gradient
maintains that output.  If CA adds pure noise, gradient pushes it toward zero.
The backbone cannot "route around" CA because its weights don't change.

Inference is identical to v11 CFG:
    eps_guided = eps_null + s × (eps_cond − eps_null)

Diagnostic: guidance_delta should GROW (not decline) once CA learns a
subtype-specific direction the frozen backbone can use.
"""
from __future__ import annotations

import copy
import logging

import torch

from src.cross_attention.model import GenomicCaLitModel
from src.cross_attention.genomic_cross_attn import GenomicCrossAttentionBlock
from .genomic_config import FrozenBackboneCaConfig

log = logging.getLogger(__name__)


class FrozenBackboneCaLitModel(GenomicCaLitModel):
    """MoPaDi with frozen pretrained backbone; only the CA block is trained."""

    def __init__(self, conf: FrozenBackboneCaConfig):
        super().__init__(conf)
        self.conf: FrozenBackboneCaConfig = conf

        if conf.frozen_backbone and conf.pretrained_backbone_ckpt:
            self._load_and_freeze_backbone(conf.pretrained_backbone_ckpt)

    def _load_and_freeze_backbone(self, ckpt_path: str) -> None:
        """Load backbone from a finished checkpoint, freeze it, install fresh CA."""
        log.info("Loading pretrained backbone from: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]

        # Load backbone weights (exclude CA block which we'll re-init below).
        model_state = {
            k[len("model."):]: v
            for k, v in state.items()
            if k.startswith("model.") and "genomic_cross_attn" not in k
        }
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        log.info(
            "Backbone loaded into model: %d missing keys, %d unexpected",
            len(missing), len(unexpected),
        )

        ema_state = {
            k[len("ema_model."):]: v
            for k, v in state.items()
            if k.startswith("ema_model.") and "genomic_cross_attn" not in k
        }
        self.ema_model.load_state_dict(ema_state, strict=False)
        log.info("Backbone loaded into ema_model.")

        # Freeze entire backbone — no reconstruction gradient reaches backbone.
        self.model.requires_grad_(False)
        self.ema_model.requires_grad_(False)

        # Fresh CA: re-initialise with Xavier (override near-zero weights from v11).
        bottleneck_ch = self.conf.net_ch * max(self.conf.net_ch_mult)
        fresh_ca = GenomicCrossAttentionBlock(
            spatial_channels=bottleneck_ch,
            gene_dim=self.conf.feat_dim,
            n_heads=self.conf.genomic_ca_heads,
            n_gene_tokens=self.conf.genomic_ca_n_tokens,
        )
        self.model.genomic_cross_attn = fresh_ca
        self.model.genomic_cross_attn.requires_grad_(True)

        self.ema_model.genomic_cross_attn = copy.deepcopy(fresh_ca)
        # EMA CA is updated by the EMA mechanism (not trained directly).

        n_ca = sum(p.numel() for p in fresh_ca.parameters())
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "Frozen backbone setup complete. "
            "Training only CA: %d / %d parameters (%.2f %%)",
            n_ca, n_total, 100.0 * n_ca / n_total,
        )

    def configure_optimizers(self):
        # Only include CA parameters — frozen backbone has requires_grad=False.
        ca_params = [p for p in self.model.parameters() if p.requires_grad]
        if not ca_params:
            raise RuntimeError(
                "No trainable parameters found. "
                "Check that genomic_cross_attn.requires_grad_(True) was called."
            )
        log.info("Optimizer will update %d parameter tensors.", len(ca_params))
        optim = torch.optim.Adam(
            ca_params,
            lr=self.conf.lr,
            weight_decay=self.conf.weight_decay,
        )
        return {"optimizer": optim}

"""
Joint Genomic Encoder + Diffusion Model.

Subclasses mopadi's LitModel, adding only:
  - A genomic encoder to encode bulk-RNA data
  - A projection head mapping encoder latent → UNet conditioning
  - Modified training_step with pure diffusion loss (no VAE loss)

Everything else (EMA, optimizer, scheduler, gradient clipping, checkpointing,
sample visualization, DDP) is inherited from mopadi's LitModel.

Attribution:
    - mopadi repository: https://github.com/KatherLab/mopadi
    - We reuse mopadi modules via imports (no vendored mopadi source in this file).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add sibling mopadi/src to path if present (for non-installed mopadi copy).
project_root = Path(__file__).resolve().parents[2]
mopadi_src = (project_root.parent / "mopadi" / "src").resolve()
if mopadi_src.exists() and str(mopadi_src) not in sys.path:
    sys.path.insert(0, str(mopadi_src))

import numpy as np
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader

from mopadi.train_diff_autoenc import LitModel, ema
from mopadi.configs.config import TrainConfig
from mopadi.configs.choices import OptimizerType
from mopadi.configs.templates import autoenc_base
from mopadi.utils.dist_utils import get_world_size

try:
    from encoding.architecture import ProbabilisticEncoder
except ImportError:
    from src.encoding.architecture import ProbabilisticEncoder  # type: ignore[import-not-found]


# ──────────────────────────────────────────────────────────────────────
#  Projection Head: VAE latent → UNet conditioning space
# ──────────────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """MLP mapping VAE latent dim → UNet conditioning dim."""

    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * max(num_layers - 1, 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────
#  Build mopadi TrainConfig from our YAML config
# ──────────────────────────────────────────────────────────────────────

def build_conf(joint_cfg: dict) -> TrainConfig:
    """Build a mopadi TrainConfig from the joint_training YAML section."""
    conf = autoenc_base()

    # Conditioning dimension
    cond_dim = joint_cfg.get("cond_dim", 512)
    conf.feat_dim = cond_dim
    conf.style_ch = cond_dim
    conf.net_beatgans_embed_channels = cond_dim

    # UNet architecture
    conf.net_ch = int(joint_cfg.get("net_ch", 128))
    net_ch_mult_list = joint_cfg.get("net_ch_mult", (1, 1, 2, 2, 4, 4))
    conf.net_ch_mult = tuple(int(x) for x in net_ch_mult_list)  # type: ignore[assignment]
    conf.net_num_res_blocks = int(joint_cfg.get("net_num_res_blocks", 2))
    conf.img_size = int(joint_cfg.get("img_size", 512))
    conf.sample_size = int(joint_cfg.get("sample_size", 8))

    # Diffusion
    conf.T = int(joint_cfg.get("T", 1000))
    conf.T_eval = int(joint_cfg.get("T_eval", 20))
    conf.fp16 = bool(joint_cfg.get("fp16", False))
    conf.dropout = float(joint_cfg.get("dropout", 0.1))

    # Training
    conf.batch_size = int(joint_cfg.get("batch_size", 4))
    conf.lr = float(joint_cfg.get("lr", 1e-4))
    conf.weight_decay = float(joint_cfg.get("weight_decay", 0.0))
    conf.grad_clip = float(joint_cfg.get("grad_clip", 1.0))
    conf.ema_decay = float(joint_cfg.get("ema_decay", 0.9999))
    conf.warmup = int(joint_cfg.get("warmup_steps", 0))
    conf.accum_batches = int(joint_cfg.get("accumulate_grad_batches", 1))

    opt = joint_cfg.get("optimizer", "adam")
    conf.optimizer = {"adam": OptimizerType.adam, "adamw": OptimizerType.adamw}.get(
        opt, OptimizerType.adam
    )

    # Scheduling (mopadi-style: sample-count based)
    conf.total_samples = int(joint_cfg.get("total_samples", 200_000_000))
    conf.steps_per_epoch = int(joint_cfg.get("steps_per_epoch", 5_000))
    conf.save_every_samples = int(joint_cfg.get("save_every_samples", 200_000))
    conf.reconstruct_every_samples = int(joint_cfg.get("reconstruct_every_samples", 50_000))

    # Disable mopadi's built-in FID eval (our dataset format differs)
    conf.eval_every_samples = 0
    conf.eval_ema_every_samples = 0

    # Output
    conf.base_dir = joint_cfg.get("out_dir", "experiments/joint_training")
    conf.name = "joint"
    conf.load_pretrained_autoenc = False

    conf.make_model_conf()
    return conf


# ──────────────────────────────────────────────────────────────────────
#  Joint Lightning Module (subclasses mopadi's LitModel)
# ──────────────────────────────────────────────────────────────────────

class JointLitModel(LitModel):
    """
    mopadi LitModel + genomic encoder + projection head.

    Uses a genomic encoder to encode bulk-RNA data into a latent space,
    then projects to the UNet conditioning space. The diffusion model is
    trained with only diffusion loss (no VAE loss).
    Inherits from mopadi: EMA, gradient clipping, sample visualization,
    checkpointing, DDP support.
    """

    def __init__(self, conf: TrainConfig, joint_cfg: dict, n_genes: int):
        # Save full hyperparameters (including conf) so checkpoints carry all required metadata
        super().__init__(conf)
        self.save_hyperparameters({
            "conf": conf,
            "joint_cfg": joint_cfg,
            "n_genes": n_genes,
            "joint_variant": "joint_training",
        })
        self.joint_cfg = joint_cfg

        # ── Genomic Encoder (VAE encoder only, no decoder) ──────────────────────────────────
        latent_dim = int(joint_cfg.get("latent_dim", 512))
        hidden_dims = [int(x) for x in joint_cfg.get("vae_hidden_dims", [2048, 1024])]
        vae_dropout = float(joint_cfg.get("vae_dropout", 0.2))

        self.encoder = ProbabilisticEncoder(
            input_dim=n_genes, hidden_dim=hidden_dims,
            latent_dim=latent_dim, dropout=vae_dropout,
        )

        # ── Projection head ───────────────────────────────────────────
        cond_dim = int(joint_cfg.get("cond_dim", conf.feat_dim))
        self.projection = ProjectionHead(
            in_dim=latent_dim, out_dim=cond_dim,
            hidden_dim=int(joint_cfg.get("proj_hidden_dim", 512)),
            num_layers=int(joint_cfg.get("proj_layers", 2)),
            dropout=float(joint_cfg.get("proj_dropout", 0.1)),
        )

        # ── Load pre-trained weights ──────────────────────────────────
        diffusion_ckpt = joint_cfg.get("diffusion_ckpt")
        if diffusion_ckpt and os.path.exists(diffusion_ckpt):
            print(f"[Joint] Loading diffusion checkpoint: {diffusion_ckpt}")
            state = torch.load(diffusion_ckpt, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            # Filter out keys whose shape doesn't match the current model
            # (e.g. x_T buffer varies with sample_size between checkpoints).
            # strict=False skips missing/unexpected keys but still errors on
            # shape mismatches, so we must drop them explicitly.
            current_sd = self.state_dict()
            mismatched = [k for k, v in sd.items()
                          if k in current_sd and current_sd[k].shape != v.shape]
            if mismatched:
                print(f"[Joint] Skipping {len(mismatched)} shape-mismatched key(s) "
                      f"from checkpoint: {mismatched}")
                sd = {k: v for k, v in sd.items() if k not in mismatched}
            self.load_state_dict(sd, strict=False)

        encoder_ckpt = joint_cfg.get("encoder_ckpt")
        if encoder_ckpt and os.path.exists(encoder_ckpt):
            print(f"[Joint] Loading encoder checkpoint: {encoder_ckpt}")
            self.encoder.load_state_dict(
                torch.load(encoder_ckpt, map_location="cpu", weights_only=False),
                strict=False,
            )

        n_encoder = sum(p.numel() for p in self.encoder.parameters())
        n_proj = sum(p.numel() for p in self.projection.parameters())
        n_unet = sum(p.numel() for p in self.model.parameters())
        print(f"[Joint] Encoder: {n_encoder:,}  Proj: {n_proj:,}  UNet: {n_unet:,}  "
              f"Total: {n_encoder + n_proj + n_unet:,}")

    # ──────────────────────────────────────────────────────────────────
    #  Dataset (overrides mopadi's WebDataset setup)
    # ──────────────────────────────────────────────────────────────────

    def setup(self, stage=None):
        """Create genomic datasets with patient-level train/val/test split."""
        if self.conf.seed is not None:
            seed = self.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        try:
            from joint_training.dataset import (
                GenomicTileDataset, patient_split, save_split, load_split, canonical_patient_id,
            )
        except ImportError:
            from src.joint_training.dataset import (  # type: ignore[import-not-found]
                GenomicTileDataset, patient_split, save_split, load_split, canonical_patient_id,
            )

        cfg = self.joint_cfg
        gene_list = None
        glp = cfg.get("gene_list_path")
        if glp and os.path.exists(glp):
            with open(glp) as f:
                gene_list = [line.strip() for line in f if line.strip()]

        # Common dataset kwargs
        ds_kwargs = dict(
            csv_path=cfg["csv_path"],
            tiles_zip_dir=cfg["tiles_zip_dir"],
            img_size=self.conf.img_size,
            patient_col=cfg.get("patient_col", "Patient_ID"),
            label_col=cfg.get("label_col"),
            gene_list=gene_list,
            max_tiles_per_patient=cfg.get("max_tiles_per_patient"),
        )

        # -- Patient-level split (deterministic, logged) ---------------
        split_path = os.path.join(self.conf.base_dir, "patient_splits.json")
        if os.path.exists(split_path):
            # Reuse existing split (e.g. after resume)
            splits = load_split(split_path)
            if self.global_rank == 0:
                print(f"[Joint] Loaded existing patient split from {split_path}")
        else:
            # Build a temporary full dataset just to discover matched patients
            discovery_ds = GenomicTileDataset(**ds_kwargs)  # type: ignore[arg-type]
            all_patients = discovery_ds.patient_ids
            del discovery_ds  # free memory

            splits = patient_split(
                all_patients,
                val_fraction=cfg.get("val_fraction", 0.1),
                test_fraction=cfg.get("test_fraction", 0.1),
                seed=cfg.get("seed", 42),
            )
            if self.global_rank == 0:
                saved = save_split(splits, self.conf.base_dir)
                print(f"[Joint] Patient split saved to {saved}")

        # Guardrail: enforce disjoint patient-level splits after canonicalization.
        split_sets = {
            name: {canonical_patient_id(pid) for pid in pts}
            for name, pts in splits.items()
        }
        overlap_train_val = split_sets.get("train", set()) & split_sets.get("val", set())
        overlap_train_test = split_sets.get("train", set()) & split_sets.get("test", set())
        overlap_val_test = split_sets.get("val", set()) & split_sets.get("test", set())
        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                "Patient split integrity violation: at least one canonical patient ID appears in multiple splits"
            )

        if self.global_rank == 0:
            print(f"[Joint] Patients — train: {len(splits['train'])}, "
                  f"val: {len(splits['val'])}, test: {len(splits['test'])}")

        # Create separate datasets for train and val (test is held out)
        self.train_data = GenomicTileDataset(**ds_kwargs, patient_ids=splits["train"])  # type: ignore[arg-type]
        norm_means, norm_stds, apply_log1p = self.train_data.get_normalization_state()

        # Persist normalization stats so reconstruction can use training-fitted
        # statistics rather than recomputing them from the (possibly smaller)
        # inference set, which would introduce a distribution shift.
        if self.global_rank == 0:
            import json
            norm_stats_path = os.path.join(self.conf.base_dir, "normalization_stats.json")
            if not os.path.exists(norm_stats_path):
                with open(norm_stats_path, "w") as _nf:
                    json.dump({
                        "means": norm_means.tolist(),
                        "stds": norm_stds.tolist(),
                        "apply_log1p": bool(apply_log1p),
                    }, _nf)
                print(f"[Joint] Normalization stats saved to {norm_stats_path}")

        self.val_data = GenomicTileDataset(  # type: ignore[arg-type]
            **ds_kwargs,
            patient_ids=splits["val"],
            norm_means=norm_means,
            norm_stds=norm_stds,
            apply_log1p=apply_log1p,
        )

        if self.global_rank == 0:
            print(f"[Joint] Train tiles: {len(self.train_data)}, "
                  f"Val tiles: {len(self.val_data)}")

    def train_dataloader(self):
        nw = self.joint_cfg.get("num_workers", 4)
        prefetch_factor = self.joint_cfg.get("prefetch_factor", 2)

        loader_kwargs = {
            "dataset": self.train_data,
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": nw,
            "pin_memory": True,
            "drop_last": True,
            "persistent_workers": nw > 0,
        }
        if nw > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**loader_kwargs)

    def val_dataloader(self):
        nw = self.joint_cfg.get("val_num_workers", self.joint_cfg.get("num_workers", 4))
        val_batch_size = self.joint_cfg.get("val_batch_size", self.batch_size)
        prefetch_factor = self.joint_cfg.get("val_prefetch_factor", self.joint_cfg.get("prefetch_factor", 2))

        loader_kwargs = {
            "dataset": self.val_data,
            "batch_size": val_batch_size,
            "shuffle": False,
            "num_workers": nw,
            "pin_memory": True,
            "drop_last": False,
            "persistent_workers": nw > 0,
        }
        if nw > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        return DataLoader(**loader_kwargs)

    def on_fit_start(self):
        """Move encoder and projection to correct device before training starts."""
        self.encoder = self.encoder.to(self.device)
        self.projection = self.projection.to(self.device)
        
        if self.global_rank == 0:
            print(f"[Joint] Moved encoder and projection to device: {self.device}")

    # ──────────────────────────────────────────────────────────────────
    #  Training (overrides mopadi's training_step)
    # ──────────────────────────────────────────────────────────────────

    def encode_genomic(self, genomic, deterministic: bool = False):
        """RNA-seq → encoder → projection → conditioning vector.
        
        Note: encoder and projection are already on device (moved in on_fit_start).
        No device checks here to avoid overhead in hot path.
        """
        mean, log_var = self.encoder(genomic)
        if deterministic:
            z = mean
        else:
            # Use reparameterization trick (stochastic latent)
            z = mean + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        cond = self.projection(z)
        return cond

    def training_step(self, batch, batch_idx):
        """
        Diffusion-only training step (no VAE loss).
        Follows mopadi's pattern: encoder → projection → diffusion loss.
        """
        with autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            genomic = batch['genomic'].to(self.device, dtype=torch.float32)

            # Encoder → conditioning vector (stochastic during training)
            cond = self.encode_genomic(genomic)

            # Diffusion loss (same as mopadi's sampler)
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            losses = self.sampler.training_losses(
                model=self.model, x_start=imgs, cond=cond,
                t=t, model_kwargs={'cond': cond},
            )
            loss = losses['loss'].mean()

        # Standard PyTorch Lightning logging
        self.log('loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log('loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))

        if self.global_rank == 0 and hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_scalar('loss', loss.item(), self.num_samples)  # type: ignore[union-attr]

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Diffusion-only validation step (no VAE loss).
        """
        with autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            genomic = batch['genomic'].to(self.device, dtype=torch.float32)

            # Deterministic encoding: val_loss is used by ModelCheckpoint for
            # best-checkpoint selection, so removing the reparameterisation
            # noise is important for a stable and reproducible metric.
            cond = self.encode_genomic(genomic, deterministic=True)

            # Diffusion loss
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            losses = self.sampler.training_losses(
                model=self.model, x_start=imgs, cond=cond,
                t=t, model_kwargs={'cond': cond},
            )
            loss = losses['loss'].mean()

        # Log validation metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        
        return {'val_loss': loss}

    def on_train_batch_start(self, batch, batch_idx):
        """Called at the start of each training batch.
        
        Note: Device assignment is handled once in on_fit_start().
        Removed redundant device checks (were a performance bottleneck).
        """
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update + sample visualization (inherited patterns from mopadi)."""
        if self.is_last_accum(batch_idx):
            ema(self.model, self.ema_model, self.conf.ema_decay)

            # Compute cond from genomic for mopadi's log_sample
            with torch.no_grad():
                genomic = batch['genomic'].to(self.device, dtype=torch.float32)
                cond = self.encode_genomic(genomic)

            self.log_sample(x_start=batch['img'], cond=cond)
            self.evaluate_scores()

    # ──────────────────────────────────────────────────────────────────
    #  Optimizer (extends mopadi's to include VAE + projection)
    # ──────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        conf = self.conf
        jcfg = self.joint_cfg
        lr = float(conf.lr)

        param_groups = [
            {"params": self.model.parameters(), "lr": float(jcfg.get("unet_lr", lr))},
            {"params": self.encoder.parameters(), "lr": float(jcfg.get("encoder_lr", lr))},
            {"params": self.projection.parameters(), "lr": float(jcfg.get("proj_lr", lr))},
        ]

        if conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                param_groups, betas=(0.9, 0.99), eps=1e-6,
                weight_decay=conf.weight_decay,
            )
        else:
            optim = torch.optim.Adam(param_groups, weight_decay=conf.weight_decay)

        # Scheduler based on actual training steps (not total_samples)
        # Calculate from: epochs * steps_per_epoch = actual optimizer steps
        epochs = int(jcfg.get("epochs", conf.max_epochs if hasattr(conf, 'max_epochs') else 100))
        steps_per_epoch = int(conf.steps_per_epoch)
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(conf.warmup)
        
        if warmup_steps > 0:
            warmup = LambdaLR(
                optim, lr_lambda=lambda s: min(s + 1, warmup_steps) / warmup_steps
            )
            cosine = CosineAnnealingLR(
                optim, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6
            )
            sched = SequentialLR(
                optim, schedulers=[warmup, cosine], milestones=[warmup_steps]
            )
        else:
            sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=1e-6)

        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # ──────────────────────────────────────────────────────────────────
    #  Inference helpers
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, genomic: torch.Tensor) -> torch.Tensor:
        """Encode gene expression → diffusion conditioning (deterministic)."""
        self.eval()
        mean, _ = self.encoder(genomic)
        return self.projection(mean)

    @torch.no_grad()
    def generate(self, genomic: torch.Tensor) -> torch.Tensor:
        """Generate tiles from genomic vectors using the EMA model."""
        cond = self.encode(genomic)
        noise = torch.randn(
            cond.size(0), 3, self.conf.img_size, self.conf.img_size,
            device=self.device,
        )
        return self.render(noise, cond)

    @torch.no_grad()
    def save_latent_features(self, out_dir: str, split: str = "all") -> str:
        """Extract encoder latent features and save one h5 file per patient.

        Each ``<patient_id>.h5`` contains:
            - ``feats`` : (latent_dim,) latent vector from encoder (deterministic mean)
            - Patient ID and split as HDF5 attributes.

        Parameters
        ----------
        out_dir : str
            Directory where h5 files will be written.
        split : str
            Which split to extract ("train", "val", "test", or "all").
            "all" iterates over train + val datasets.

        Returns
        -------
        str
            Path to the output directory.
        """
        import h5py

        os.makedirs(out_dir, exist_ok=True)
        self.eval()

        # Collect the datasets to iterate
        datasets = []
        if split in ("train", "all") and hasattr(self, "train_data"):
            datasets.append(("train", self.train_data))
        if split in ("val", "all") and hasattr(self, "val_data"):
            datasets.append(("val", self.val_data))
        if split == "test":
            # Test data isn't loaded by default — build it on the fly
            try:
                from joint_training.dataset import GenomicTileDataset, load_split
            except ImportError:
                from src.joint_training.dataset import GenomicTileDataset, load_split  # type: ignore[import-not-found]
            split_path = os.path.join(self.conf.base_dir, "patient_splits.json")
            if os.path.exists(split_path):
                splits = load_split(split_path)
                cfg = self.joint_cfg
                _nm, _ns, _ali = (
                    self.train_data.get_normalization_state()
                    if hasattr(self, "train_data") else (None, None, None)
                )
                test_ds = GenomicTileDataset(
                    csv_path=cfg["csv_path"],
                    tiles_zip_dir=cfg["tiles_zip_dir"],
                    img_size=self.conf.img_size,
                    patient_col=cfg.get("patient_col", "Patient_ID"),
                    label_col=cfg.get("label_col"),
                    patient_ids=splits["test"],
                    norm_means=_nm,
                    norm_stds=_ns,
                    apply_log1p=_ali,
                )
                datasets.append(("test", test_ds))

        # We only need one genomic vector per patient (not per tile)
        seen_patients: set[str] = set()
        saved = 0

        for split_name, ds in datasets:
            # Get patient → genomic mapping from dataset internals
            raw_ds = ds.dataset if hasattr(ds, "dataset") else ds
            for pid, genomic_vec in raw_ds._genomic.items():
                if pid in seen_patients:
                    continue
                # Check this patient is actually in the dataset's sample list
                if hasattr(raw_ds, "patient_ids") and pid not in raw_ds.patient_ids:
                    continue
                seen_patients.add(pid)

                g = genomic_vec.unsqueeze(0).to(self.device)
                mu, log_var = self.encoder(g)
                z = mu  # deterministic: use the mean
                cond = self.projection(z)

                h5_path = os.path.join(out_dir, f"{pid}.h5")
                with h5py.File(h5_path, "w") as f:
                    # Save 'feats' with shape (1, latent_dim) as expected by downstream tools
                    f.create_dataset("feats", data=z.cpu().numpy().astype(np.float32))
                    
                    f.attrs["patient_id"] = pid
                    f.attrs["split"] = split_name
                saved += 1

        print(f"[Joint] Saved latent features for {saved} patients to {out_dir}")
        return out_dir

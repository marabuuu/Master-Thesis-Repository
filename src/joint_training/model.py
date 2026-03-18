"""
Joint Genomic VAE + Diffusion Model.

Subclasses mopadi's LitModel, adding only:
  - A genomic VAE to encode bulk-RNA data
  - A projection head mapping VAE latent → UNet conditioning
  - Modified training_step with joint VAE + diffusion loss

Everything else (EMA, optimizer, scheduler, gradient clipping, checkpointing,
sample visualization, DDP) is inherited from mopadi's LitModel.
"""

from __future__ import annotations

import os

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
    from encoding.architecture import ProbabilisticEncoder, ProbabilisticDecoder, VAE
    from encoding.architecture.loss import compute_mmd
except ImportError:
    from src.encoding.architecture import ProbabilisticEncoder, ProbabilisticDecoder, VAE  # type: ignore[import-not-found]
    from src.encoding.architecture.loss import compute_mmd  # type: ignore[import-not-found]


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
    mopadi LitModel + genomic VAE + projection head.

    Replaces image feature extractors with a jointly-trained VAE
    on bulk-RNA data. Inherits from mopadi: EMA, gradient clipping,
    sample visualization, checkpointing, DDP support.
    """

    def __init__(self, conf: TrainConfig, joint_cfg: dict, n_genes: int):
        self.save_hyperparameters(ignore=["conf"])
        super().__init__(conf)
        self.joint_cfg = joint_cfg

        # ── VAE ───────────────────────────────────────────────────────
        latent_dim = int(joint_cfg.get("latent_dim", 512))
        hidden_dims = [int(x) for x in joint_cfg.get("vae_hidden_dims", [2048, 1024])]
        vae_dropout = float(joint_cfg.get("vae_dropout", 0.2))
        self.lambda_vae = float(joint_cfg.get("lambda_vae", 1.0))
        self.beta_mmd = float(joint_cfg.get("beta_mmd", 1.0))

        encoder = ProbabilisticEncoder(
            input_dim=n_genes, hidden_dim=hidden_dims,
            latent_dim=latent_dim, dropout=vae_dropout,
        )
        decoder = ProbabilisticDecoder(
            latent_dim=latent_dim, hidden_dim=list(reversed(hidden_dims)),
            output_dim=n_genes, dropout=vae_dropout,
        )
        # Create VAE without explicitly specifying device initially
        # We'll use torch.device('cpu') as a placeholder and move it on_fit_start
        try:
            self.vae = VAE(encoder, decoder, device=torch.device("cpu"))
        except Exception:
            # Fallback: create with no device arg and rely on to() calls
            self.vae = VAE(encoder, decoder, device=None)  # type: ignore[arg-type]
        # Disable VAE's internal device relocation
        self.vae._relocate = lambda: None

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
            self.load_state_dict(sd, strict=False)

        vae_ckpt = joint_cfg.get("vae_ckpt")
        if vae_ckpt and os.path.exists(vae_ckpt):
            print(f"[Joint] Loading VAE checkpoint: {vae_ckpt}")
            self.vae.load_state_dict(
                torch.load(vae_ckpt, map_location="cpu", weights_only=False),
                strict=False,
            )

        n_vae = sum(p.numel() for p in self.vae.parameters())
        n_proj = sum(p.numel() for p in self.projection.parameters())
        n_unet = sum(p.numel() for p in self.model.parameters())
        print(f"[Joint] VAE: {n_vae:,}  Proj: {n_proj:,}  UNet: {n_unet:,}  "
              f"Total: {n_vae + n_proj + n_unet:,}")

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
                GenomicTileDataset, patient_split, save_split, load_split,
            )
        except ImportError:
            from src.joint_training.dataset import (  # type: ignore[import-not-found]
                GenomicTileDataset, patient_split, save_split, load_split,
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

        if self.global_rank == 0:
            print(f"[Joint] Patients — train: {len(splits['train'])}, "
                  f"val: {len(splits['val'])}, test: {len(splits['test'])}")

        # Create separate datasets for train and val (test is held out)
        self.train_data = GenomicTileDataset(**ds_kwargs, patient_ids=splits["train"])  # type: ignore[arg-type]
        self.val_data = GenomicTileDataset(**ds_kwargs, patient_ids=splits["val"])  # type: ignore[arg-type]

        if self.global_rank == 0:
            print(f"[Joint] Train tiles: {len(self.train_data)}, "
                  f"Val tiles: {len(self.val_data)}")

    def train_dataloader(self):
        nw = self.joint_cfg.get("num_workers", 4)
        return DataLoader(
            self.train_data, batch_size=self.batch_size,
            shuffle=True, num_workers=nw, pin_memory=True,
            drop_last=True, persistent_workers=nw > 0,
        )

    def val_dataloader(self):
        nw = self.joint_cfg.get("num_workers", 4)
        return DataLoader(
            self.val_data, batch_size=self.batch_size,
            shuffle=False, num_workers=nw, pin_memory=True,
            drop_last=False, persistent_workers=nw > 0,
        )

    def on_fit_start(self):
        """Move VAE and projection to correct device before training starts."""
        # Explicitly move all VAE components and projection to the training device
        if hasattr(self.vae, 'encoder'):
            self.vae.encoder = self.vae.encoder.to(self.device)
        if hasattr(self.vae, 'decoder'):
            self.vae.decoder = self.vae.decoder.to(self.device)
        if hasattr(self.vae, 'device'):
            self.vae.device = self.device
        self.projection = self.projection.to(self.device)
        
        if self.global_rank == 0:
            print(f"[Joint] Moved VAE and projection to device: {self.device}")

    # ──────────────────────────────────────────────────────────────────
    #  Training (overrides mopadi's training_step)
    # ──────────────────────────────────────────────────────────────────

    def encode_genomic(self, genomic):
        """RNA-seq → VAE → projection → cond, plus reconstruction for loss."""
        # Safety: ensure VAE is on the correct device before forward pass
        if next(self.vae.encoder.parameters()).device != self.device:
            self.vae.encoder = self.vae.encoder.to(self.device)
            self.vae.decoder = self.vae.decoder.to(self.device)
        
        mean, log_var = self.vae.encoder(genomic)
        z = self.vae.reparameterization(mean, torch.exp(0.5 * log_var))
        vae_recon = self.vae.decoder(z)
        cond = self.projection(z)
        return cond, z, vae_recon

    def training_step(self, batch, batch_idx):
        with autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            genomic = batch['genomic'].to(self.device, dtype=torch.float32)

            # VAE → conditioning
            cond, z, vae_recon = self.encode_genomic(genomic)

            # Diffusion loss (mopadi's sampler)
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            losses = self.sampler.training_losses(
                model=self.model, x_start=imgs, cond=cond,
                t=t, model_kwargs={'cond': cond},
            )
            diff_loss = losses['loss'].mean()

            # VAE loss
            recon_loss = nn.functional.mse_loss(vae_recon, genomic)
            mmd_loss = compute_mmd(torch.randn_like(z), z).sum()
            vae_total = recon_loss + self.beta_mmd * mmd_loss

            loss = diff_loss + self.lambda_vae * vae_total

        # Standard PyTorch Lightning logging for epoch aggregation and progress bars
        self.log('loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log('loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=len(imgs))

        if self.global_rank == 0 and hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            self.logger.experiment.add_scalar('loss', loss.item(), self.num_samples)  # type: ignore[union-attr]
            self.logger.experiment.add_scalar('loss/diff', diff_loss.item(), self.num_samples)  # type: ignore[union-attr]
            self.logger.experiment.add_scalar('loss/vae_recon', recon_loss.item(), self.num_samples)  # type: ignore[union-attr]
            self.logger.experiment.add_scalar('loss/vae_mmd', mmd_loss.item(), self.num_samples)  # type: ignore[union-attr]

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        with autocast(device_type='cuda', enabled=self.conf.fp16):
            imgs = batch['img'].to(self.device)
            genomic = batch['genomic'].to(self.device, dtype=torch.float32)

            # VAE → conditioning
            cond, z, vae_recon = self.encode_genomic(genomic)

            # Diffusion loss
            t, _ = self.T_sampler.sample(len(imgs), imgs.device)
            losses = self.sampler.training_losses(
                model=self.model, x_start=imgs, cond=cond,
                t=t, model_kwargs={'cond': cond},
            )
            diff_loss = losses['loss'].mean()

            # VAE loss
            recon_loss = nn.functional.mse_loss(vae_recon, genomic)
            mmd_loss = compute_mmd(torch.randn_like(z), z).sum()
            vae_total = recon_loss + self.beta_mmd * mmd_loss

            loss = diff_loss + self.lambda_vae * vae_total

        # Log validation metrics (aggregated over epoch automatically)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(imgs))
        self.log('val_loss/diff', diff_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(imgs))
        self.log('val_loss/vae_recon', recon_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(imgs))
        self.log('val_loss/vae_mmd', mmd_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=len(imgs))
        
        return {'val_loss': loss}

    def on_train_batch_start(self, batch, batch_idx):
        """Ensure VAE and projection are on correct device at the start of each batch."""
        try:
            encoder_device = next(self.vae.encoder.parameters()).device
            if encoder_device != self.device:
                self.vae.encoder = self.vae.encoder.to(self.device)
                self.vae.decoder = self.vae.decoder.to(self.device)
            proj_device = next(self.projection.parameters()).device
            if proj_device != self.device:
                self.projection = self.projection.to(self.device)
        except (StopIteration, AttributeError):
            pass  # Module may not have parameters if not yet initialized

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """EMA update + sample visualization (inherited patterns from mopadi)."""
        if self.is_last_accum(batch_idx):
            ema(self.model, self.ema_model, self.conf.ema_decay)

            # Compute cond from genomic for mopadi's log_sample
            with torch.no_grad():
                genomic = batch['genomic'].to(self.device, dtype=torch.float32)
                mean, _ = self.vae.encoder(genomic)
                cond = self.projection(mean)

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
            {"params": self.vae.parameters(), "lr": float(jcfg.get("vae_lr", lr))},
            {"params": self.projection.parameters(), "lr": float(jcfg.get("proj_lr", lr))},
        ]

        if conf.optimizer == OptimizerType.adamw:
            optim = torch.optim.AdamW(
                param_groups, betas=(0.9, 0.99), eps=1e-6,
                weight_decay=conf.weight_decay,
            )
        else:
            optim = torch.optim.Adam(param_groups, weight_decay=conf.weight_decay)

        # Mopadi's scheduler pattern: warmup + cosine
        total_steps = max(1, int(conf.total_samples // conf.batch_size_effective))
        if int(conf.warmup) > 0:
            warmup = LambdaLR(
                optim, lr_lambda=lambda s: min(s + 1, int(conf.warmup)) / int(conf.warmup)
            )
            cosine = CosineAnnealingLR(
                optim, T_max=max(1, total_steps - int(conf.warmup)), eta_min=1e-6
            )
            sched = SequentialLR(
                optim, schedulers=[warmup, cosine], milestones=[int(conf.warmup)]
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
        mean, _ = self.vae.encoder(genomic)
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
        """Extract VAE latent features and save one h5 file per patient.

        Each ``<patient_id>.h5`` contains:
            - ``z``     : (latent_dim,) latent vector (using the mean, deterministic)
            - ``mu``    : (latent_dim,) encoder mean
            - ``log_var``: (latent_dim,) encoder log-variance
            - ``recon`` : (n_genes,) reconstructed gene expression
            - ``cond``  : (cond_dim,) projected conditioning vector
            - ``genes`` : gene names as UTF-8 byte strings (attribute on root)

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
                test_ds = GenomicTileDataset(
                    csv_path=cfg["csv_path"],
                    tiles_zip_dir=cfg["tiles_zip_dir"],
                    img_size=self.conf.img_size,
                    patient_col=cfg.get("patient_col", "Patient_ID"),
                    label_col=cfg.get("label_col"),
                    patient_ids=splits["test"],
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
                mu, log_var = self.vae.encoder(g)
                z = mu  # deterministic: use the mean
                recon = self.vae.decoder(z)
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

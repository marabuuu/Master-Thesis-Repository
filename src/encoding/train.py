#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train VAE for Genomic Feature Encoding

This script trains a Variational Autoencoder to encode high-dimensional genomic
data (e.g., gene expression from CSV) into compact latent feature vectors.

The output feature vectors can be used for downstream tasks like:
- Conditional image generation with diffusion models
- Patient stratification and clustering
- Genomic subtype classification

Usage:
    # Basic training
    python -m src.encoding.train \\
        --csv /path/to/gene_expression.csv \\
        --out-dir ./output \\
        --latent-dim 512 \\
        --epochs 100

    # With custom architecture
    python -m src.encoding.train \\
        --csv /path/to/gene_expression.csv \\
        --out-dir ./output \\
        --hidden-dim 2048,1024 \\
        --latent-dim 512 \\
        --epochs 100 \\
        --batch-size 64

    # With external checkpoint directory (recommended)
    python -m src.encoding.train \\
        --csv /path/to/gene_expression.csv \\
        --out-dir ./output \\
        --checkpoint-dir /external/checkpoints \\
        --epochs 100
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for imports when running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.encoding.architecture import ProbabilisticEncoder, ProbabilisticDecoder, VAE
from src.preprocessing.utils import preprocess_log1p_zscore, inspect_variance


def make_unique_ids(index_iter):
    """Handle duplicate Patient_IDs by appending -DXn suffix."""
    counter = defaultdict(int)
    new_ids = []
    orig_map = {}
    for pid in index_iter:
        pid = str(pid)
        counter[pid] += 1
        if counter[pid] == 1:
            new_id = pid
        else:
            new_id = f"{pid}-DX{counter[pid]}"
        new_ids.append(new_id)
        orig_map[new_id] = pid
    return new_ids, orig_map


def save_h5(pid: str, arr: np.ndarray, outdir: str) -> str:
    """Save feature array to H5 file."""
    path = os.path.join(outdir, f"{pid}.h5")
    with h5py.File(path, 'w') as f:
        data = np.asarray(arr, dtype='float32')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        f.create_dataset('feats', data=data)
    return path


def clear_h5_folder(folder: str):
    """Remove existing .h5 files from folder."""
    try:
        for fname in os.listdir(folder):
            if fname.endswith('.h5'):
                try:
                    os.remove(os.path.join(folder, fname))
                except Exception:
                    pass
    except Exception:
        pass


def train_vae(
    csv_path: str,
    out_dir: str,
    hidden_dim: str = "2048,1024",
    latent_dim: int = 512,
    lr: float = 1e-3,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    mopadi_out_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    label_col: str = 'Majority_Subtype_mRNA',
    test_size: float = 0.2,
    random_state: int = 42,
    checkpoint_dir: Optional[str] = None,
):
    """
    Train VAE encoder and save encoded features.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV with samples x genes
    out_dir : str
        Output directory for encoded features and logs
    hidden_dim : str
        Comma-separated hidden layer dimensions (e.g., "2048,1024")
    latent_dim : int
        Latent space dimension
    lr : float
        Learning rate
    epochs : int
        Number of training epochs
    batch_size : int, optional
        Batch size (None for full-batch training)
    mopadi_out_dir : str, optional
        Output directory for MoPaDi-format H5 files
    metadata_csv : str, optional
        Path to metadata CSV with labels
    label_col : str
        Column name for labels in metadata
    test_size : float
        Fraction of data for test split
    random_state : int
        Random seed for reproducibility
    checkpoint_dir : str, optional
        External directory for saving model checkpoints
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path, index_col=0)
    print(f'Loaded data: {df.shape}')

    # Handle Patient_ID column
    if 'Patient_ID' in df.columns:
        try:
            df = df.set_index('Patient_ID')
            print(f'Set Patient_ID as index; shape: {df.shape}')
        except Exception as e:
            print(f'Could not set Patient_ID as index: {e}')

    # Extract labels before numeric conversion
    csv_labels = None
    if label_col in df.columns:
        try:
            csv_labels = df[label_col].astype(str)
            df = df.drop(columns=[label_col])
        except Exception:
            csv_labels = None

    # Ensure numeric data
    def try_cast(df_in):
        try:
            _ = df_in.values.astype(float)
            return df_in
        except Exception:
            return None

    cast_df = try_cast(df)
    if cast_df is None:
        # Try transpose (genes as rows)
        df_t = df.T
        cast_df = try_cast(df_t)
        if cast_df is not None:
            df = df_t
            print(f'Transposed to samples x genes: {df.shape}')
        else:
            # Coerce non-numeric to NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            n_nan = df.isna().sum().sum()
            print(f'Coerced non-numeric to NaN: {n_nan} values')
            
            all_nan_cols = df.columns[df.isna().all()].tolist()
            if all_nan_cols:
                print(f'Dropping {len(all_nan_cols)} all-NaN columns')
                df = df.drop(columns=all_nan_cols)
            df = df.fillna(df.mean())

    print(f'Variance summary: {inspect_variance(df)["gene_var_summary"]}')

    # Preprocess: log1p + z-score
    df_proc = preprocess_log1p_zscore(df)

    # Setup device and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    x = torch.tensor(df_proc.values, dtype=torch.float32)

    # Optional minibatch training
    if batch_size is not None:
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = None

    x = x.to(device)

    # Build model
    input_dim = x.shape[1]
    if isinstance(hidden_dim, str):
        hidden = [int(h) for h in hidden_dim.split(',')]
    else:
        hidden = hidden_dim

    enc = ProbabilisticEncoder(input_dim, hidden, latent_dim).to(device)
    dec = ProbabilisticDecoder(latent_dim, hidden, input_dim).to(device)
    vae = VAE(enc, dec, device).to(device)

    print(f'Model: input_dim={input_dim}, hidden={hidden}, latent_dim={latent_dim}')
    print(f'Parameters: {sum(p.numel() for p in vae.parameters()):,}')

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # Training loop
    print(f'\nTraining for {epochs} epochs...')
    for epoch in range(epochs):
        vae.train()
        if dataloader is None:
            optimizer.zero_grad()
            total, recon, mmd = vae.loss_components(x, beta=1.0)
            total.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}: loss={total.item():.6f} (recon={recon.item():.6f}, mmd={mmd.item():.6f})')
        else:
            epoch_total, epoch_recon, epoch_mmd = 0.0, 0.0, 0.0
            n_batches = 0
            for batch in dataloader:
                xb = batch[0].to(device)
                optimizer.zero_grad()
                total_b, recon_b, mmd_b = vae.loss_components(xb, beta=1.0)
                total_b.backward()
                optimizer.step()
                epoch_total += total_b.item()
                epoch_recon += recon_b.item()
                epoch_mmd += mmd_b.item()
                n_batches += 1
            if n_batches > 0:
                print(f'Epoch {epoch+1}/{epochs}: loss={epoch_total/n_batches:.6f} '
                      f'(recon={epoch_recon/n_batches:.6f}, mmd={epoch_mmd/n_batches:.6f})')

    # Encode all data
    vae.eval()
    with torch.no_grad():
        mean, logvar = vae.encoder(x)
        z = mean.cpu().numpy()

    # Save encoded features
    np.save(os.path.join(out_dir, 'encoded_mean.npy'), z)

    # Create DataFrame with unique IDs
    col_names = [f'feat_{i}' for i in range(z.shape[1])]
    df_z = pd.DataFrame(z, index=df_proc.index, columns=col_names)
    unique_ids, orig_map = make_unique_ids(df_z.index)
    df_z.index = unique_ids

    # Save ID mapping
    try:
        with open(os.path.join(out_dir, 'id_mapping.json'), 'w') as f:
            json.dump(orig_map, f)
    except Exception:
        pass

    df_z.to_csv(os.path.join(out_dir, 'encoded_mean.csv'))

    # Save model checkpoint
    ckpt_dir = checkpoint_dir or out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(vae.encoder.state_dict(), os.path.join(ckpt_dir, 'encoder.pth'))
    torch.save(vae.decoder.state_dict(), os.path.join(ckpt_dir, 'decoder.pth'))
    
    print(f'\nSaved encoder to {ckpt_dir}/encoder.pth')
    print(f'Latent shape: {z.shape}')
    print(f'Latent std: mean={np.std(z, axis=0).mean():.4f}, '
          f'min={np.std(z, axis=0).min():.4f}, max={np.std(z, axis=0).max():.4f}')

    # Prepare MoPaDi features
    if mopadi_out_dir is None:
        mopadi_out_dir = os.path.join(out_dir, 'mopadi_features')
    
    train_dir = os.path.join(mopadi_out_dir, 'train')
    test_dir = os.path.join(mopadi_out_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    clear_h5_folder(train_dir)
    clear_h5_folder(test_dir)

    # Build labels
    labels = None
    if metadata_csv is not None and os.path.exists(metadata_csv):
        try:
            meta = pd.read_csv(metadata_csv, index_col=0)
            if 'Patient_ID' in meta.columns:
                meta = meta.set_index('Patient_ID')
            if meta.index.duplicated().any():
                meta = meta[~meta.index.duplicated(keep='first')]
            if label_col in meta.columns:
                lab_vals = []
                for uid in df_z.index:
                    orig = orig_map.get(uid, uid)
                    if orig in meta.index:
                        val = meta.loc[orig, label_col]
                        if isinstance(val, (pd.Series, pd.DataFrame)):
                            val = val.iloc[0]
                        lab_vals.append(val if pd.notna(val) else pd.NA)
                    else:
                        lab_vals.append(pd.NA)
                labels = pd.Series(lab_vals, index=df_z.index)
        except Exception as e:
            print(f'Could not load metadata: {e}')

    if labels is None and csv_labels is not None:
        lab_vals = []
        for uid in df_z.index:
            orig = orig_map.get(uid, uid)
            if orig in csv_labels.index:
                val = csv_labels.loc[orig]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.iloc[0]
                lab_vals.append(val if pd.notna(val) else pd.NA)
            else:
                lab_vals.append(pd.NA)
        labels = pd.Series(lab_vals, index=df_z.index)

    # Train/test split
    try:
        train_idx, test_idx = train_test_split(
            df_z.index, test_size=test_size, random_state=random_state,
            stratify=(labels.values if labels is not None else None)
        )
    except Exception:
        train_idx, test_idx = train_test_split(
            df_z.index, test_size=test_size, random_state=random_state
        )

    # Remove any overlap
    inter = set(train_idx) & set(test_idx)
    if inter:
        print(f'Warning: {len(inter)} overlapping IDs, removing from test')
        test_idx = [i for i in test_idx if i not in inter]

    # Save H5 files
    for pid in train_idx:
        save_h5(pid, df_z.loc[pid].values, train_dir)
    for pid in test_idx:
        save_h5(pid, df_z.loc[pid].values, test_dir)

    # Save normalization state
    try:
        train_mat = df_z.loc[train_idx].values.astype(np.float32)
        conds_mean = train_mat.mean(axis=0)
        conds_std = np.maximum(train_mat.std(axis=0), 1e-6)
        
        norm_state = {
            'conds_mean': torch.from_numpy(conds_mean),
            'conds_std': torch.from_numpy(conds_std),
            'feature_dim': z.shape[1],
        }
        
        if labels is not None:
            unique_labels = pd.Series(labels.dropna().unique()).astype(str)
            class2idx = {str(l): int(i) for i, l in enumerate(sorted(unique_labels))}
            norm_state['class2idx'] = class2idx
        
        torch.save(norm_state, os.path.join(mopadi_out_dir, 'norm_state.pth'))
        print(f'Saved norm_state.pth')
    except Exception as e:
        print(f'Could not save norm_state: {e}')

    # Clinical table
    clin_df = pd.DataFrame({'PATIENT': list(df_z.index)})
    if labels is not None:
        clin_df[label_col] = labels.values
    else:
        clin_df[label_col] = pd.NA
    clin_df['split'] = clin_df['PATIENT'].apply(
        lambda x: 'train' if x in set(train_idx) else 'test'
    )
    clin_df.to_csv(os.path.join(mopadi_out_dir, 'clinical_table.csv'), index=False)

    print(f'\nMoPaDi features saved to {mopadi_out_dir}')
    print(f'  Train: {len(train_idx)} patients')
    print(f'  Test: {len(test_idx)} patients')


def main():
    parser = argparse.ArgumentParser(
        description="Train VAE for genomic feature encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required
    parser.add_argument('--csv', required=True, help='CSV file with samples x genes')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    
    # Model architecture
    parser.add_argument('--hidden-dim', default='2048,1024',
                        help='Comma-separated hidden dimensions (default: 2048,1024)')
    parser.add_argument('--latent-dim', type=int, default=512,
                        help='Latent dimension (default: 512)')
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (None for full-batch)')
    
    # Data
    parser.add_argument('--metadata-csv', type=str, default=None,
                        help='Optional metadata CSV with labels')
    parser.add_argument('--label-col', type=str, default='Majority_Subtype_mRNA',
                        help='Label column name')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test split fraction')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--mopadi-out-dir', type=str, default=None,
                        help='MoPaDi features output (default: <out-dir>/mopadi_features)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='External checkpoint directory (default: <out-dir>)')
    
    args = parser.parse_args()
    
    train_vae(
        csv_path=args.csv,
        out_dir=args.out_dir,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mopadi_out_dir=args.mopadi_out_dir,
        metadata_csv=args.metadata_csv,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    main()

"""Minimal demo trainer: preprocess CSV, train VAE for a few epochs, save encoder and print latent stats.

Usage: python -m src.scripts.demo_train --csv path/to/data.csv --out_dir results/demo

Example: python /data/horse/ws/mala059b-rna2wsi/Master-Thesis-Repository/src/scripts/demo_train.py --csv /data/horse/ws/mala059b-rna2wsi/data/brca_gene_expression_with_subtypes.csv --out_dir /data/horse/ws/mala059b-rna2wsi/vae_output/full_train --hidden_dim 2048,1024 --latent_dim 1024 --epochs 100 --batch_size 64
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import h5py
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json

# Ensure project root on sys.path so `src` package imports work when running as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.utils import preprocess_log1p_zscore, inspect_variance
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.models.vae import VAE


def train_demo(csv_path, out_dir, hidden_dim, latent_dim, lr, epochs, batch_size=None,
               mopadi_out_dir=None, metadata_csv=None, label_col='Majority_Subtype_mRNA',
               test_size=0.2, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)
    print('Loaded', df.shape)

    # If Patient_ID is provided as a column, use it as the index so filenames keep patient IDs
    if 'Patient_ID' in df.columns:
        try:
            df = df.set_index('Patient_ID')
            print('Set Patient_ID column as index; new shape:', df.shape)
        except Exception as e:
            print('Could not set Patient_ID as index:', e)

    # Preserve label column from input CSV (if present) before numeric coercion
    csv_labels = None
    if label_col in df.columns:
        try:
            csv_labels = df[label_col].astype(str)
            # drop the label column so numeric conversion doesn't coerce it to NaN/float
            df = df.drop(columns=[label_col])
        except Exception:
            csv_labels = None

    # Ensure dataframe is numeric: try casting, transpose if needed, then coerce non-numeric -> NaN
    def try_cast(df_in):
        try:
            _ = df_in.values.astype(float)
            return df_in
        except Exception:
            return None

    cast_df = try_cast(df)
    if cast_df is None:
        # maybe file is transposed (genes as rows). Try transpose
        df_t = df.T
        cast_df = try_cast(df_t)
        if cast_df is not None:
            df = df_t
            print('Transposed dataframe to match samples x genes orientation:', df.shape)
        else:
            # coerce non-numeric values to NaN, then impute column means
            df = df.apply(pd.to_numeric, errors='coerce')
            n_nan = df.isna().sum().sum()
            print(f'Coerced non-numeric to NaN: total NaNs = {n_nan}')
            # drop columns that are entirely NaN
            all_nan_cols = df.columns[df.isna().all()].tolist()
            if all_nan_cols:
                print('Dropping non-numeric columns:', all_nan_cols[:5], '...')
                df = df.drop(columns=all_nan_cols)
            # impute remaining NaNs with column mean
            df = df.fillna(df.mean())

    print('Variance summary before preprocess:', inspect_variance(df)['gene_var_summary'])

    # Preprocess: log1p + z-score (decoder expects real-valued outputs now)
    df_proc = preprocess_log1p_zscore(df)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.tensor(df_proc.values, dtype=torch.float32)

    # Optionally use a DataLoader for minibatch training; keep full `x` available for final encoding
    if batch_size is not None:
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = None

    x = x.to(device)

    input_dim = x.shape[1]
    if isinstance(hidden_dim, str):
        hidden = [int(h) for h in hidden_dim.split(',')]
    else:
        hidden = hidden_dim

    enc = ProbabilisticEncoder(input_dim, hidden, latent_dim).to(device)
    dec = ProbabilisticDecoder(latent_dim, hidden, input_dim).to(device)
    vae = VAE(enc, dec, device).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        vae.train()
        if dataloader is None:
            optimizer.zero_grad()
            total, recon, mmd = vae.loss_components(x, beta=1.0)
            total.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}: total={total.item():.6f}, recon={recon.item():.6f}, mmd={mmd.item():.6f}')
        else:
            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_mmd = 0.0
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
                print(f'Epoch {epoch+1}/{epochs}: total={epoch_total/n_batches:.6f}, recon={epoch_recon/n_batches:.6f}, mmd={epoch_mmd/n_batches:.6f}')

    # Save encoder and print latent statistics
    vae.eval()
    with torch.no_grad():
        mean, logvar = vae.encoder(x)
        z = mean.cpu().numpy()
    np.save(os.path.join(out_dir, 'encoded_mean.npy'), z)
    # Handle duplicate Patient_IDs: append -DXn to make unique IDs
    def make_unique_ids(index_iter):
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

    col_names = [f'feat_{i}' for i in range(z.shape[1])]
    # create DataFrame with original index first
    df_z = pd.DataFrame(z, index=df_proc.index, columns=col_names)
    # generate unique IDs and apply them to the DataFrame index
    unique_ids, orig_map = make_unique_ids(df_z.index)
    df_z.index = unique_ids
    # save mapping for traceability
    mapping_path = os.path.join(out_dir, 'encoded_id_to_orig_map.json')
    try:
        with open(mapping_path, 'w') as mf:
            json.dump(orig_map, mf)
    except Exception:
        pass
    df_z.to_csv(os.path.join(out_dir, 'encoded_mean.csv'))
    torch.save(vae.encoder.state_dict(), os.path.join(out_dir, 'encoder.pth'))
    print('Saved encoder and encoded_mean; latent shape:', z.shape)
    print('Latent std per-dim summary:', np.std(z, axis=0).mean(), np.std(z, axis=0).min(), np.std(z, axis=0).max())

    # Prepare MoPaDi features: one H5 per patient in train/test folders + clinical table
    if mopadi_out_dir is None:
        mopadi_out_dir = os.path.join(out_dir, 'mopadi_features')
    train_dir = os.path.join(mopadi_out_dir, 'train')
    test_dir = os.path.join(mopadi_out_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # Clear any existing .h5 files in the output folders to avoid leftover duplicates
    def clear_h5_folder(folder):
        try:
            for fname in os.listdir(folder):
                if fname.endswith('.h5'):
                    try:
                        os.remove(os.path.join(folder, fname))
                    except Exception:
                        pass
        except Exception:
            pass

    clear_h5_folder(train_dir)
    clear_h5_folder(test_dir)

    # Load metadata (optional) to obtain labels. Build labels aligned to unique IDs.
    labels = None
    meta = None
    if metadata_csv is not None and os.path.exists(metadata_csv):
        try:
            meta = pd.read_csv(metadata_csv, index_col=0)
            # If metadata has Patient_ID as a column instead of index, set it
            if 'Patient_ID' in meta.columns and 'Patient_ID' not in meta.index:
                try:
                    meta = meta.set_index('Patient_ID')
                except Exception:
                    pass
            # Drop duplicated metadata indices (keep first) to avoid ambiguity
            if meta.index.duplicated().any():
                meta = meta[~meta.index.duplicated(keep='first')]
            if label_col in meta.columns:
                # Build a Series of scalar labels aligned to the new unique IDs using orig_map
                lab_vals = []
                for uid in df_z.index:
                    orig = orig_map.get(uid, uid)
                    if orig in meta.index:
                        val = meta.loc[orig, label_col]
                        # If the metadata index was duplicated, .loc[...] may return a Series/DataFrame; take first scalar
                        if isinstance(val, (pd.Series, pd.DataFrame)):
                            try:
                                val = val.iloc[0]
                            except Exception:
                                val = pd.NA
                        lab_vals.append(val if pd.notna(val) else pd.NA)
                    else:
                        lab_vals.append(pd.NA)
                labels = pd.Series(lab_vals, index=df_z.index)
        except Exception as e:
            print('Could not load metadata CSV for labels:', e)
    # If metadata did not provide labels, fall back to labels extracted from input CSV
    if labels is None and csv_labels is not None:
        lab_vals = []
        for uid in df_z.index:
            orig = orig_map.get(uid, uid)
            if orig in csv_labels.index:
                val = csv_labels.loc[orig]
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    try:
                        val = val.iloc[0]
                    except Exception:
                        val = pd.NA
                lab_vals.append(val if pd.notna(val) else pd.NA)
            else:
                lab_vals.append(pd.NA)
        labels = pd.Series(lab_vals, index=df_z.index)
    

    # Train/test split (stratify if labels available)
    try:
        train_idx, test_idx = train_test_split(
            df_z.index, test_size=test_size, random_state=random_state,
            stratify=(labels.values if labels is not None else None)
        )
    except Exception:
        # fallback to random split
        train_idx, test_idx = train_test_split(df_z.index, test_size=test_size, random_state=random_state)

    # Safety: ensure no overlap between train and test indices (shouldn't happen)
    inter = set(train_idx) & set(test_idx)
    if len(inter) > 0:
        print(f'Warning: found {len(inter)} overlapping ids between train/test splits — removing from test set')
        test_idx = [i for i in test_idx if i not in inter]

    # Save individual H5 files
    def save_h5(pid, arr, outdir):
        path = os.path.join(outdir, f"{pid}.h5")
        with h5py.File(path, 'w') as f:
            data = np.asarray(arr, dtype='float32')
            if data.ndim == 1:
                data = data.reshape(1, -1)
            f.create_dataset('feats', data=data)
        return path

    for pid in train_idx:
        arr = df_z.loc[pid].values
        save_h5(pid, arr, train_dir)
    for pid in test_idx:
        arr = df_z.loc[pid].values
        save_h5(pid, arr, test_dir)

    # --- Save normalization state used for conditioning (conds_mean, conds_std) ---
    try:
        # compute mean/std over training samples (rows of df_z indexed by train_idx)
        train_mat = df_z.loc[train_idx].values.astype(np.float32)
        conds_mean = train_mat.mean(axis=0).astype(np.float32)
        conds_std = train_mat.std(axis=0).astype(np.float32)
        # guard against zero std
        conds_std = np.maximum(conds_std, 1e-6)
        norm_state = {
            'conds_mean': torch.from_numpy(conds_mean).cpu(),
            'conds_std': torch.from_numpy(conds_std).cpu(),
            'feature_dim': int(z.shape[1]),
        }
        # include class2idx mapping if labels are available
        try:
            if labels is not None:
                unique_labels = pd.Series(labels.dropna().unique()).astype(str)
                class2idx = {str(l): int(i) for i, l in enumerate(sorted(unique_labels))}
                norm_state['class2idx'] = class2idx
        except Exception:
            pass

        torch.save(norm_state, os.path.join(mopadi_out_dir, 'norm_state.pth'))
        print(f'Saved normalization state to {os.path.join(mopadi_out_dir, "norm_state.pth")}')
    except Exception as e:
        print('Could not save norm_state.pth:', e)

    # Clinical table
    clin_df = pd.DataFrame({'PATIENT': list(df_z.index)})
    if labels is not None:
        clin_df[label_col] = labels.values
    else:
        clin_df[label_col] = pd.NA
    clin_df['split'] = clin_df['PATIENT'].apply(lambda x: 'train' if x in set(train_idx) else 'test')
    clin_path = os.path.join(mopadi_out_dir, 'clinical_table.csv')
    clin_df.to_csv(clin_path, index=False)
    # Also save original-id mapping alongside the clinical table for audit
    try:
        pd.DataFrame({'unique_id': list(orig_map.keys()), 'orig_patient': list(orig_map.values())}).to_csv(
            os.path.join(mopadi_out_dir, 'encoded_id_to_orig_map.csv'), index=False
        )
    except Exception:
        pass
    print(f'Saved MoPaDi features to {mopadi_out_dir} and clinical table to {clin_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='CSV with samples x genes')
    p.add_argument('--out_dir', default='results/demo', help='Output directory')
    p.add_argument('--hidden_dim', default='512,256', help='Comma-separated hidden dims')
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=None, help='Optional minibatch size')
    p.add_argument('--metadata_csv', type=str, default=None, help='Optional metadata CSV for labels')
    p.add_argument('--label_col', type=str, default='Majority_Subtype_mRNA', help='Column name in metadata CSV with labels')
    p.add_argument('--mopadi_out_dir', type=str, default=None, help='Output directory for MoPaDi H5 files (defaults to <out_dir>/mopadi_features)')
    p.add_argument('--test_size', type=float, default=0.2, help='Test split fraction')
    p.add_argument('--random_state', type=int, default=42, help='Random seed for splits')
    args = p.parse_args()
    train_demo(
        args.csv,
        args.out_dir,
        args.hidden_dim,
        args.latent_dim,
        args.lr,
        args.epochs,
        args.batch_size,
        mopadi_out_dir=args.mopadi_out_dir,
        metadata_csv=args.metadata_csv,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

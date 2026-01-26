"""Minimal demo trainer: preprocess CSV, train VAE for a few epochs, save encoder and print latent stats.

Usage: python -m src.scripts.demo_train --csv path/to/data.csv --out_dir results/demo
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Ensure project root on sys.path so `src` package imports work when running as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.preprocessing.utils import preprocess_log1p_zscore, inspect_variance
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.models.vae import VAE


def train_demo(csv_path, out_dir, hidden_dim, latent_dim, lr, epochs):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)
    print('Loaded', df.shape)

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
    x = torch.tensor(df_proc.values, dtype=torch.float32).to(device)

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
        optimizer.zero_grad()
        total, recon, mmd = vae.loss_components(x, beta=1.0)
        total.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}: total={total.item():.6f}, recon={recon.item():.6f}, mmd={mmd.item():.6f}')

    # Save encoder and print latent statistics
    vae.eval()
    with torch.no_grad():
        mean, logvar = vae.encoder(x)
        z = mean.cpu().numpy()
    np.save(os.path.join(out_dir, 'encoded_mean.npy'), z)
    torch.save(vae.encoder.state_dict(), os.path.join(out_dir, 'encoder.pth'))
    print('Saved encoder and encoded_mean; latent shape:', z.shape)
    print('Latent std per-dim summary:', np.std(z, axis=0).mean(), np.std(z, axis=0).min(), np.std(z, axis=0).max())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='CSV with samples x genes')
    p.add_argument('--out_dir', default='results/demo', help='Output directory')
    p.add_argument('--hidden_dim', default='512,256', help='Comma-separated hidden dims')
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    train_demo(args.csv, args.out_dir, args.hidden_dim, args.latent_dim, args.lr, args.epochs)

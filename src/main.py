import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import csv
import pandas as pd
import yaml
from src.preprocessing.data_loader import GeneExpressionDataLoader
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.models.vae import VAE
from src.loss.mmd_loss import MMDLoss
from plots.plot_utils import plot_loss
import numpy as np

def main(config):
    # Load and preprocess data
    data_loader = GeneExpressionDataLoader(
        config['data']['csv_path'],
        config['data']['columns_to_drop'],
        config['data'].get('id_column')
    )
    data = data_loader.load_data()
    sample_ids = data.index.astype(str).tolist()  # preserve sample identifiers
    preprocessed_data = data_loader.preprocess_data(data)

    # Convert preprocessed data to PyTorch tensor
    data_tensor = torch.tensor(preprocessed_data, dtype=torch.float32)

    # Define device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_tensor = data_tensor.to(device)

    # Define VAE model and loss function
    input_dim = data_tensor.shape[1]
    hidden_dim = config['model']['hidden_dim']
    latent_dim = config['model']['latent_dim']

    encoder = ProbabilisticEncoder(input_dim, hidden_dim, latent_dim).to(device)
    decoder = ProbabilisticDecoder(latent_dim, hidden_dim, input_dim).to(device)
    vae = VAE(encoder, decoder, device)
    loss_fn = MMDLoss()

    # Define optimizer and train VAE
    optimizer = optim.Adam(vae.parameters(), lr=config['training']['learning_rate'])
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    training_log_path = config['output'].get('training_log_path')
    encoded_csv_path = config['output'].get('encoded_csv_path')
    encoded_h5ad_path = config['output'].get('encoded_h5ad_path')

    loss_values = []
    log_records = []

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        recon_loss_epoch = 0
        mmd_loss_epoch = 0
        batches = 0
        for i in range(0, data_tensor.shape[0], batch_size):
            batch = data_tensor[i:i+batch_size]
            optimizer.zero_grad()
            loss, recon, mmd = vae.loss_components(batch, beta=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            recon_loss_epoch += recon.item()
            mmd_loss_epoch += mmd.item()
            batches += 1

        avg_total = total_loss / batches
        avg_recon = recon_loss_epoch / batches
        avg_mmd = mmd_loss_epoch / batches
        loss_values.append(avg_total)
        log_records.append({
            'epoch': epoch + 1,
            'recon_loss': avg_recon,
            'mmd_loss': avg_mmd,
            'total_loss': avg_total
        })
        print(f'Epoch {epoch+1}, Loss: {avg_total:.4f} (recon {avg_recon:.4f}, mmd {avg_mmd:.4f})')

    plot_loss(loss_values)
    
    # Use trained VAE to encode data
    vae.eval()
    with torch.no_grad():
        mean, log_var = vae.encoder(data_tensor)
        encoded_data = mean.cpu().numpy()

    # create output directory if it doesn't exist
    output_dir = os.path.dirname(config['output']['encoded_data_path'])
    os.makedirs(output_dir, exist_ok=True)

    # Save encoded data to file
    np.save(config['output']['encoded_data_path'], encoded_data)

    # Save CSV representation if configured
    if encoded_csv_path:
        csv_dir = os.path.dirname(encoded_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        df_encoded = pd.DataFrame(
            encoded_data,
            index=sample_ids,
            columns=[f"z{i}" for i in range(encoded_data.shape[1])]
        )
        df_encoded.to_csv(encoded_csv_path, index_label="Patient_ID")
        print(f"Saved encoded data CSV to {encoded_csv_path}")

    # Save h5ad representation if configured and anndata is available
    if encoded_h5ad_path:
        try:
            import anndata as ad
        except ImportError:
            print("anndata not installed; skipping h5ad export.")
        else:
            h5ad_dir = os.path.dirname(encoded_h5ad_path)
            if h5ad_dir:
                os.makedirs(h5ad_dir, exist_ok=True)
            var = pd.DataFrame(index=[f"z{i}" for i in range(encoded_data.shape[1])])
            obs = pd.DataFrame(index=sample_ids)
            ad.AnnData(X=encoded_data, obs=obs, var=var).write_h5ad(encoded_h5ad_path)
            print(f"Saved encoded data h5ad to {encoded_h5ad_path}")

    # Save training log to CSV if configured
    if training_log_path:
        log_dir = os.path.dirname(training_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(training_log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'recon_loss', 'mmd_loss', 'total_loss'])
            writer.writeheader()
            writer.writerows(log_records)
        print(f"Saved training log to {training_log_path}")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)
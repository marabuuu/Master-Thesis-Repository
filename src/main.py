import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
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
    data_loader = GeneExpressionDataLoader(config['data']['csv_path'], config['data']['columns_to_drop'])
    data = data_loader.load_data()
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

    loss_values = []

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        for i in range(0, data_tensor.shape[0], batch_size):
            batch = data_tensor[i:i+batch_size]
            optimizer.zero_grad()
            loss = vae.loss(batch, beta=1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loss_values.append(total_loss / (data_tensor.shape[0] / batch_size))
        print(f'Epoch {epoch+1}, Loss: {total_loss / (data_tensor.shape[0] / batch_size)}')

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

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)
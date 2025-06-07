"""Load and train a Variational Autoencoder (VAE) for Tetris game states."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 64
GRID_SIZE = 200
PIECE_CLASSES = 7

class TetrisVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Tetris states.
    This model encodes the game state (grid and piece type) into a latent space,
    and decodes it back to reconstruct the original input.
    """

    def __init__(self, grid_size=200, piece_classes=7, latent_dim=64, hidden_dim=128):

        super(TetrisVAE, self).__init__()
        self.grid_size = grid_size
        self.piece_classes = piece_classes
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc_encoder = nn.Linear(grid_size + piece_classes, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc_decoder = nn.Linear(latent_dim, grid_size + piece_classes)

    def encode(self, x):
        """Encodes the input into mean and variance vectors of the latent space vector z."""
        h = F.relu(self.fc_encoder(x))
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar

    def reparameterise(self, z_mean, z_logvar):
        """Applies the reparameterisation trick to sample z from the latent space."""
        std = torch.exp(0.5 * z_logvar) # σ = exp(0.5 × log(σ²)) = √(σ²)
        eps = torch.randn_like(std)
        return z_mean + eps * std # z = μ + σ × ε

    def decode(self, z):
        """Decodes the latent space vector z back to the original input space."""
        # Outputs raw logits
        decoded_features = self.fc_decoder(z)
        # Sigmoid converts to binary probabilities for grid reconstruction
        grid_out = torch.sigmoid(decoded_features[:, :self.grid_size])
        # Softmax converts to probabilities for piece classification
        piece_out = F.softmax(decoded_features[:, self.grid_size:], dim=1)
        return grid_out, piece_out

    def forward(self, x, training=True):
        """
        Forward pass through the VAE
        """
        # Ensure input is a batch of 1D feature lists (flattened grid + one-hot encoded piece)
        assert isinstance(x, torch.Tensor) and x.dim() == 2, \
            "Input must be a 2D tensor (batch_size, features)"
        assert x.shape[1] == self.grid_size + self.piece_classes, \
            f"Expected shape[1]: {self.grid_size + self.piece_classes}, got {x.shape[1]}"

        if not training:
            self.eval()
            return self.reparameterise(*self.encode(x))

        # Encode to latent space
        z_mean, z_logvar = self.encode(x)

        # Reparameterisation trick
        z = self.reparameterise(z_mean, z_logvar)

        # Decode reconstructions
        grid_recon, piece_recon = self.decode(z)

        return grid_recon, piece_recon, z_mean, z_logvar

def save_model(model, path):
    """Saves the model state dictionary to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Loads the model state dictionary from the specified path."""
    model.load_state_dict(torch.load(path))
    return model

def test_model():
    model = TetrisVAE().to(DEVICE)
    load_model(model, "out/tetris_vae.pth")
    model.forward(training=False)

def vae_loss(grid_true, grid_recon, piece_true, piece_recon, z_mean, z_logvar):
    """Computes the loss for the VAE model."""

    # Grid reconstruction loss (binary cross-entropy)
    grid_bce_loss = F.binary_cross_entropy(
        grid_recon,
        grid_true,
        reduction='sum'
    ) / grid_true.size(0) # Average per sample

    # Piece reconstruction loss (categorical cross-entropy)
    piece_cce_loss = F.cross_entropy(
        torch.log(piece_recon + 1e-9), # Add small constant for numerical stability
        torch.argmax(piece_true, dim=1),
        reduction='mean'
    )

    # Weighted reconstruction loss
    reconstruction_loss = (0.95 * grid_bce_loss) + (0.05 * piece_cce_loss)

    # KL Divergence loss
    kl_div_loss = -0.5 * torch.sum(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),
        dim=1 # Sum KLD over all latent dimensions
    )
    kl_div_loss = kl_div_loss.mean() # Average KLD per sample

    # Total loss
    return reconstruction_loss + kl_div_loss

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    import tetris_dataset

    # Load dataset
    dataloader = DataLoader(
        tetris_dataset.TetrisDataset(device=DEVICE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type=='cuda')
    )

    # Initialise model and optimiser
    model = TetrisVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    def train_epoch():
        model.train()
        total_loss = 0

        for batch_idx, sample_batch in enumerate(dataloader):

            grid_true, piece_true = sample_batch[:, :-7], sample_batch[:, -7:]

            optimiser.zero_grad()

            # Forward pass
            grid_recon, piece_recon, z_mean, z_logvar = model(sample_batch)

            # Calculate loss
            loss = vae_loss(
                grid_true, grid_recon,
                piece_true, piece_recon,
                z_mean, z_logvar
            )

            # Backward pass
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        return total_loss / len(dataloader)

    # Train for one epoch
    avg_loss = train_epoch()
    print(f"Average Loss: {avg_loss:.4f}")

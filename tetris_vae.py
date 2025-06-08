"""Load and train a Variational Autoencoder (VAE) for Tetris game states."""
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np

import tetris_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 64
GRID_SIZE = 200
PIECE_CLASSES = 7
NUM_EPOCHS = 100
PATIENCE = 10 # Epochs to wait before early stopping

class TetrisVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Tetris states.
    This model encodes the game state (grid and piece type) into a latent space,
    and decodes it back to reconstruct the original input.
    """

    def __init__(
            self,
            grid_size=200,
            piece_classes=7,
            latent_dim=64,
            hidden_dim=128,
            dropout_rate=0.2
        ):

        super(TetrisVAE, self).__init__()
        self.grid_size = grid_size
        self.piece_classes = piece_classes
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(grid_size + piece_classes, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Latent space
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers with symmetric structure
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Multi-head output
        self.fc_grid = nn.Linear(256, grid_size)
        self.fc_piece = nn.Linear(256, piece_classes)

    def encode(self, x):
        """Encodes the input into mean and variance vectors of the latent space vector z."""
        x = self.encoder(x)
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar

    def reparameterise(self, z_mean, z_logvar):
        """Applies the reparameterisation trick to sample z from the latent space distribution."""
        std = torch.exp(0.5 * z_logvar) # σ = exp(0.5 × log(σ²)) = √(σ²)
        eps = torch.randn_like(std)
        return z_mean + eps * std # z = μ + σ × ε

    def decode(self, z):
        """
        Decodes the latent space vector z back to the original input space.
        returns probabilities for reconstructed grid and
        raw logits for the reconstructed piece.
        """
        # Outputs raw logits
        z = self.decoder(z)
        # Sigmoid converts to binary probabilities for grid reconstruction
        grid_out = torch.sigmoid(self.fc_grid(z))
        # return raw logits for piece classification
        piece_logits = self.fc_piece(z)
        return grid_out, piece_logits

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

def encode_sample_to_latent(sample):
    """Maps input sample to latent space"""
    model = TetrisVAE().to(DEVICE)
    model = load_model(model, "./out/best_model.pth")
    return model(sample, training=False)

def vae_loss(
        grid_true, grid_recon,
        piece_true, piece_recon,
        z_mean, z_logvar,
        epoch
    ):
    """Computes the loss for the VAE model."""

    # Grid reconstruction loss (binary cross-entropy)

    # Totals per pixel_ce between each reconstructed grid and true grid
    # then divides over all dimensions (num_pixels * batch_size)
    # giving mean pixel_ce in the batch
    pixel_bce = F.binary_cross_entropy(
        grid_recon, grid_true, reduction='mean'
    )

    grid_bce = pixel_bce * grid_true.size(1) # Scale by number of pixels

    # Piece reconstruction loss (categorical cross-entropy)

    # Mean piece_ce per sample in the batch
    piece_ce = F.cross_entropy(
        piece_recon, # raw logits
        torch.argmax(piece_true, dim=1),
        reduction='mean'
    )

    reconstruction_loss = pixel_bce + piece_ce

    # KL Divergence loss
    kl_div_loss = -0.5 * torch.sum(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),
        dim=1 # Sum KLD over all latent dimensions
    )
    kl_div_loss = kl_div_loss.mean() # Average KLD per sample

    # KL annealing allows the reconstruction loss to dominate in early epochs
    kl_weight = min(1.0, epoch / NUM_EPOCHS)

    elbo_loss = reconstruction_loss + kl_weight * kl_div_loss

    return elbo_loss, pixel_bce, piece_ce, kl_div_loss

def train_model():
    """
    Trains the Tetris VAE model on the Tetris dataset.
    """

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load and split dataset
    full_dataset = tetris_dataset.TetrisDataset(device=DEVICE)
    # 80 / 20 split
    train_set, validation_set = random_split(full_dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_set,
        batch_size=BATCH_SIZE
    )

    # Initialise model and optimiser
    model = TetrisVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

    epochs_no_improvement = 0
    piece_correct = 0
    total_samples = 0
    history = defaultdict(list)
    best_validation_loss = float('inf')

    # Main loop
    for epoch in range(NUM_EPOCHS):

        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:

            optimiser.zero_grad()

            grid_true, piece_true = batch[:, :-7], batch[:, -7:]
            grid_recon, piece_recon, z_mean, z_logvar = model(batch)

            loss, pixel_bce, piece_ce, kl_div_loss = vae_loss(
                grid_true, grid_recon,
                piece_true, piece_recon,
                z_mean, z_logvar,
                epoch=epoch
            )

            loss.backward()
            optimiser.step()
            train_loss += loss.item()

        # Stores the last training batch's loss values for logging
        history['pixel_bce'].append(round(pixel_bce.item(), 3))
        history['piece_ce'].append(round(piece_ce.item(), 3))
        history['kl_div_loss'].append(round(kl_div_loss.item(), 3))

        # Validation phase
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in validation_loader:

                grid_true, piece_true = batch[:, :-7], batch[:, -7:]

                grid_recon, piece_recon, z_mean, z_logvar = model(batch)
                loss, pixel_bce, piece_ce, kl_div_loss = vae_loss(
                    grid_true, grid_recon,
                    piece_true, piece_recon,
                    z_mean, z_logvar,
                    epoch=epoch
                )

                validation_loss += loss.item()

                _, preds = torch.max(F.softmax(piece_recon, dim=1), 1)
                # Count correct predictions for piece classification
                piece_correct += (preds == torch.argmax(piece_true, dim=1)).sum().item()
                # Count total samples processed, used to calculate piece accuracy
                total_samples += piece_true.size(0)

        # Update metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_validation_loss = validation_loss / len(validation_loader)
        piece_accuracy = piece_correct / total_samples

        history['train_loss'].append(round(avg_train_loss, 3))
        history['validation_loss'].append(round(avg_validation_loss, 3))
        history['piece_accuracy'].append(round(piece_accuracy, 3))

        # Update learning rate
        scheduler.step(avg_validation_loss)

        # Save history every epoch
        np.save("./out/tetris_vae_history.npy", history)

        # Early stopping check
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            epochs_no_improvement = 0
            save_model(model, "./out/best_model.pth")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= PATIENCE:
                break

if __name__ == "__main__":
    train_model()

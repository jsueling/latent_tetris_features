"""Load and train a Variational Autoencoder (VAE) for Tetris game states."""
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

import tetris_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 64
GRID_SIZE = 200
PIECE_CLASSES = 7
NUM_EPOCHS = 200
WARMUP_EPOCHS = 10 # Epochs to allow KL annealing
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
            dropout_rate=0.3
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

    def forward(self, x, training=None):
        """
        Forward pass through the VAE
        """
        # Ensure input is a batch of 1D feature lists (flattened grid + one-hot encoded piece)
        assert isinstance(x, torch.Tensor) and x.dim() == 2, \
            "Input must be a 2D tensor (batch_size, features)"
        assert x.shape[1] == self.grid_size + self.piece_classes, \
            f"Expected shape[1]: {self.grid_size + self.piece_classes}, got {x.shape[1]}"
        assert training in [True, False], \
            "training must be set to True or False"

        if training is True:
            self.train()
        else:
            self.eval()

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
    """Computes the loss for the VAE model. Returns losses per sample of this batch"""

    # Grid reconstruction loss (binary cross-entropy)

    # Totals per pixel_ce between each reconstructed grid and true grid
    # then divides over all dimensions (num_pixels * batch_size)
    # giving mean pixel_ce in the batch
    pixel_bce = F.binary_cross_entropy(
        grid_recon, grid_true, reduction='mean'
    )

    # grid_bce = pixel_bce * grid_true.size(1) # Scale by number of pixels

    # Piece reconstruction loss (categorical cross-entropy)

    # Mean piece_ce per sample in the batch
    piece_ce = F.cross_entropy(
        piece_recon, # raw logits
        torch.argmax(piece_true, dim=1),
        reduction='mean'
    )

    reconstruction_loss = 5 * pixel_bce + piece_ce

    # KL Divergence loss
    kl_div_loss = -0.5 * torch.sum(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),
        dim=1 # Sum KLD over all latent dimensions
    )

    kl_div_loss = kl_div_loss.mean() # Average KLD per sample

    # KL annealing allows the reconstruction loss to dominate in early epochs
    max_kl_weight = 0.05
    kl_weight = min(max_kl_weight, epoch / WARMUP_EPOCHS * max_kl_weight)

    elbo_loss = reconstruction_loss + kl_weight * kl_div_loss

    return elbo_loss, pixel_bce, piece_ce, kl_div_loss

def train_model():
    """
    Trains the Tetris VAE model on the Tetris dataset.
    """

    # Load and split dataset
    full_dataset = tetris_dataset.TetrisDataset(device=DEVICE)

    print(f"Dataset size: {len(full_dataset)} samples")

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
    history = defaultdict(list)
    best_validation_loss = float('inf')
    validation_samples = len(validation_loader.dataset)
    training_samples = len(train_loader.dataset)

    # Main loop
    for epoch in range(NUM_EPOCHS):

        # Training phase
        train_loss = 0

        for batch in train_loader:

            optimiser.zero_grad()

            grid_true, piece_true = batch[:, :-7], batch[:, -7:]
            grid_recon, piece_recon, z_mean, z_logvar = model(batch, training=True)

            loss, pixel_bce, piece_ce, kl_div_loss = vae_loss(
                grid_true, grid_recon,
                piece_true, piece_recon,
                z_mean, z_logvar,
                epoch=epoch
            )

            loss.backward()
            optimiser.step()
            train_loss += loss.item() * batch.size(0)

        # Validation phase
        validation_loss = 0
        validation_correct_pieces = 0
        validation_correct_pixels = 0
        validation_pixel_bce = 0
        validation_piece_ce = 0
        validation_kl_div_loss = 0

        with torch.no_grad():
            for batch in validation_loader:

                grid_true, piece_true = batch[:, :-7], batch[:, -7:]

                grid_recon, piece_recon, z_mean, z_logvar = model(batch, training=False)

                loss, pixel_bce, piece_ce, kl_div_loss = vae_loss(
                    grid_true, grid_recon,
                    piece_true, piece_recon,
                    z_mean, z_logvar,
                    epoch=epoch
                )

                validation_pixel_bce += pixel_bce.item() * batch.size(0)
                validation_piece_ce += piece_ce.item() * batch.size(0)
                validation_kl_div_loss += kl_div_loss.item() * batch.size(0)

                validation_loss += loss.item() * batch.size(0)

                piece_predictions = torch.argmax(F.softmax(piece_recon, dim=1), dim=1)
                piece_truths = torch.argmax(piece_true, dim=1)
                # Count correct predictions for piece classification
                validation_correct_pieces += (piece_predictions == piece_truths).sum().item()

                pixel_predictions = (grid_recon > 0.5).float()
                # Count correct pixel predictions
                validation_correct_pixels += (pixel_predictions == grid_true).float().sum().item()

        # Per sample metrics calculated from the validation set
        avg_train_loss = train_loss / training_samples
        avg_validation_loss = validation_loss / validation_samples

        avg_piece_accuracy = validation_correct_pieces / validation_samples
        avg_pixel_accuracy = validation_correct_pixels / (validation_samples * GRID_SIZE)
        avg_pixel_bce = validation_pixel_bce / validation_samples
        avg_piece_ce = validation_piece_ce / validation_samples
        avg_kl_div_loss = validation_kl_div_loss / validation_samples

        history['avg_train_loss'].append(avg_train_loss)
        history['avg_validation_loss'].append(avg_validation_loss)
        history['avg_piece_accuracy'].append(avg_piece_accuracy)
        history['avg_pixel_accuracy'].append(avg_pixel_accuracy)
        history['avg_pixel_bce'].append(avg_pixel_bce)
        history['avg_piece_ce'].append(avg_piece_ce)
        history['avg_kl_div_loss'].append(avg_kl_div_loss)

        # Update learning rate
        if epoch > WARMUP_EPOCHS:
            scheduler.step(avg_validation_loss)

        # Save history every epoch
        np.save("./out/tetris_vae_history.npy", history)

        if epoch <= WARMUP_EPOCHS:
            continue

        # Early stopping check
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            epochs_no_improvement = 0
            save_model(model, "./out/best_model.pth")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= PATIENCE:
                break

    return history, epoch

def plot_history(history_dict):
    """
    Plots the training history of the VAE model.
    """

    plt.figure(figsize=(15, 12))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history_dict['avg_train_loss'], label='Training Loss')
    plt.plot(history_dict['avg_validation_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(2, 2, 2)
    plt.plot(history_dict['avg_piece_accuracy'], label='Piece Accuracy')
    plt.plot(history_dict['avg_pixel_accuracy'], label='Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Piece and Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot component losses
    plt.subplot(2, 2, 3)
    plt.plot(history_dict['avg_pixel_bce'], label='Pixel BCE')
    plt.plot(history_dict['avg_piece_ce'], label='Piece CE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Component Reconstruction Losses')
    plt.legend()
    plt.grid(True)

    # Plot KL divergence loss
    plt.subplot(2, 2, 4)
    plt.plot(history_dict['avg_kl_div_loss'], label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('KL Divergence Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('./out/training_history.png')
    plt.show()

def reconstruction_test():
    """
    Tests the reconstruction quality of Tetris states using the trained VAE model.
    """
    model = TetrisVAE().to(DEVICE)
    model = load_model(model, "./out/best_model.pth")
    data_loader = DataLoader(tetris_dataset.TetrisDataset(device=DEVICE), shuffle=True)

    for _ in range(10):
        true_sample = next(iter(data_loader))
        with torch.no_grad():
            grid_recon, piece_recon, _, _ = model(true_sample, training=False)
        piece = torch.zeros(1, 7)
        piece[0, torch.argmax(F.softmax(piece_recon, dim=1), dim=1)] = 1
        reconstructed_sample = torch.cat([(grid_recon > 0.5).float(), piece], dim=1)

        print("True sample:")
        for row in true_sample:
            for col_idx, col_val in enumerate(row):
                if col_idx <= 200 and col_idx % 10 == 0:
                    print()
                print(int(col_val), end='')
            print()
            print("-"*20)

        print("Reconstructed sample:")
        for row in reconstructed_sample:
            for col_idx, col_val in enumerate(row):
                if col_idx <= 200 and col_idx % 10 == 0:
                    print()
                print(int(col_val), end='')
            print()
            print("-"*20)

def latent_space_interpolation_test():
    """
    Tests the latent space interpolation of the VAE model.
    """
    pass

if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    history, epochs = train_model()

    print(f"Training completed after {epochs} epochs.")

    plot_history(history)

    # reconstruction_test()

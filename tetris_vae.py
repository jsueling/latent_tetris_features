"""Load and train a Variational Autoencoder (VAE) for Tetris game states."""
from collections import defaultdict
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

import tetris_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 16
GRID_SIZE = 200
NUM_EPOCHS = 200
WARMUP_EPOCHS = 50 # Epochs to wait before early stopping may occur
PATIENCE = 20 # Epochs to wait before early stopping after no improvement

class TetrisVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Tetris states.
    This model encodes the game state (grid) into a latent space,
    and decodes it back to reconstruct the original input.
    """

    def __init__(
            self,
            grid_size=200,
            latent_dim=16,
        ):

        super(TetrisVAE, self).__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        # Encoder
        encoder_hidden_dims = [512, 256, 128, 64, 32]

        encoder_layers = []
        input_dim = grid_size
        for output_dim in encoder_hidden_dims:
            encoder_layers.append(nn.Linear(input_dim, output_dim))
            encoder_layers.append(nn.BatchNorm1d(output_dim))
            encoder_layers.append(nn.ReLU())
            input_dim = output_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mean = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Decoder
        decoder_hidden_dims = [32, 64, 128, 256]

        decoder_layers = []
        input_dim = latent_dim
        for output_dim in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(input_dim, output_dim))
            decoder_layers.append(nn.BatchNorm1d(output_dim))
            decoder_layers.append(nn.ReLU())
            input_dim = output_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output layer
        self.fc_grid = nn.Linear(decoder_hidden_dims[-1], grid_size)

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
        returns logits of the reconstructed grid
        """
        z = self.decoder(z)
        return self.fc_grid(z)

    def forward(self, x, training=None):
        """
        Forward pass through the VAE
        """
        # Ensure input is a batch of 1D feature lists (flattened grids)
        assert isinstance(x, torch.Tensor) and x.dim() == 2, \
            "Input must be a 2D tensor (batch_size, features)"
        assert x.shape[1] == self.grid_size, \
            f"Expected shape[1]: {self.grid_size}, got {x.shape[1]}"
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
        grid_recon_logits = self.decode(z)

        return grid_recon_logits, z_mean, z_logvar

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
        grid_true, grid_recon_logits,
        z_mean, z_logvar,
        epoch
    ):
    """Computes the loss for the VAE model. Returns losses per sample of this batch"""

    # Grid reconstruction loss (binary cross-entropy)

    # Totals per pixel_ce between each reconstructed grid and true grid
    # then divides over all dimensions (num_pixels * batch_size)
    # giving mean pixel_ce in the batch
    pixel_bce = F.binary_cross_entropy_with_logits(
        grid_recon_logits, grid_true, reduction='mean'
    )

    reconstruction_loss = pixel_bce

    # Kullback-Leibler Divergence loss between the latent space distribution
    # and the standard normal distribution N(0, 1)

    kl_div_loss = (-0.5 * torch.sum(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),
        dim=1 # Sum KLD over all latent dimensions
    )).mean()  # Mean over batch

    # KL weight is scaled linearly during the warmup phase to allow the model
    # to learn to reconstruct inputs well before regularising the latent space
    kl_weight = 1e-3 * min(epoch / WARMUP_EPOCHS, 1.0)

    elbo_loss = reconstruction_loss + kl_weight * kl_div_loss

    return elbo_loss, pixel_bce, kl_div_loss

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

    # Adam optimiser with learning rate scheduling
    # to reduce the learning rate when validation loss plateaus
    # and L2 regularisation to prevent overfitting
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=3, factor=0.5)

    epochs_no_improvement = 0
    history = defaultdict(list)
    best_validation_loss = float('inf')
    validation_samples = len(validation_loader.dataset)
    training_samples = len(train_loader.dataset)

    # Main loop
    for epoch in range(NUM_EPOCHS):

        # Training phase
        train_loss = 0

        for grid_true in train_loader:

            optimiser.zero_grad()

            batch_size = grid_true.size(0)

            grid_recon_logits, z_mean, z_logvar = model(grid_true, training=True)

            loss, pixel_bce, kl_div_loss = vae_loss(
                grid_true, grid_recon_logits,
                z_mean, z_logvar,
                epoch=epoch
            )

            loss.backward()
            optimiser.step()
            train_loss += loss.item() * batch_size

        # Validation phase
        validation_loss = 0
        validation_correct_pixels = 0
        validation_pixel_bce = 0
        validation_kl_div_loss = 0

        with torch.no_grad():
            for grid_true in validation_loader:

                batch_size = grid_true.size(0)

                grid_recon_logits, z_mean, z_logvar = model(grid_true, training=False)

                loss, pixel_bce, kl_div_loss = vae_loss(
                    grid_true, grid_recon_logits,
                    z_mean, z_logvar,
                    epoch=epoch
                )

                validation_pixel_bce += pixel_bce.item() * batch_size
                validation_kl_div_loss += kl_div_loss.item() * batch_size

                validation_loss += loss.item() * batch_size

                pixel_predictions = (torch.sigmoid(grid_recon_logits) > 0.5).float()
                # Count correct pixel predictions
                validation_correct_pixels += (pixel_predictions == grid_true).float().sum().item()

        # Per sample metrics calculated from the validation set
        avg_train_loss = train_loss / training_samples
        avg_validation_loss = validation_loss / validation_samples

        avg_pixel_accuracy = validation_correct_pixels / (validation_samples * GRID_SIZE)
        avg_pixel_bce = validation_pixel_bce / validation_samples
        avg_kl_div_loss = validation_kl_div_loss / validation_samples

        history['avg_train_loss'].append(avg_train_loss)
        history['avg_validation_loss'].append(avg_validation_loss)
        history['avg_pixel_accuracy'].append(avg_pixel_accuracy)
        history['avg_pixel_bce'].append(avg_pixel_bce)
        history['avg_kl_div_loss'].append(avg_kl_div_loss)

        # Save history every epoch
        np.save("./out/tetris_vae_history.npy", history)

        # Skip early stopping and learning rate scheduling during warmup
        if epoch <= WARMUP_EPOCHS:
            continue

        # Learning rate scheduling
        scheduler.step(avg_validation_loss)

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
    plt.plot(history_dict['avg_pixel_accuracy'], label='Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot component losses
    plt.subplot(2, 2, 3)
    plt.plot(history_dict['avg_pixel_bce'], label='Pixel BCE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Component Reconstruction Loss')
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
    data_iterator = iter(data_loader)

    for _ in range(10):
        true_sample = next(data_iterator)

        with torch.no_grad():
            grid_recon_logits, _, _ = model(true_sample, training=False)

        reconstructed_sample = (torch.sigmoid(grid_recon_logits) > 0.5).float()

        print("True sample:")
        for row in true_sample:
            for col_idx, col_val in enumerate(row):
                if col_idx % 10 == 0:
                    print()
                print(int(col_val), end='')
            print()
            print("-"*20)

        print("Reconstructed sample:")
        for row in reconstructed_sample:
            for col_idx, col_val in enumerate(row):
                if col_idx % 10 == 0:
                    print()
                print(int(col_val), end='')
            print()
            print("-"*20)

def latent_space_interpolation_test():
    """
    Tests the latent space interpolation of the VAE model.
    """
    pass

def map_latent_space_to_grid():
    """
    Maps the latent space of the VAE model to visualise in 2d.
    """
    model = TetrisVAE(latent_dim=LATENT_DIM).to(DEVICE)
    model = load_model(model, "./out/best_model.pth")
    plt.figure(figsize=(15, 12))

    dataset = tetris_dataset.TetrisDataset(device=DEVICE)
    indices = torch.randperm(len(dataset))[:10000]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=128)
    with torch.no_grad():
        for sample in dataloader:
            _, z_mean, z_logvar = model(sample, training=False)
            z = model.reparameterise(z_mean, z_logvar)
            plt.scatter(z[:, 0].cpu(), z[:, 1].cpu(), alpha=0.5)

    plt.xlabel('z[0] (latent dimension 1)')
    plt.ylabel('z[1] (latent dimension 2)')
    plt.title('Latent Space Mapping of Tetris States')
    plt.grid(True)
    plt.savefig('./out/latent_space.png')
    plt.show()

if __name__ == "__main__":

    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    history, epochs = train_model()

    print(f"Training completed after {epochs} epochs.")

    plot_history(history)

    # reconstruction_test()
    # latent_space_interpolation_test()
    # map_latent_space_to_grid()

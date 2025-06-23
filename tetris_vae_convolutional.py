"""Load and train a Variational Autoencoder (VAE) for Tetris game states."""
from collections import defaultdict
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm

import tetris_dataset
import tetris_vae_utils as utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LATENT_DIM = 2
MAX_KLD_WEIGHT = 1.0
GRID_SIZE = 200
GRID_HEIGHT = 20
GRID_WIDTH = 10
NUM_EPOCHS = 200
WARMUP_EPOCHS = 1 # Epochs to wait before early stopping may occur
PATIENCE = 0 # Epochs to wait before early stopping after no improvement

class TetrisConvolutionalVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for Tetris states using a convolutional architecture.
    This model encodes the game state (grid) into a latent space,
    and decodes it back to reconstruct the original input.
    """

    def __init__(
            self,
            grid_height=20,
            grid_width=10,
            latent_dim=LATENT_DIM,
        ):

        super(TetrisConvolutionalVAE, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.latent_dim = latent_dim

        # Encoder
        encoder_channels = [32, 64, 128, 256]

        encoder_layers = []
        input_channels = 1
        for output_channels in encoder_channels:
            encoder_layers.append(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
            encoder_layers.append(nn.BatchNorm2d(output_channels))
            encoder_layers.append(nn.LeakyReLU())
            input_channels = output_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate the flattened encoder output size dynamically after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.grid_height, self.grid_width)
            dummy_output = self.encoder(dummy_input)
            self.conv_output_size = int(np.prod(dummy_output.shape))
            self.conv_output_shape = dummy_output.shape[1:]

        # Latent space
        self.fc_mean = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        # Decoder
        decoder_channels = [256, 128, 64]

        # Maps from latent space to the number of features of the first decoder layer
        self.fc_decode = nn.Linear(latent_dim, self.conv_output_size)

        decoder_layers = []
        input_channels = decoder_channels[0]
        for output_channels in decoder_channels:
            decoder_layers.append(
                nn.ConvTranspose2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                )
            )
            decoder_layers.append(nn.BatchNorm2d(output_channels))
            decoder_layers.append(nn.LeakyReLU())
            input_channels = output_channels

        # Final layer
        decoder_layers.append(
            nn.ConvTranspose2d(
                input_channels,
                1,
                kernel_size=(5, 3),
                stride=1,
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encodes the input into mean and variance vectors of the latent space vector z."""
        x = self.encoder(x)
        x = x.flatten(start_dim=1)  # Flatten the convolutional encoder output
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
        # Preprocess z for decoding - expand latent_dim to flattened conv_output_size
        z = self.fc_decode(z)
        # Reshape z to match the output shape of the last convolutional layer
        z = z.view(-1, *self.conv_output_shape)
        # Invert the convolutional layers to reconstruct the grid
        z = self.decoder(z)
        return z

    def forward(self, x, training=None):
        """
        Forward pass through the VAE
        """
        assert isinstance(x, torch.Tensor) and x.dim() == 4, \
            "Input must be a tensor of shape (B, C, H, W)"
        assert x.shape[1] == 1, \
            f"Expected shape[1]: 1 (single channel), got {x.shape[1]}"
        assert x.shape[2] == GRID_HEIGHT, \
            f"Expected shape[2]: {GRID_HEIGHT}, got {x.shape[2]}"
        assert x.shape[3] == GRID_WIDTH, \
            f"Expected shape[3]: {GRID_WIDTH}, got {x.shape[3]}"
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

def vae_loss(
        grid_true, grid_recon_logits,
        z_mean, z_logvar,
        epoch,
        max_kld_weight
    ):
    """Computes the loss for the VAE model. Returns losses per sample of this batch"""

    # Grid reconstruction loss (binary cross-entropy)

    # Totals per pixel_ce between each reconstructed grid and true grid
    # then divides over all dimensions (num_pixels * batch_size)
    # giving mean pixel_ce in the batch
    pixel_bce = F.binary_cross_entropy_with_logits(
        grid_recon_logits, grid_true, reduction='mean'
    )

    # As there are GRID_SIZE pixels in each grid, we multiply by the mean pixel_bce
    # to get the total reconstruction loss per grid sample
    reconstruction_loss = pixel_bce * GRID_SIZE

    # Kullback-Leibler Divergence loss between the current latent space distribution
    # and the standard normal distribution N(0, 1)

    kl_div_loss = (-0.5 * torch.sum(
        1 + z_logvar - z_mean.pow(2) - z_logvar.exp(),
        dim=1 # Sum KLD over all latent dimensions
    )).mean()  # Mean over batch

    # KL weight is scaled linearly during the warmup phase to allow the model
    # to learn to reconstruct inputs well before regularising the latent space
    kld_weight = max_kld_weight * min(epoch / WARMUP_EPOCHS, 1.0)

    elbo_loss = reconstruction_loss + kld_weight * kl_div_loss

    return elbo_loss, pixel_bce, kl_div_loss

def train_model(
        latent_dim=LATENT_DIM,
        max_kld_weight=MAX_KLD_WEIGHT,
        train_loader=None,
        validation_loader=None
    ):
    """
    Trains the Tetris VAE model on the Tetris dataset.
    """

    # Initialise model and optimiser
    model = TetrisConvolutionalVAE(latent_dim=latent_dim).to(DEVICE)

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
    for epoch in tqdm(
        range(NUM_EPOCHS),
        desc=f"{latent_dim}D, max_kld={max_kld_weight}",
        unit="epoch"
    ):

        # Training phase
        train_loss = 0

        for grid_true in train_loader:

            optimiser.zero_grad()

            batch_size = grid_true.size(0)

            grid_recon_logits, z_mean, z_logvar = model(grid_true, training=True)

            loss, pixel_bce, kl_div_loss = vae_loss(
                grid_true, grid_recon_logits,
                z_mean, z_logvar,
                epoch=epoch,
                max_kld_weight=max_kld_weight
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
                    epoch=epoch,
                    max_kld_weight=max_kld_weight
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
        np.save(f"./out/conv_dim_{latent_dim}_max_kld_{max_kld_weight}.npy", history)

        # Skip early stopping and learning rate scheduling during warmup
        if epoch <= WARMUP_EPOCHS:
            continue

        # Learning rate scheduling
        scheduler.step(avg_validation_loss)

        # Early stopping check
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            epochs_no_improvement = 0
            utils.save_model(model, f"./out/conv_dim_{latent_dim}_max_kld_{max_kld_weight}.pth")
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= PATIENCE:
                break

if __name__ == "__main__":

    # Set random seed for reproducibility
    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

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

    for latent_dim in range(2, 17):
        for max_kld_weight in [0.2, 0.4, 1.0, 2.0, 4.0]:
            train_model(
                latent_dim=latent_dim,
                max_kld_weight=max_kld_weight,
                train_loader=train_loader,
                validation_loader=validation_loader
            )

    latent_dim = 8
    kld_weight = 1.25
    history_file_path = f"../out/dim_{latent_dim}_max_kld_{kld_weight}.npy"
    model_file_path = f"../out/dim_{latent_dim}_max_kld_{kld_weight}.pth"

    utils.plot_history(history_file_path)

    vae_model = TetrisConvolutionalVAE(latent_dim=latent_dim).to(DEVICE)
    vae_model = utils.load_model(vae_model, model_file_path)
    data = tetris_dataset.TetrisDataset(device=DEVICE)

    # utils.reconstruction_test(vae_model, data)
    # utils.map_latent_space_to_grid(vae_model, data, latent_dim=latent_dim)
    # utils.latent_space_traversal(vae_model, data, latent_dim=latent_dim)

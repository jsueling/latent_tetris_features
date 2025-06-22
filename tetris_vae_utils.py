"""A collection of utility functions for training and testing a Variational Autoencoder (VAE) on Tetris states."""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 8
GRID_HEIGHT = 20
GRID_WIDTH = 10

def plot_history(history_file_path=None):
    """
    Plots the training history of the VAE model.
    """

    history_dict = np.load(history_file_path, allow_pickle=True).item()

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

def reconstruction_test(model, dataset):
    """
    Tests the reconstruction quality of Tetris states using the trained VAE model.
    """

    data_loader = DataLoader(dataset, shuffle=True)
    data_iterator = iter(data_loader)

    num_tests = 10

    for _ in range(num_tests):
        true_sample = next(data_iterator)

        with torch.no_grad():
            grid_recon_logits, _, _ = model(true_sample, training=False)

        reconstructed_sample = (torch.sigmoid(grid_recon_logits) > 0.5).float()

        print("True sample:")
        true_sample = true_sample.int().detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
        print(true_sample)

        print("Reconstructed sample:")
        reconstructed_sample = reconstructed_sample.int().detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
        print(reconstructed_sample)

def latent_space_interpolation_test():
    """
    Tests the latent space interpolation of the VAE model.
    """
    pass

def latent_space_traversal(model, dataset, latent_dim=LATENT_DIM):
    """
    Tests for disentangled latent representations created by the VAE by
    visually comparing a single sample which is perturbed along each latent dimension.
    As found in Fig 7 of the beta-VAE paper: https://openreview.net/forum?id=Sy2fzU9gl
    """
    data_loader = DataLoader(dataset, shuffle=True)
    data_iterator = iter(data_loader)

    sample = next(data_iterator) # 200 dimensional Tetris state

    # "3 standard deviations around the unit gaussian prior
    # while keeping the remaining latent units fixed"
    # Fig 7, https://openreview.net/forum?id=Sy2fzU9gl
    num_samples_per_dimension = 7
    perturbation_range = 3.0

    all_dimension_samples = []

    with torch.no_grad():
        _, z_mean, _ = model(sample, training=False)

    for dim_index in range(latent_dim):

        dimension_samples = []

        # Create a grid of latent space vectors by varying one dimension
        for perturbation_value in np.linspace(
            -perturbation_range,
            perturbation_range,
            num=num_samples_per_dimension
        ):
            z_modified = z_mean.clone()
            z_modified[:, dim_index] += perturbation_value
            grid_recon_logits = model.decode(z_modified).squeeze(0)
            reconstructed_sample = (torch.sigmoid(grid_recon_logits) > 0.5).float()
            reconstructed_sample = reconstructed_sample.detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH)
            dimension_samples.append(grid_recon_logits.detach().numpy().reshape(GRID_HEIGHT, GRID_WIDTH))

        all_dimension_samples.append(dimension_samples)

    # Visualise the reconstructed samples for each dimension
    _, axes = plt.subplots(
        latent_dim,
        num_samples_per_dimension,
        figsize=(num_samples_per_dimension, latent_dim * 1.5)
    )

    for dim_index in range(latent_dim):
        for sample_index in range(num_samples_per_dimension):
            ax = axes[dim_index, sample_index]
            ax.imshow(
                all_dimension_samples[dim_index][sample_index],
                cmap='Blues',
                interpolation='nearest',
                vmin=0,
                vmax=1
            )
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('./out/latent_space_traversal.png')
    plt.show()

def map_latent_space_to_grid(model, dataset, latent_dim=LATENT_DIM):
    """
    Maps the latent space of the VAE model to visualise in 2d or 3d.
    """

    if latent_dim not in [2, 3]:
        print(f"Latent space visualisation is only supported for latent_dim 2 or 3. \
              Current latent_dim is {latent_dim}.")
        return

    plt.figure(figsize=(15, 12))
    ax = plt.axes(projection='3d' if latent_dim == 3 else None)

    indices = torch.randperm(len(dataset))[:10000]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=128)
    all_z = []
    with torch.no_grad():
        for sample in dataloader:
            _, z_mean, z_logvar = model(sample, training=False)
            z = model.reparameterise(z_mean, z_logvar)
            all_z.append(z)

    all_z = torch.cat(all_z, dim=0)

    if latent_dim == 3:
        ax.scatter3D(
            all_z[:, 0], all_z[:, 1], all_z[:, 2],
            alpha=0.5, c=all_z[:, 2], cmap='viridis'
        )
        ax.set_xlabel('z[0] (latent dimension 1)')
        ax.set_ylabel('z[1] (latent dimension 2)')
        ax.set_zlabel('z[2] (latent dimension 3)')
    else: # latent_dim == 2
        ax.scatter(
            all_z[:, 0], all_z[:, 1],
            alpha=0.5, c=all_z[:, 1], cmap='viridis'
        )
        ax.set_xlabel('z[0] (latent dimension 1)')
        ax.set_ylabel('z[1] (latent dimension 2)')

    plt.title('Latent Space Mapping of Tetris States')
    plt.grid(True)
    plt.savefig('./out/latent_space.png')
    plt.show()

def save_model(model, path):
    """Saves the model state dictionary to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Loads the model state dictionary from the specified path."""
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model

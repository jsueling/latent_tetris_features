"""t-SNE visualisation of Tetris VAE latent space with K-Means clustering."""
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from tetris_vae_mlp import TetrisVAE, load_model
from tetris_dataset import TetrisDataset

LATENT_DIM = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_tsne(
        data,
        labels,
        n_components=2,
        seed=0
    ):
    """
    Visualises high-dimensional data using t-SNE with labels from K-Means clustering.
    Parameters:
        data: High-dimensional data to visualise.
        labels: Labels corresponding to the data points.
        perplexity: Perplexity parameter for t-SNE.
        n_components: Number of dimensions for the output space.
        seed: Random seed for reproducibility.
    """

    # Optimal perplexity ranges to test suggested by van der Maaten & Hinton:
    # https://distill.pub/2016/misread-tsne/
    perplexities = [5, 30, 50]

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_colour = {
        label: plt.cm.nipy_spectral(i / n_labels)
        for i, label in enumerate(unique_labels)
    }

    for perplexity in perplexities:

        fig = plt.figure(figsize=(10, 10))
        tsne = TSNE(
            perplexity=perplexity,
            n_components=n_components,
            random_state=seed,
        )
        tsne_output = tsne.fit_transform(data)
        plt.scatter(
            tsne_output[:, 0],
            tsne_output[:, 1],
            c=list(map(label_to_colour.get, labels)),
            alpha=0.5
        )
        plt.title(
            f"t-SNE Visualisation of Tetris State Latent Space with K-Means Clustering Labels" \
            f"\n (Perplexity={perplexity}, Sample size={len(data):,}, " \
            f"k-clusters={n_labels}, Reduction: 8D → 2D)",
            pad=20
        )
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(True)
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {label_index + 1}',
                           markerfacecolor=label_to_colour[label_index], markersize=10)
                for label_index in range(n_labels)
            ],
            loc='upper right',
            bbox_to_anchor=(1.2, 1),
            title='Cluster groups'
        )
        plt.tight_layout()
        plt.subplots_adjust(right=0.825)
        plt.savefig(f'./out/t_sne_perplexity_{perplexity}_clusters_{n_labels}.png')
        plt.show()
        plt.close(fig)

if __name__ == "__main__":
    # Set random seed for reproducibility
    RANDOM_SEED = 0
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Load the data
    vae_model = TetrisVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae_model = load_model(vae_model, "./out/model_8dim_1e-3.pth")
    dataset = TetrisDataset(device=DEVICE)
    # Sample a subset of the dataset for visualisation
    indices = torch.randperm(len(dataset))[:10000]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=128)

    latent_samples = []
    with torch.no_grad():
        for sample in dataloader:
            _, z_mean, _ = vae_model(sample, training=False)
            latent_samples.append(z_mean.cpu().numpy())
        latent_samples = np.concatenate(latent_samples, axis=0)

    # Informed by clustering methods and visual inspection
    K_CLUSTERS = 22

    clusterer = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_SEED)
    cluster_labels = clusterer.fit_predict(latent_samples)

    plot_tsne(latent_samples, cluster_labels, seed=RANDOM_SEED)

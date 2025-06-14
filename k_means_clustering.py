"""K-Means clustering on Tetris VAE latent space"""

import random

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from tetris_vae import TetrisVAE, load_model
from tetris_dataset import TetrisDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 8

def elbow_plot(data, min_k=2, max_k=40):
    """
    Generates an elbow plot for determining the optimal number of clusters in K-Means clustering.
    """

    inertias = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(min_k, max_k + 1), inertias, marker='o')
    plt.title('Elbow plot for K-Means clustering')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Inertia)')
    plt.savefig('./out/elbow_plot.png')
    plt.show()

def avg_silhouette(data, min_k=2, max_k=40):
    """Suggests optimal k value based on average silhouette score for a range of k values."""

    silhouette_avgs = []
    n_clusters = range(min_k, max_k + 1)

    # Calculate avg silhouette scores for each k
    for k in n_clusters:
        clusterer = KMeans(n_clusters=k, random_state=10)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_avgs.append(silhouette_avg)

    _, ax1 = plt.subplots(1, 1)
    ax1.plot(n_clusters, silhouette_avgs, marker='o', color='blue')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Average silhouette score')
    ax1.set_title('Average silhouette score for K-Means clustering')
    ax1.grid()
    plt.savefig('./out/avg_silhouette.png')
    plt.show()

def gap_statistic(data, min_k=2, max_k=40, num_ref_datasets=10):
    """
    Compares within-cluster dispersion to random reference distribution
    for multiple values of k. Large gap values indicate that the clustering structure
    in the data is significantly better than random uniform distribution.
    """

    def compute_inertia(data, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        return kmeans.inertia_

    gaps = []
    standard_errors = []
    for k in range(min_k, max_k + 1):
        # Original data inertia
        original_inertia = compute_inertia(data, k)

        # Reference data inertias
        ref_inertias = []

        for _ in range(num_ref_datasets):
            random_data = np.random.uniform(data.min(), data.max(), data.shape)
            ref_inertias.append(compute_inertia(random_data, k))

        # Gap - For this k, what's the difference between inertia of the initial dataset and
        # the average of reference dataset inertias (uniformly sampled from the same range)
        gap = np.log(np.mean(ref_inertias)) - np.log(original_inertia)
        gaps.append(gap)

        # Standard error of the gap statistic
        std_log_ref_inertias = np.std(np.log(ref_inertias))
        standard_error_gap = std_log_ref_inertias * np.sqrt(1 + 1 / num_ref_datasets)
        standard_errors.append(standard_error_gap)

    # Choose smallest k such that Gap(k) ≥ Gap(k+1) - se(k+1)
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - standard_errors[i + 1]:
            print(f"Optimal number of clusters (k) is {i + min_k} based on Gap statistic.")
            break

    print("Gap statistic values:", [f"{i+min_k}: {round(g, 4)}" for i, g in enumerate(gaps)])
    plt.plot(range(min_k, max_k + 1), gaps, marker='o')
    plt.title('Gap Statistic for K-Means Clustering')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.savefig('./out/gap_statistic.png')
    plt.show()
    return gaps

def visualise_centroids(data, model, k=20, n_samples=8):
    """Visualizes the centroids of the clusters in the latent space."""
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    all_centroid_samples = []
    noise_scale = 3

    for centroid in centroids:

        # Generate noisy samples around the centroid
        centroid_samples = []
        for _ in range(n_samples):
            noise = np.random.normal(0, noise_scale, size=centroid.shape)
            noisy_centroid = centroid + noise
            centroid_samples.append(noisy_centroid)

        # Convert to numpy array
        centroid_samples = np.array(centroid_samples)
        # Decode the noisy centroid samples
        decoded_logits = model.decode(
            torch.tensor(
                centroid_samples,
                device=DEVICE,
                dtype=torch.float32
            )
        )
        decoded_probabilities = torch.sigmoid(decoded_logits).detach().cpu().numpy()
        # Possible to convert into binary grids from probabilities
        # reconstructed_grids = (decoded_probabilities > 0.5).astype(np.float32)
        all_centroid_samples.append(decoded_probabilities)
        # all_centroid_samples.append(reconstructed_grids)

    # Visualise the reconstructed centroid samples
    _, axes = plt.subplots(n_samples, k, figsize=(k * 2, n_samples * 1.5))
    for centroid_index in range(k):
        for sample_index in range(n_samples):
            axes[sample_index, centroid_index].imshow(
                all_centroid_samples[centroid_index][sample_index].reshape(20, 10),
                cmap='Reds',
                vmin=0,
                vmax=1,
                interpolation='nearest'
            )
            axes[sample_index, centroid_index].axis('off')
    plt.savefig('./out/centroids.png')
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    vae_model = TetrisVAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae_model = load_model(vae_model, "./out/best_model.pth")
    dataset = TetrisDataset(device=DEVICE)
    # Sample a subset of the dataset for clustering
    indices = torch.randperm(len(dataset))[:10000]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=128)

    latent_samples = []
    with torch.no_grad():
        for sample in dataloader:
            _, z_mean, _ = vae_model(sample, training=False)
            latent_samples.append(z_mean.cpu().numpy())
        latent_samples = np.concatenate(latent_samples, axis=0)

    # Determining the optimal number of clusters: https://uc-r.github.io/kmeans_clustering#gap
    elbow_plot(latent_samples, min_k=2, max_k=int(len(latent_samples) ** 0.5) + 1)
    avg_silhouette(latent_samples, min_k=2, max_k=int(len(latent_samples) ** 0.5) + 1)
    gap_statistic(latent_samples, min_k=2, max_k=int(len(latent_samples) ** 0.5) + 1)

    # Using the optimal clusters determined from the above methods
    visualise_centroids(latent_samples, vae_model, k=25)

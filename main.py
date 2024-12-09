import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load and preprocess the data
data = pd.read_csv('segmentation data.csv')
data_cleaned = data.dropna().drop_duplicates()
features = data_cleaned.drop(columns=['ID']).values

# Standardize the dataset
mean = features.mean(axis=0)
std = features.std(axis=0)
scaled_features = (features - mean) / std

# PCA Implementation
def pca(data, n_components):
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    top_eigenvectors = eigenvectors[:, :n_components]
    reduced_data = np.dot(data, top_eigenvectors)
    return reduced_data

reduced_features = pca(scaled_features, 2)

# Split dataset into training and testing sets
def train_test_split(data, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]

train_data, test_data = train_test_split(reduced_features, test_size=0.2)

# Custom K-Means implementation
def calculate_distance(point, centroids):
    return np.sum((point - centroids) ** 2, axis=1)

def kmeans_manual(data, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iters):
        distances = np.array([calculate_distance(point, centroids) for point in data])
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == cluster].mean(axis=0) for cluster in range(k)])
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, labels

def calculate_inertia(data, centroids, labels):
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

# Elbow Method
def elbow_method(data, max_k=10, use_custom=True):
    inertias = []
    for k in range(1, max_k + 1):
        if use_custom:
            centroids, labels = kmeans_manual(data, k)
            inertia = calculate_inertia(data, centroids, labels)
        else:                                                                                                                                           
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertia = kmeans.inertia_
        inertias.append(inertia)
    return inertias

# Perform Elbow Method
max_k = 10
custom_inertias = elbow_method(reduced_features, max_k, use_custom=True)
sklearn_inertias = elbow_method(reduced_features, max_k, use_custom=False)

# Plot Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), custom_inertias, marker='o', linestyle='--', label='Custom K-Means')
plt.plot(range(1, max_k + 1), sklearn_inertias, marker='x', linestyle='-', label='Scikit-learn K-Means')
plt.title('Elbow Method Comparison', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=14)
plt.ylabel('Inertia (Sum of Squared Distances)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()

k = 4

def calculate_distance_silhouette(point, other_points):
    return np.linalg.norm(other_points - point, axis=1)

def silhouette_score_manual(X, labels):
    n_samples = len(X)
    silhouette_scores = []

    for i in range(n_samples):
        same_cluster_points = X[labels == labels[i]]
        other_cluster_points = X[labels != labels[i]]

        a_i = np.mean(calculate_distance_silhouette(X[i], same_cluster_points))

        b_i_values = []
        for cluster_label in np.unique(labels):
            if cluster_label != labels[i]:
                other_cluster_points = X[labels == cluster_label]
                b_i = np.mean(calculate_distance_silhouette(X[i], other_cluster_points))
                b_i_values.append(b_i)
        
        b_i_min = min(b_i_values)
        s_i = (b_i_min - a_i) / max(a_i, b_i_min)
        silhouette_scores.append(s_i)

    avg_silhouette_score = np.mean(silhouette_scores)
    return avg_silhouette_score

# Perform Clustering
custom_centroids_train, custom_labels_train = kmeans_manual(train_data, k)
custom_centroids_test, custom_labels_test = kmeans_manual(test_data, k)

sklearn_kmeans_train = KMeans(n_clusters=k, random_state=42)
sklearn_kmeans_train.fit(train_data)
sklearn_labels_train = sklearn_kmeans_train.labels_

sklearn_kmeans_test = KMeans(n_clusters=k, random_state=42)
sklearn_kmeans_test.fit(test_data)
sklearn_labels_test = sklearn_kmeans_test.labels_

# Plot Clusters
def plot_clusters(data, labels, centroids, title, ax):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}', s=30)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.legend(fontsize=10)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Custom K-Means - Training
plot_clusters(train_data, custom_labels_train, custom_centroids_train, "Custom K-Means - Training Data", axs[0, 0])
# Sklearn K-Means - Training
plot_clusters(train_data, sklearn_labels_train, sklearn_kmeans_train.cluster_centers_, "Sklearn K-Means - Training Data", axs[0, 1])
# Custom K-Means - Testing
plot_clusters(test_data, custom_labels_test, custom_centroids_train, "Custom K-Means - Testing Data", axs[1, 0])
# Sklearn K-Means - Testing
plot_clusters(test_data, sklearn_labels_test, sklearn_kmeans_train.cluster_centers_, "Sklearn K-Means - Testing Data", axs[1, 1])

plt.tight_layout()
plt.show()

# Calculate Metrics
custom_inertia_train = calculate_inertia(train_data, custom_centroids_train, custom_labels_train)
custom_inertia_test = calculate_inertia(test_data, custom_centroids_test, custom_labels_test)
custom_silhouette_train = silhouette_score_manual(train_data, custom_labels_train)
custom_silhouette_test = silhouette_score_manual(test_data, custom_labels_test)

sklearn_inertia_train = sklearn_kmeans_train.inertia_
sklearn_inertia_test = sklearn_kmeans_test.inertia_
sklearn_silhouette_train = silhouette_score(train_data, sklearn_labels_train)
sklearn_silhouette_test = silhouette_score(test_data, sklearn_labels_test)

# Metrics Comparison Table
metrics = [
    ("Metric", "Custom K-Means (Train)", "Custom K-Means (Test)", "Sklearn K-Means (Train)", "Sklearn K-Means (Test)"),
    ("Inertia", f"{custom_inertia_train:.2f}", f"{custom_inertia_test:.2f}", f"{sklearn_inertia_train:.2f}", f"{sklearn_inertia_test:.2f}"),
    ("Silhouette Score", f"{custom_silhouette_train:.2f}", f"{custom_silhouette_test:.2f}", f"{sklearn_silhouette_train:.2f}", f"{sklearn_silhouette_test:.2f}"),
]

# Plot Metrics Table
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
table = ax.table(
    cellText=metrics,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.title("Comparison of Metrics", fontsize=16)
plt.show()

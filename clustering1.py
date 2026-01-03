"""
END-TO-END PIPELINE:
Clustering a categorical-label time series using co-occurrence embeddings
"""

import numpy as np
from gensim.models import Word2Vec
import umap
import hdbscan
from collections import Counter

import matplotlib
matplotlib.use("TkAgg")  # reliable GUI backend for VSCode
import matplotlib.pyplot as plt

# prepare data
import pandas as pd

import config

CSV_PATH = f"{config.MY_PATH}/{config.COMBINED_FILE}.csv"

# Read CSV
df = pd.read_csv(CSV_PATH)

# Optional: ensure correct sorting by time
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Extract the 7 categorical labels per timestep
label_columns = ["d1", "d2", "d3", "d4", "d5", "d6", "bonus"]

# Convert to the expected format:
# data[t] = [label1, label2, ..., label7]
data = df[label_columns].astype(int).values.tolist()

# Word2Vec expects strings as tokens
data_str = [[str(label) for label in timestep] for timestep in data]

# Optional sanity check
print("Number of timesteps:", len(data))
print("Example timestep:", data[0])


# ---------------------------------------------------------
# 1. INPUT DATA
# ---------------------------------------------------------
# data[t] = list of 7 categorical IDs at timestep t
# Example:
# data = [
#     [3, 7, 10, 11, 22, 31, 45],
#     [3, 7, 10, 14, 22, 31, 45],
#     ...
# ]

# Convert integer IDs to strings
# Word2Vec expects tokens, not numeric values
data_str = [[str(label) for label in timestep] for timestep in data]


# ---------------------------------------------------------
# 2. LEARN EMBEDDINGS FOR LABEL IDS
# ---------------------------------------------------------
# We treat each timestep as a "sentence"
# and each label ID as a "word"
#
# The model learns vectors such that labels
# appearing together in time have similar embeddings

label_model = Word2Vec(
    sentences=data_str,
    vector_size=32,    # embedding size (small is enough for 50 labels)
    window=5,          # temporal context
    min_count=1,       # all labels are kept
    sg=1,              # skip-gram (better for small datasets)
    workers=4,
    epochs=50
)


# ---------------------------------------------------------
# 3. EMBED EACH TIMESTEP
# ---------------------------------------------------------
# Each timestep has 7 labels → we average their embeddings
# Mean pooling is simple and works well for sets

def embed_timestep(timestep):
    return np.mean(
        [label_model.wv[label] for label in timestep],
        axis=0
    )

# X_timestep[t] is a dense vector describing timestep t
X_timestep = np.array([embed_timestep(ts) for ts in data_str])
# Shape: (T, 32)


# ---------------------------------------------------------
# 4. ADD TEMPORAL CONTEXT (SLIDING WINDOWS)
# ---------------------------------------------------------
# Clustering individual timesteps is noisy.
# Instead, we cluster short windows to capture regimes.

WINDOW_SIZE = 10   # number of timesteps per window
STRIDE = 5         # overlap between windows

def make_windows(X, window_size, stride):
    windows = []
    for i in range(0, len(X) - window_size + 1, stride):
        windows.append(X[i:i+window_size].mean(axis=0))
    return np.array(windows)

X_window = make_windows(X_timestep, WINDOW_SIZE, STRIDE)
# Shape: (num_windows, 32)


# ---------------------------------------------------------
# 5. DIMENSIONALITY REDUCTION (METRIC LEARNING)
# ---------------------------------------------------------
# Word2Vec space is noisy and anisotropic.
# UMAP reshapes it into a space where density-based clustering works.

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=10,   # lower dimension for clustering
    random_state=42
)

X_reduced = reducer.fit_transform(X_window)
# Shape: (num_windows, 10)


# ---------------------------------------------------------
# 6. CLUSTERING
# ---------------------------------------------------------
# HDBSCAN finds dense regions and labels noise automatically.
# This is ideal when the number of regimes is unknown.

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,   # smallest regime size
    min_samples=10,        # noise sensitivity
    metric="euclidean"
)

clusters = clusterer.fit_predict(X_reduced)
# clusters[i] = cluster ID for window i, or -1 if noise

#
# Visualize cluster assignments over time
plt.figure(figsize=(12, 3))
plt.plot(clusters, drawstyle="steps-post")
plt.title("Cluster assignment over time windows")
plt.xlabel("Window index")
plt.ylabel("Cluster ID")
plt.show()


# ---------------------------------------------------------
# 7. VISUALIZATION (OPTIONAL BUT USEFUL)
# ---------------------------------------------------------
# Project to 2D only for visualization

X_vis = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(X_window)

plt.figure(figsize=(8, 6))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=clusters, cmap="tab20", s=20)
plt.colorbar(label="Cluster ID")
plt.title("Temporal Regimes Discovered by Clustering")
plt.show()


# ---------------------------------------------------------
# 8. INTERPRET CLUSTERS
# ---------------------------------------------------------
# For each cluster, look at which labels dominate

# Recreate the label windows so we can inspect them
label_windows = [
    data_str[i:i+WINDOW_SIZE]
    for i in range(0, len(data_str) - WINDOW_SIZE + 1, STRIDE)
]

cluster_label_stats = {}

for cluster_id in set(clusters):
    if cluster_id == -1:
        continue  # skip noise
    indices = np.where(clusters == cluster_id)[0]
    labels = []
    for idx in indices:
        for timestep in label_windows[idx]:
            labels.extend(timestep)
    cluster_label_stats[cluster_id] = Counter(labels)

# Print most common labels per cluster
for cid, counter in cluster_label_stats.items():
    print(f"\nCluster {cid}:")
    for label, count in counter.most_common(10):
        print(f"  Label {label}: {count}")


# ---------------------------------------------------------
# 9. (OPTIONAL) CLUSTER QUALITY CHECK
# ---------------------------------------------------------
from sklearn.metrics import silhouette_score

mask = clusters != -1
if mask.sum() > 1:
    score = silhouette_score(X_reduced[mask], clusters[mask])
    print("\nSilhouette score:", score)

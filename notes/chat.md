ChatGPT_TimeSeries_Clustering.md
# Time Series Clustering with Categorical Labels

**Date:** 2026-01-02  
**Assistant:** GPT-5 mini  

This document is a complete reference for analyzing a time series dataset with 7 categorical labels per timestep (IDs 0–49), using embeddings, windowing, dimensionality reduction, and clustering.

---

## 1. Problem Description

- Dataset: Time series
- Each timestep: 7 categorical labels (integers from 0–49)
- Goal: Discover **clusters/regimes** in the data
- Challenge: Labels are **categorical IDs**, not semantic text
- Proposed approach:
  1. Learn embeddings for label IDs
  2. Aggregate embeddings per timestep
  3. Optionally aggregate over windows
  4. Reduce dimensionality
  5. Cluster windows
  6. Analyze clusters

---

## 2. Evaluating the Approach

- Embedding labels is necessary to **capture co-occurrence**.
- Options:
  - Word2Vec / Skip-gram embeddings for IDs (best fit)
  - Frequency-based embeddings (simpler but less flexible)
- Clustering can be:
  - **Point-level** (per timestep)
  - **Window-level** (recommended)
  - **Sequence-level** (for multiple time series)

---

## 3. Embedding Strategy for Categorical IDs

### 3.1 Word2Vec (Recommended)

```python
from gensim.models import Word2Vec

# data: list of timesteps, each timestep is a list of 7 labels (strings)
model = Word2Vec(
    sentences=data,
    vector_size=32,
    window=5,
    min_count=1,
    sg=1
)

def embed_timestep(labels):
    return sum(model.wv[label] for label in labels) / len(labels)


Pros:

Captures co-occurrence structure

Works well with small vocabularies

Stable and interpretable

3.2 Alternative: Semantic embeddings

Only if labels have meaning (e.g., words)

Not needed for integer IDs

4. Sliding Window Aggregation
def sliding_windows(embeddings, window_size=10, stride=5):
    windows = []
    for i in range(0, len(embeddings) - window_size + 1, stride):
        windows.append(np.mean(window[i:i+window_size], axis=0))
    return np.array(windows)


Purpose:

Reduces noise

Captures temporal regimes

5. Dimensionality Reduction (UMAP)
import umap

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=10,
    random_state=42
)
X_reduced = reducer.fit_transform(X_window)


Why:

Word2Vec embeddings live on an anisotropic manifold

Reduces noise

Makes clusters density-based clustering friendly

6. Clustering (HDBSCAN)
import hdbscan

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    metric='euclidean'
)
clusters = clusterer.fit_predict(X_reduced)


Automatically finds cluster count

Labels -1 are noise points

Density-based → works well on embeddings

7. Cluster Interpretation
from collections import Counter

window_labels = [
    data[i:i+10]  # window_size
    for i in range(0, len(data)-9, 5)
]

cluster_label_stats = {}

for cluster_id in set(clusters):
    if cluster_id == -1: continue
    idxs = np.where(clusters == cluster_id)[0]
    labels = []
    for i in idxs:
        for ts in window_labels[i]:
            labels.extend(ts)
    cluster_label_stats[cluster_id] = Counter(labels)

for cid, counter in cluster_label_stats.items():
    print(f"\nCluster {cid}:")
    for label, count in counter.most_common(10):
        print(f"  Label {label}: {count}")


Reveals dominant labels per cluster

Helps identify distinct regimes

8. Silhouette Score
from sklearn.metrics import silhouette_score

mask = clusters != -1
silhouette = silhouette_score(X_reduced[mask], clusters[mask])
print("Silhouette:", silhouette)


For categorical embeddings, 0.1–0.2 is normal

Low silhouette ≠ meaningless clustering

Temporal coherence matters more than pure Euclidean separation

9. Reading CSV Input
import pandas as pd

CSV_PATH = "your_data.csv"

df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

label_columns = ["d1","d2","d3","d4","d5","d6","bonus"]
data = df[label_columns].astype(int).values.tolist()
data_str = [[str(label) for label in timestep] for timestep in data]

print("Number of timesteps:", len(data))
print("Example timestep:", data[0])


Preserves temporal ordering

Produces the data structure expected by pipeline

10. Cluster-over-Time Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12,3))
plt.plot(clusters, drawstyle="steps-post")
plt.title("Cluster assignment over time windows")
plt.xlabel("Window index")
plt.ylabel("Cluster ID")
plt.yticks(sorted(set(clusters)))
plt.grid(alpha=0.3)
plt.show()


Visual sanity check

Look for plateaus → stable regimes

Noise (-1) often occurs at transitions

Optional: Plot against dates
window_dates = df["date"].iloc[WINDOW_SIZE-1::STRIDE]
plt.figure(figsize=(12,3))
plt.step(window_dates, clusters, where="post")
plt.xlabel("Date")
plt.ylabel("Cluster ID")
plt.title("Cluster regimes over time")
plt.show()

11. Debugging Plots in VS Code

Ensure plt.show() is called

For VS Code terminal, force backend:

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


Alternatively, run script in Interactive Mode / Jupyter Notebook

12. Optional Improvements

Increase window size → smoother regimes

Compute relative frequencies per cluster

Transition matrix → temporal dynamics

import itertools
lengths = []
for _, group in itertools.groupby(clusters):
    lengths.append(len(list(group)))
print("Average regime length:", sum(lengths)/len(lengths))

unique = sorted(set(c for c in clusters if c != -1))
idx = {c:i for i,c in enumerate(unique)}
M = np.zeros((len(unique), len(unique)))
for a,b in zip(clusters[:-1], clusters[1:]):
    if a != -1 and b != -1:
        M[idx[a], idx[b]] += 1
M = M / M.sum(axis=1, keepdims=True)
print(M)

13. Notes on Model Choice

Word2Vec is optimal for categorical co-occurrence

Vocabulary is small → deep models are overkill

Node2Vec / graph-based embeddings are optional for advanced patterns

14. Package Requirements

Preinstalled in Anaconda:

numpy, pandas, matplotlib, scikit-learn

Needs installation:

conda install -c conda-forge gensim umap-learn hdbscan


Alternative fallback with PCA/KMeans possible if you avoid extra installs

15. Summary

Pipeline: CSV → Word2Vec → timestep embedding → window → UMAP → HDBSCAN → cluster interpretation → temporal visualization

Clusters correspond to temporal regimes

Silhouette score is secondary; temporal coherence is primary

Pipeline is fully reproducible and adaptable to multiple time series
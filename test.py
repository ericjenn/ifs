# Load the trained embedder and apply it to new traces
from ifs import TraceEmbedder, generate_trace_pair
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import random

all_traces = []

for _ in range(20):
    
    # Call generate_trace_pair to get two traces and a label
    trace1, trace2 = generate_trace_pair()
    
    # Append both traces to the all_traces list
    all_traces.append(trace1)
    all_traces.append(trace2)

print(f"Total number of generated traces: {len(all_traces)}")
print("First trace sample:")
print(all_traces[0])

print("\nComputing embeddings for all generated traces...")

# Load the embedder
print("Loading pre-trained TraceEmbedder from 'trace_embedder.pt'...")
# 1. Determine the device to use (CPU or CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load the pre-trained TraceEmbedder model from the file
# The TraceEmbedder class was defined in the first executed cell and should be available.
embedder = TraceEmbedder.load_pretrained("trace_embedder.pt", device=device)

# 3. Move the embedder model to the determined device (already handled by load_pretrained, but explicit for clarity)
embedder = embedder.to(device)

# 4. Set the embedder to evaluation mode
embedder.eval()

print(f"✓ TraceEmbedder loaded successfully on device: {device}")

# Compute embeddings for all traces in a batch
with torch.no_grad():
    all_embeddings = embedder.embed_batch(all_traces, device=device)

print(f"Total embeddings shape: {all_embeddings.shape}")
print("\n✓ All traces embedded successfully.")

import itertools
import torch.nn.functional as F

print("\nCalculating pairwise cosine distances...")

pairwise_distances = []
num_traces = len(all_embeddings)

for i, j in itertools.combinations(range(num_traces), 2):
    embedding1 = all_embeddings[i]
    embedding2 = all_embeddings[j]

    # Cosine similarity is already normalized, so we can use it directly
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
    
    # Cosine distance = 1 - cosine_similarity
    distance = 1 - similarity

    pairwise_distances.append({
        'trace_index_1': i,
        'trace_index_2': j,
        'cosine_similarity': similarity,
        'cosine_distance': distance
    })

print(f"✓ Calculated {len(pairwise_distances)} pairwise distances.")

import numpy as np

print("\n--- Analysis of Pairwise Cosine Distances ---")

distances = [d['cosine_distance'] for d in pairwise_distances]
similarities = [d['cosine_similarity'] for d in pairwise_distances]

# Descriptive statistics for distances
min_distance = np.min(distances)
max_distance = np.max(distances)
avg_distance = np.mean(distances)
std_distance = np.std(distances)

# Descriptive statistics for similarities
min_similarity = np.min(similarities)
max_similarity = np.max(similarities)
avg_similarity = np.mean(similarities)
std_similarity = np.std(similarities)

print(f"Total unique pairs: {len(pairwise_distances)}")
print("\nCosine Distance Statistics:")
print(f"  Min Distance: {min_distance:.4f}")
print(f"  Max Distance: {max_distance:.4f}")
print(f"  Avg Distance: {avg_distance:.4f}")
print(f"  Std Dev Distance: {std_distance:.4f}")

print("\nCosine Similarity Statistics:")
print(f"  Min Similarity: {min_similarity:.4f}")
print(f"  Max Similarity: {max_similarity:.4f}")
print(f"  Avg Similarity: {avg_similarity:.4f}")
print(f"  Std Dev Similarity: {std_similarity:.4f}")

# Find most similar pair (lowest distance, highest similarity)
most_similar_pair = min(pairwise_distances, key=lambda x: x['cosine_distance'])
least_similar_pair = max(pairwise_distances, key=lambda x: x['cosine_distance'])

print("\nMost Similar Pair:")
print(f"  Traces ({most_similar_pair['trace_index_1']}, {most_similar_pair['trace_index_2']})")
print(f"  Similarity: {most_similar_pair['cosine_similarity']:.4f}")
print(f"  Distance: {most_similar_pair['cosine_distance']:.4f}")

print("\nLeast Similar Pair:")
print(f"  Traces ({least_similar_pair['trace_index_1']}, {least_similar_pair['trace_index_2']})")
print(f"  Similarity: {least_similar_pair['cosine_similarity']:.4f}")
print(f"  Distance: {least_similar_pair['cosine_distance']:.4f}")

# Interpret interference levels based on similarity thresholds
high_similarity_threshold = 0.99
medium_similarity_threshold = 0.98

high_sim_count = sum(1 for d in pairwise_distances if d['cosine_similarity'] >= high_similarity_threshold)
medium_sim_count = sum(1 for d in pairwise_distances if d['cosine_similarity'] >= medium_similarity_threshold and d['cosine_similarity'] < high_similarity_threshold)
low_sim_count = sum(1 for d in pairwise_distances if d['cosine_similarity'] < medium_similarity_threshold)

print("\nInterference Level Distribution (Custom Thresholds):")
print(f"  High Similarity (>= {high_similarity_threshold:.2f}): {high_sim_count} pairs")
print(f"  Medium Similarity (>= {medium_similarity_threshold:.2f} and < {high_similarity_threshold:.2f}): {medium_sim_count} pairs")
print(f"  Low Similarity (< {medium_similarity_threshold:.2f}): {low_sim_count} pairs")

print("\n✓ Analysis complete.")

# Visualize embedding distributions and global structure

def visualize_embedding_distribution(embeddings, title_prefix='Trace Embeddings'):
    # embeddings: torch.Tensor shape (N, D)
    embeddings_np = embeddings.detach().cpu().numpy() if hasattr(embeddings, 'detach') else np.array(embeddings)

    # Flattened value distribution
    plt.figure(figsize=(10, 4))
    plt.hist(embeddings_np.flatten(), bins=120, color='mediumseagreen', alpha=0.75)
    plt.title(f'{title_prefix} - Component Value Distribution (flattened)')
    plt.xlabel('Embedding component value')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Per-dimension mean and std
    per_dim_mean = embeddings_np.mean(axis=0)
    per_dim_std = embeddings_np.std(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(per_dim_mean, color='navy', linewidth=1.0, label='Mean')
    plt.fill_between(range(len(per_dim_std)), per_dim_mean - per_dim_std, per_dim_mean + per_dim_std, color='lightblue', alpha=0.4, label='Mean ± Std')
    plt.title(f'{title_prefix} - Per-Dimension Mean ± Std')
    plt.xlabel('Embedding dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # PCA 2D projection
    pca = PCA(n_components=2)
    projected = pca.fit_transform(embeddings_np)
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(projected[:, 0], projected[:, 1], c=np.arange(projected.shape[0]), cmap='viridis', s=40, edgecolor='k', alpha=0.75)
    plt.colorbar(sc, label='Trace index')
    plt.title(f'{title_prefix} - PCA 2D projection (PC1 {explained[0]*100:.1f}%, PC2 {explained[1]*100:.1f}%)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

visualize_embedding_distribution(all_embeddings, title_prefix='ARM Trace Embeddings')


import seaborn as sns
import matplotlib.pyplot as plt

print("\n--- Visualizing Pairwise Cosine Similarities ---")

# Create a square matrix for similarities
num_traces = len(all_embeddings)
similarity_matrix = np.eye(num_traces) # Initialize with 1s on diagonal

# Fill the upper triangle of the similarity matrix
for pair in pairwise_distances:
    i = pair['trace_index_1']
    j = pair['trace_index_2']
    similarity = pair['cosine_similarity']
    similarity_matrix[i, j] = similarity
    similarity_matrix[j, i] = similarity # Symmetric matrix

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    similarity_matrix,
    annot=True,
    cmap='viridis',
    fmt=".3f",
    linewidths=.5,
    linecolor='lightgray',
    cbar_kws={'label': 'Cosine Similarity'}
)

plt.title('Pairwise Cosine Similarity between ARM Traces')
plt.xlabel('Trace Index')
plt.ylabel('Trace Index')
plt.tight_layout()
plt.show()

print("\n✓ Heatmap generated successfully.")
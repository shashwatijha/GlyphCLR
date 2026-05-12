from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Reuse the X and Y from above or reload
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
all_embeddings = np.concatenate([X, Y])
reduced = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(reduced[:len(X), 0], reduced[:len(X), 1], label='Linear A', alpha=0.5, c='blue')
plt.scatter(reduced[len(X):, 0], reduced[len(X):, 1], label='Linear B', alpha=0.5, c='orange')
plt.legend()
plt.title("Joint Embedding Space: Linear A & B Manifolds")
plt.savefig('results/tsne_manifold_map.png')
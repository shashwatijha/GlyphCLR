import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import ast

# 1. Load the Embeddings
print("Loading embeddings...")
df = pd.read_csv('data/text/glyph_embeddings.csv')
df['Embedding'] = df['Embedding'].apply(ast.literal_eval)

# 2. Preparation for t-SNE
X = np.array(df['Embedding'].tolist())
scripts = df['Script'].values

# 3. Dimensionality Reduction
print("Reducing 384 dimensions to 2D")
# We removed n_iter to be compatible with your specific library version
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)
# 4. Plotting
plt.figure(figsize=(14, 10))
colors = {'LinearA': '#e74c3c', 'LinearB': '#3498db', 'Cuneiform': '#2ecc71', 'Egyptian': '#f1c40f'}

for script in np.unique(scripts):
    idx = np.where(scripts == script)
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=colors.get(script, 'gray'), 
                label=script, alpha=0.5, s=15)

plt.legend(loc='best', fontsize=12)
plt.title("Visual Manifold Alignment of Ancient Scripts", fontsize=16)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('data/visuals/tsne_map.png', dpi=300)
print("Success! Your 'Constellation Map' is saved at visuals/tsne_map.png")
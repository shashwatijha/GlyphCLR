import pandas as pd
import numpy as np
import ot
import ast
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt

import os
os.makedirs('results', exist_ok=True)

print("Loading embeddings...")
df = pd.read_csv('data/text/glyph_embeddings.csv')
df['Embedding'] = df['Embedding'].apply(lambda x: np.array(ast.literal_eval(x)))


la_df = df[df['Script'] == 'linear_a'].reset_index(drop=True)
lb_df = df[df['Script'] == 'linear_b'].reset_index(drop=True)

X = np.stack(la_df['Embedding'].values)
Y = np.stack(lb_df['Embedding'].values)

# 3. Compute Intra-script Distance Matrices (The "Structure")
# This represents how each script relates to ITSELF
print("Computing manifold structures...")
C1 = ot.dist(X, X, metric='cosine')
C2 = ot.dist(Y, Y, metric='cosine')

# Normalize the matrices
C1 /= C1.max()
C2 /= C2.max()

# 4. Run Gromov-Wasserstein Alignment
# This aligns the manifolds based on their internal relationships
print("Aligning manifolds (this may take a minute)...")
p = ot.unif(len(X))
q = ot.unif(len(Y))
gw_coupling = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=1e-2)

# 5. Visualizing the Result (The "Heatmap" for your slides)
plt.figure(figsize=(10, 8))
sns.heatmap(gw_coupling[:25, :25], cmap='viridis')
plt.title("Gromov-Wasserstein Coupling: Linear A vs Linear B")
plt.xlabel("Linear B Indices")
plt.ylabel("Linear A Indices")
plt.savefig('results/gw_alignment_heatmap.png')


print("\n TOP ALIGNMENT RESULTS ")
for i in range(10): # Show top 10 matches
    j = np.argmax(gw_coupling[i, :])
    la_sign = la_df.iloc[i]['SignID']
    lb_sign = lb_df.iloc[j]['SignID']
    score = gw_coupling[i, j]
    print(f"Linear A: {la_sign}  : Linear B: {lb_sign} (Prob: {score:.4f})")

print("\nSuccess! Heatmap saved to results/gw_alignment_heatmap.png")
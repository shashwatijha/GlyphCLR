import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# 1. Load the "Cloud" of data
print("Loading embeddings...")
df = pd.read_csv('data/text/glyph_embeddings.csv')
# Convert string representation of lists back to actual numpy arrays
df['Embedding'] = df['Embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

# 2. Separate the Script Manifolds
la_manifold = df[df['Script'] == 'linear_a']
lb_manifold = df[df['Script'] == 'linear_b']

if la_manifold.empty or lb_manifold.empty:
    print("Error: Ensure you have both Linear A and Linear B images in data/processed!")
    exit()

# 3. Calculate Centroids (The "Shape" of each script)
lb_matrix = np.stack(lb_manifold['Embedding'].values)
lb_labels = lb_manifold['SignID'].values

def get_prediction(la_sign_id):
    # Get the vector for our mystery Linear A sign
    query_vec = la_manifold[la_manifold['SignID'] == la_sign_id]['Embedding'].values[0].reshape(1, -1)
    
    # Manifold Alignment via Cosine Similarity
    # This finds which Linear B "points" are closest to our Linear A "point"
    similarities = cosine_similarity(query_vec, lb_matrix)[0]
    
    # Get Top 3 matches
    top_indices = np.argsort(similarities)[-3:][::-1]
    
    print(f"\nResults for Linear A Sign: {la_sign_id}")
    for i in top_indices:
        print(f"Match: {lb_labels[i]} | Confidence Score: {similarities[i]:.4f}")


try:
    get_prediction('linear_a_0023') 
except:
    print("\nExtraction still running or file name mismatch. Run 'ls data/processed/linear_a' to check names.")

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# 1. Load Data
emb_df = pd.read_csv('data/text/glyph_embeddings.csv')
la_sig = pd.read_csv('data/text/linear_a_signals.csv')
lb_sig = pd.read_csv('data/text/linear_b_signals.csv')

emb_df['Embedding'] = emb_df['Embedding'].apply(ast.literal_eval)

# 2. Extract Feature Matrices
def get_matrix(df, script_name):
    subset = df[df['Script'] == script_name]
    grouped = subset.groupby('SignID')['Embedding'].apply(lambda x: np.mean(x.tolist(), axis=0))
    return np.stack(grouped.values), grouped.index.tolist()

la_vis_mtx, la_vis_ids = get_matrix(emb_df, 'LinearA')
lb_vis_mtx, lb_vis_ids = get_matrix(emb_df, 'LinearB')

# 3. Calculate Visual Similarity (This is the anchor!)
vis_sim_matrix = cosine_similarity(la_vis_mtx, lb_vis_mtx)

# 4. Generate Results without relying on ID strings
results = []
for i, la_id in enumerate(la_vis_ids):
    # Get top 5 visual matches for this LA sign
    top_lb_indices = np.argsort(vis_sim_matrix[i])[-5:][::-1]
    
    for idx in top_lb_indices:
        v_score = vis_sim_matrix[i, idx]
        lb_id = lb_vis_ids[idx]
        
        # We store the visual matches. 
        # Even if we can't match the linguistic ID yet, 
        # this will show us what the AI "sees."
        results.append({
            'LinearA_Image': la_id,
            'LinearB_VisualMatch': lb_id,
            'Visual_Similarity': round(float(v_score), 4)
        })

# 5. Output
if results:
    report = pd.DataFrame(results)
    report.to_csv('data/text/final_decipherment_report.csv', index=False)
    print("\n--- TOP VISUAL MATCHES (ZERO-SHOT) ---")
    print(report.head(20))
else:
    print("Error: No visual matches found. Check your embedding dimensions.")
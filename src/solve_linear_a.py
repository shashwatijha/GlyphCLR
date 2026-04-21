import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast

# 1. Load the data
print("Loading datasets...")
embeddings_df = pd.read_csv('data/text/glyph_embeddings.csv')
lb_signals = pd.read_csv('data/text/linear_b_signals.csv')
la_signals = pd.read_csv('data/text/linear_a_signals.csv')

# Helper: Convert string-list to numpy array
embeddings_df['Embedding'] = embeddings_df['Embedding'].apply(ast.literal_eval)

# 2. Separate Linear A and Linear B visual data
la_vis = embeddings_df[embeddings_df['Script'] == 'LinearA']
lb_vis = embeddings_df[embeddings_df['Script'] == 'LinearB']

def get_combined_score(la_sign_id):
    # Get LA Linguistic Signal
    la_sig = la_signals[la_signals['SignID'] == la_sign_id]
    if la_sig.empty: return []
    
    # Get LA Visual Embedding (average if multiple images exist)
    la_emb_list = la_vis[la_vis['SignID'] == la_sign_id]['Embedding'].tolist()
    if not la_emb_list: return []
    la_emb = np.mean(la_emb_list, axis=0).reshape(1, -1)

    results = []
    
    for _, lb_row in lb_signals.iterrows():
        lb_sign_id = lb_row['SignID']
        
        # A. Linguistic Distance (Lower is better)
        ling_dist = np.sqrt(
            (la_sig['Freq'].values[0] - lb_row['Freq'])**2 +
            (la_sig['InitBias'].values[0] - lb_row['InitBias'])**2 +
            (la_sig['FinalBias'].values[0] - lb_row['FinalBias'])**2
        )
        
        # B. Visual Distance (Cosine Similarity, Higher is better)
        lb_emb_list = lb_vis[lb_vis['SignID'] == lb_sign_id]['Embedding'].tolist()
        if not lb_emb_list: continue
        lb_emb = np.mean(lb_emb_list, axis=0).reshape(1, -1)
        
        vis_sim = cosine_similarity(la_emb, lb_emb)[0][0]
        
        # C. Combined Score (Normalize ling_dist to 0-1 and flip it)
        # Higher score = Stronger Match
        combined_score = (vis_sim * 0.5) + ((1 - ling_dist) * 0.5)
        
        results.append((lb_sign_id, combined_score, vis_sim, ling_dist))

    # Sort by combined score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]

# 3. Run for all Linear A signs
print("Calculating matches...")
final_results = []
for la_sign in la_signals['SignID'].unique():
    top_matches = get_combined_score(la_sign)
    if top_matches:
        final_results.append({
            'LinearA_Sign': la_sign,
            'TopMatch_LB': top_matches[0][0],
            'Confidence': round(top_matches[0][1], 4),
            'Visual_Sim': round(top_matches[0][2], 4),
            'Ling_Dist': round(top_matches[0][3], 4)
        })

df_final = pd.DataFrame(final_results)
df_final.to_csv('data/text/decipherment_candidates.csv', index=False)
print("\nDone! Candidates saved to data/text/decipherment_candidates.csv")
print(df_final.sort_values(by='Confidence', ascending=False).head(10))
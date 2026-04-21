import pandas as pd

# Load the files
emb = pd.read_csv('data/text/glyph_embeddings.csv')
la_sig = pd.read_csv('data/text/linear_a_signals.csv')
lb_sig = pd.read_csv('data/text/linear_b_signals.csv')

print("--- GLYPH IDS (Visual) ---")
print(emb['SignID'].unique()[:5])

print("\n--- LINEAR A IDS (Text) ---")
print(la_sig['SignID'].unique()[:5])

print("\n--- LINEAR B IDS (Text) ---")
print(lb_sig['SignID'].unique()[:5])
import pandas as pd
from collections import Counter
import numpy as np

# Load the Linear B tablets (using the semicolon separator)
df = pd.read_csv(
    'data/text/tablets.csv', 
    sep=';', 
    on_bad_lines='warn', 
    engine='python'
)

print("I found these columns:", df.columns.tolist()) # Add this line

def extract_signals(df, text_column='text'):
    # Storage for counts
    all_signs = []
    initial_counts = Counter()
    final_counts = Counter()
    total_sign_counts = Counter()
    
    for text in df[text_column].dropna():
        # Split by spaces into words, then by hyphens into signs
        words = str(text).split(' ')
        for word in words:
            signs = word.split('-')
            if not signs or signs == ['']: continue
            
            # Update position signals
            initial_counts[signs[0]] += 1
            final_counts[signs[-1]] += 1
            
            # Update total counts
            for s in signs:
                total_sign_counts[s] += 1
                all_signs.append(s)

    # Convert to a DataFrame of Features
    unique_signs = list(total_sign_counts.keys())
    total_sum = sum(total_sign_counts.values())
    
    features = []
    for s in unique_signs:
        freq = total_sign_counts[s] / total_sum
        init_bias = initial_counts[s] / total_sign_counts[s]
        final_bias = final_counts[s] / total_sign_counts[s]
        features.append([s, freq, init_bias, final_bias])
        
    return pd.DataFrame(features, columns=['SignID', 'Freq', 'InitBias', 'FinalBias'])


lb_features = extract_signals(df, text_column='inscription')


lb_features.to_csv('data/text/linear_b_signals.csv', index=False)

print("Success! Signals saved to data/text/linear_b_signals.csv")

print(lb_features.head())
import pandas as pd
from collections import Counter
import re

# Load the file you just uploaded
df = pd.read_csv('data/text/linearA.csv')

def extract_la_signals(df, text_column='transliteratedWords'):
    total_sign_counts = Counter()
    initial_counts = Counter()
    final_counts = Counter()
    
    # We want to ignore numbers and punctuation like 197, 1/2, or 𐄁
    def is_valid_sign(s):
        return s and not s.isdigit() and s not in ['|', ' ', '𐄁', '—', '≈', '[?]']

    for text in df[text_column].dropna():
        # 1. Split by the '|' or spaces to get words
        words = re.split(r'[| ]+', str(text))
        
        for word in words:
            # 2. Split by '-' to get individual signs
            signs = [s for s in word.split('-') if is_valid_sign(s)]
            
            if len(signs) > 0:
                initial_counts[signs[0]] += 1
                final_counts[signs[-1]] += 1
                for s in signs:
                    total_sign_counts[s] += 1

    # Convert to Signal Table
    unique_signs = list(total_sign_counts.keys())
    total_sum = sum(total_sign_counts.values())
    
    features = [[s, 
                 total_sign_counts[s] / total_sum, 
                 initial_counts[s] / total_sign_counts[s], 
                 final_counts[s] / total_sign_counts[s]] 
                for s in unique_signs]
        
    return pd.DataFrame(features, columns=['SignID', 'Freq', 'InitBias', 'FinalBias'])

la_features = extract_la_signals(df)
la_features.to_csv('data/text/linear_a_signals.csv', index=False)

print("Success! Linear A signals generated.")
print(la_features.sort_values(by='Freq', ascending=False).head(10))
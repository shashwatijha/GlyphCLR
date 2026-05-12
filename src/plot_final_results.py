import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Setup
os.makedirs('results', exist_ok=True)
data_path = 'data/text/percentage_results.csv'

# 2. Load and Clean
df = pd.read_csv(data_path)
df['Percentage'] = pd.to_numeric(df['Percentage'])
df = df.sort_values(by='Percentage', ascending=False)

# 3. Create the Visualization
plt.figure(figsize=(11, 7))
sns.set_style("white") # Clean academic look
palette = sns.color_palette("flare", len(df)) # Professional gradient

ax = sns.barplot(x='Percentage', y='Dictionary', data=df, palette=palette)

# Add value labels for precision
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}%', 
                (p.get_width() + 0.3, p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontweight='bold', color='#333333')

# 4. Styling
plt.title('Linear A Manifold Alignment: Language Family affinity', fontsize=16, pad=20, fontweight='bold')
plt.xlabel('Correspondence Confidence (%)', fontsize=12)
plt.ylabel('Candidate Language Family', fontsize=12)
sns.despine() # Removes the box around the plot for a modern look
plt.tight_layout()

# 5. Save
plt.savefig('results/final_language_scores.png', dpi=300)
print("\n--- DONE ---")
print("Your final result chart is here: results/final_language_scores.png")
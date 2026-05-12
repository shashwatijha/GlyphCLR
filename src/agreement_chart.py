import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# Ensure 'decipherment_validation_summary.csv' is in the same folder as this script
try:
    df = pd.read_csv('../results/decipherment_validation_summary.csv')
except FileNotFoundError:
    print("Error: CSV file not found. Check the filename!")
    exit()

# 2. Set the aesthetic style
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 3. Calculate counts for the Agreement Score
# This groups by the score and counts how many signs fall into each
score_counts = df['Agreement_Score'].value_counts().sort_index()

# 4. Create the Bar Plot
# 'viridis' creates that nice blue-to-green gradient
sns.barplot(
    x=score_counts.index, 
    y=score_counts.values, 
    hue=score_counts.index, 
    palette="viridis", 
    legend=False
)

# 5. Add Labels and Title
plt.title('Linguistic Consensus Distribution (Linear A)', fontsize=16, fontweight='bold')
plt.xlabel('Agreement Score (Number of Languages in Consensus)', fontsize=12)
plt.ylabel('Count of Signs', fontsize=12)

# 6. Add value labels on top of bars (optional but looks pro)
for i, v in enumerate(score_counts.values):
    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 7. Show and Save the plot
plt.tight_layout()
plt.savefig('../results/plots/consensus_plot.png', dpi=300)
plt.show()

print("Plot saved as consensus_plot.png")
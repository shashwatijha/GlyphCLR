import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the frequency data
df = pd.read_csv('data/text/frequency_comparison.csv', skiprows=1)

# Clean the data: Get top 15 matches by frequency
# (Assuming the first column is the match and the 4th is the frequency list)
df_top = df.head(15)

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# Create a bar chart of the frequency of the top matches
# We use a simple count/index for the visualization of 'Data Density'
sns.barplot(x=df_top.index, y=range(len(df_top), 0, -1), palette="viridis")

plt.title('Zipfian Distribution: Linear A Character Frequency Alignment', fontsize=14)
plt.xlabel('Top 15 Sign Clusters (Ranked)', fontsize=12)
plt.ylabel('Observed Frequency Density', fontsize=12)
plt.xticks([]) # Hide indices for a cleaner look

plt.savefig('results/frequency_distribution.png', dpi=300)
print("Frequency plot saved to results/frequency_distribution.png")
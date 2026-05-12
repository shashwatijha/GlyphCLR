import pandas as pd

df = pd.read_csv('../results/decipherment_validation_summary.csv')

# Group by Archaeological Site
site_stats = df.groupby('Site').agg({
    'Sign_Code': 'count',
    'Agreement_Score': 'mean'
}).reset_index()

site_stats.columns = ['Archaeological_Site', 'Total_Matches', 'Avg_Confidence']
site_stats = site_stats.sort_values(by='Total_Matches', ascending=False)

print("\n--- DATA FOR 'GEOGRAPHIC DISTRIBUTION' CHART ---")
print(site_stats.to_string(index=False))

# Save for charting
site_stats.to_csv('../results/site_distribution_stats.csv', index=False)
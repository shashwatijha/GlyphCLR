import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PATH SETUP ---
# Adjust these paths if your folder structure is different
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'results', 'decipherment_validation_summary.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'results', 'plots')

# Create the plots folder if it doesn't exist
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def generate_plots():
    try:
        print(f"Loading data from: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        sns.set_theme(style="white")

        # 1. CONFIDENCE LEVELS (Linguistic Consensus)
        print("Generating Confidence Levels plot...")
        plt.figure(figsize=(10, 6))
        score_counts = df['Agreement_Score'].value_counts().sort_index()
        colors = sns.color_palette("Blues", n_colors=len(score_counts))
        bars = plt.bar(score_counts.index.astype(str), score_counts.values, color=colors, edgecolor='black')
        plt.title('Validation Confidence: Linguistic Consensus Counts', fontsize=15, pad=20, fontweight='bold')
        plt.xlabel('Number of Languages in Agreement (Consensus Score)', fontsize=12)
        plt.ylabel('Number of Unique Sign Assignments', fontsize=12)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, yval, ha='center', va='bottom', fontweight='bold')
        plt.savefig(os.path.join(PLOT_DIR, 'confidence_levels.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. GEOGRAPHIC REACH (Site Distribution)
        print("Generating Geographic Reach plot...")
        plt.figure(figsize=(12, 7))
        site_counts = df['Site'].value_counts().head(12)
        sns.barplot(x=site_counts.values, y=site_counts.index, palette="flare", hue=site_counts.index, legend=False)
        plt.title('Geographic Reach: Top 12 Archaeological Find-Sites', fontsize=15, pad=20, fontweight='bold')
        plt.xlabel('Total Validated Sign Matches', fontsize=12)
        plt.ylabel('Archaeological Site', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'geographic_reach.png'), dpi=300)
        plt.close()

        # 3. EVIDENCE HEATMAP (Site vs Confidence)
        print("Generating Evidence Heatmap...")
        pivot_df = df[df['Agreement_Score'] >= 1].groupby(['Site', 'Agreement_Score']).size().unstack(fill_value=0)
        top_sites = df['Site'].value_counts().head(10).index
        pivot_df = pivot_df.reindex(top_sites).fillna(0).astype(int)
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt='d', cmap="YlGnBu", linewidths=.5, cbar_kws={'label': 'Match Count'})
        plt.title('Heatmap of Confidence Results per Site', fontsize=15, pad=20, fontweight='bold')
        plt.xlabel('Agreement Score', fontsize=12)
        plt.ylabel('Site Name', fontsize=12)
        plt.savefig(os.path.join(PLOT_DIR, 'evidence_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. TOP ANCHORS (Primary Phonetic Matches)
        print("Generating Top Anchors visual...")
        top_15 = df.sort_values(['Agreement_Score', 'Site'], ascending=[False, True]).head(15)
        top_15['Label'] = top_15['Sign_Code'] + " (" + top_15['Site'] + ")"
        plt.figure(figsize=(12, 8))
        colors_top = sns.color_palette("viridis", n_colors=15)
        plt.barh(top_15['Label'], top_15['Agreement_Score'], color=colors_top)
        plt.gca().invert_yaxis()
        plt.title('Top 15 Primary Phonetic Anchors', fontsize=15, pad=20, fontweight='bold')
        plt.xlabel('Linguistic Consensus Score', fontsize=12)
        plt.ylabel('Sign & Site Context', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(PLOT_DIR, 'top_anchors_visual.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n All plots saved to: {PLOT_DIR}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    generate_plots()
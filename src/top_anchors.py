import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'results', 'decipherment_validation_summary.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

print(f"Looking for file at: {CSV_PATH}")

try:
    df = pd.read_csv(CSV_PATH)
    
    high_conf = df[df['Agreement_Score'] >= 3].copy()

    top_results = high_conf.sort_values(by=['Agreement_Score', 'Site'], ascending=[False, True])

    presentation_table = top_results[['Sign_Code', 'Proposed_Sound', 'Agreement_Score', 'Site', 'Transcription']]

    print("\n--- DATA FOR 'TOP RESULTS' SLIDE ---")
    print(presentation_table.head(15).to_string(index=False))

    output_path = os.path.join(OUTPUT_DIR, 'top_presentation_anchors.csv')
    presentation_table.to_csv(output_path, index=False)
    print(f"\nSaved refined slide data to: {output_path}")

except FileNotFoundError:
    print(f"ERROR: Could not find the file. Please ensure it exists at {CSV_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")
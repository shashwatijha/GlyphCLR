import pandas as pd
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, 'results', 'decipherment_validation_summary.csv')
GORILA_PATH = os.path.join(BASE_DIR, 'data', 'text/misc', 'Complied_Gorilla.csv')
SEMANTIC_PATH = os.path.join(BASE_DIR, 'src', 'sign_semantics.csv') # Path to the new CSV

def search_sign():
    # Verify files
    paths = [RESULTS_PATH, GORILA_PATH, SEMANTIC_PATH]
    if not all(os.path.exists(p) for p in paths):
        print("Error: Required CSV files missing. Check your paths!")
        return

    # Load data
    df_results = pd.read_csv(RESULTS_PATH)
    df_gorila = pd.read_csv(GORILA_PATH)
    df_semantics = pd.read_csv(SEMANTIC_PATH)

    print("--- Linear A Sign & Tablet Locator ---")
    query = input("Enter Sign Code (e.g., AB171, A301): ").strip().upper()
    
    # 1. Search in results
    pattern = rf"\[{query}\]"
    results = df_results[df_results['Transcription'].str.contains(pattern, case=False, na=False)]
    if results.empty:
        results = df_results[df_results['Transcription'].str.contains(query, case=False, na=False)]

    if not results.empty:
        # 2. Merge with Archaeological Metadata
        gorila_subset = df_gorila[['NEW FORMAT', 'Site', 'GORILA Code']].copy()
        gorila_subset.columns = ['Transcription', 'Site', 'Tablet_ID']
        merged = pd.merge(results, gorila_subset, on=['Transcription'], how='left')
        
        # 3. Merge with Semantic Mapping (The "Meaning IRL")
        # We match on our query directly
        semantic_info = df_semantics[df_semantics['Sign_Code'] == query]
        
        # Formatting output
        final = merged[['Tablet_ID', 'Site_x', 'Transcription', 'Proposed_Sound', 'Agreement_Score']].drop_duplicates()
        final.columns = ['Tablet_ID', 'Site', 'Transcription', 'Sound', 'Score']
        final = final.sort_values(by='Score', ascending=False)

        print(f"\n--- Analysis for '{query}' ---")
        
        # Display Semantic Meaning if available
        if not semantic_info.empty:
            info = semantic_info.iloc[0]
            print(f"MEANING IRL: {info['Meaning']} ({info['Category']})")
            print(f"DOMAIN:      {info['Context']}")
        else:
            print("MEANING IRL: Standard Phonetic Unit (Syllable)")
        
        print("-" * 60)
        
        # Show top 3 instances
        top_options = final.head(3)
        for _, row in top_options.iterrows():
            print(f"Sound Prediction: {row['Sound']} (Confidence: {row['Score']})")
            print(f"Tablet:           {row['Tablet_ID']} at {row['Site']}")
            print(f"Text Segment:     {row['Transcription']}")
            print("-" * 30)
            
    else:
        print(f"\nNo validated matches found for '{query}'.")

if __name__ == "__main__":
    search_sign()
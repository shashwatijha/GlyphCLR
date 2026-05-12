import pandas as pd
import os
import glob

# Search paths for flexibility
SEARCH_PATHS = ['./', 'data/text/', 'data/text/misc/']
OUTPUT_FILE = 'results/decipherment_validation_summary.csv'

if not os.path.exists('results'):
    os.makedirs('results')

def find_file(name):
    for path in SEARCH_PATHS:
        full_path = os.path.join(path, name)
        if os.path.exists(full_path):
            return full_path
    return None

# 1. Load Archaeological Anchor (GORILA)
gorila_path = find_file('Complied_Gorilla.csv')
if gorila_path:
    gorila = pd.read_csv(gorila_path)
    # Using column indices to avoid header name mismatches
    # Expected: Site, Transcription, and Code (LOWER)
    anchor_data = gorila.iloc[:, [2, 5, 11]].dropna()
    anchor_data.columns = ['Site', 'Transcription', 'Sign_Code']
else:
    print("Warning: Complied_Gorilla.csv not found.")
    anchor_data = pd.DataFrame()

# 2. Collect Linguistic Evidence
lang_files = []
for path in SEARCH_PATHS:
    lang_files.extend(glob.glob(os.path.join(path, '*_numberedWords.csv')))
    lang_files.extend(glob.glob(os.path.join(path, '*_NumberedWords.csv')))

linguistic_results = []
for f in set(lang_files): # set() removes duplicates from overlapping paths
    lang = os.path.basename(f).split('_')[0]
    try:
        df = pd.read_csv(f).iloc[:, :2]
        df.columns = ['Sign_Code', 'Phonetic_Value']
        df['Language'] = lang
        linguistic_results.append(df)
    except:
        continue

# 3. Process Consensus and Merge
if linguistic_results:
    master_lang = pd.concat(linguistic_results)
    
    # Calculate how many languages support a specific sound for a sign
    consensus = master_lang.groupby(['Sign_Code', 'Phonetic_Value']).agg({
        'Language': 'nunique'
    }).reset_index()
    consensus.columns = ['Sign_Code', 'Proposed_Sound', 'Agreement_Score']
    
    # Final merge with archaeological context
    if not anchor_data.empty:
        final_report = pd.merge(consensus, anchor_data, on='Sign_Code', how='inner')
    else:
        final_report = consensus

    # Sort by strongest evidence first
    final_report = final_report.sort_values('Agreement_Score', ascending=False)
    final_report.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Validation complete. High-confidence matches saved to {OUTPUT_FILE}")
    print("\nSample of Top Validated Results:")
    print(final_report.head(10).to_string(index=False))
else:
    print("No linguistic data files were found to process.")
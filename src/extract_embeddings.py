import torch
import torchvision.transforms as T
from PIL import Image
import os
import pandas as pd

# 1. Setup DINOv2
# Using 'vits14' for a good balance of speed and accuracy
print("Loading DINOv2 model...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

# Check for GPU (Mac M-series use 'mps', Windows/Linux use 'cuda')
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# 2. Image Pre-processing
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_embedding(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(img_t)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# 3. Process the folders
glyph_base_path = 'data/glyphs/'
data_records = []

print("Starting extraction...")

for root, dirs, files in os.walk(glyph_base_path):
    # Determine script type by folder name
    path_lower = root.lower()
    
    if 'linear_a' in path_lower:
        script_type = 'LinearA'
    elif 'linear_b' in path_lower:
        script_type = 'LinearB'
    elif 'cuneiform' in path_lower:
        script_type = 'Cuneiform'
    elif 'egyptian' in path_lower:
        script_type = 'Egyptian'
    else:
        # Skip files that aren't in a designated script folder
        continue

    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, file)
            sign_id = os.path.splitext(file)[0]
            
            print(f"Processing {script_type}: {sign_id}...")
            vector = get_embedding(full_path)
            
            if vector is not None:
                data_records.append({
                    'SignID': sign_id,
                    'Script': script_type,
                    'Embedding': vector.tolist()
                })

# 4. Save to CSV
if data_records:
    df_embeddings = pd.DataFrame(data_records)
    # Ensure the directory exists
    os.makedirs('data/text/', exist_ok=True)
    df_embeddings.to_csv('data/text/glyph_embeddings.csv', index=False)
    print(f"\nSuccess! {len(df_embeddings)} visual embeddings saved to data/text/glyph_embeddings.csv")
else:
    print("\nNo images found. Please check your folder paths.")
import torch
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as T

# 1. Load the DINOv2 Model (Small version for speed)
print("Loading DINOv2 model...")
device = "cpu" # Use "mps" for Mac M1/M2 if torch is configured, but cpu is safer for 1hr deadline
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

# 2. Define Image Transformation
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract():
    embeddings_data = []
    processed_dir = 'data/processed'
    
    print(f"Starting extraction from {processed_dir}...")
    
    # Walk through script subfolders (linear_a, linear_b)
    for root, dirs, files in os.walk(processed_dir):
        script_name = os.path.basename(root)
        if not files or script_name == 'processed': continue
        
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                
                # Process image
                img = Image.open(img_path).convert('RGB')
                img_t = transform(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = model(img_t).cpu().numpy().flatten()
                
                # Store results: [Script, SignID, Vector]
                embeddings_data.append({
                    'Script': script_name,
                    'SignID': filename.split('.')[0],
                    'Embedding': features.tolist()
                })
    
    # Save to CSV for the next step (Manifold Alignment)
    df = pd.DataFrame(embeddings_data)
    os.makedirs('data/text', exist_ok=True)
    df.to_csv('data/text/glyph_embeddings.csv', index=False)
    print(f"Extraction complete! Saved {len(df)} embeddings to data/text/glyph_embeddings.csv")

if __name__ == "__main__":
    extract()
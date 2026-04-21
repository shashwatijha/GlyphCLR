import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from src.dataset import GlyphCLRDataset
from src.model import GlyphEncoder
from torchvision import transforms
from PIL import Image
import os


# 1. Setup Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Load Model
model = GlyphEncoder().to(device)
model.load_state_dict(torch.load("glyph_encoder.pth", map_location=device))
model.eval()

# 3. Simple Transform (No heavy warping, just standardizing for the model)
simple_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 4. Load Data (We want labels this time to color the plot)
dataset = GlyphCLRDataset(root_dir="data/glyphs", transform=simple_transform)
# Note: We modified dataset.py to return (v1, v2). 
# For visualization, we just need one image and its script name.

embeddings = []
labels = []

print("Extracting embeddings...")
with torch.no_grad():
    for img_path in dataset.image_paths:
        # Get script name from folder (e.g., 'egyptian')
        script_label = img_path.split('/')[-2]
        
        image = Image.open(img_path).convert("L")
        tensor = simple_transform(image).unsqueeze(0).to(device)
        
        # Get the 128-dimensional vector
        vector = model(tensor).cpu().numpy()
        embeddings.append(vector.flatten())
        labels.append(script_label)

embeddings = np.array(embeddings)

# 5. Run t-SNE (Squish 128D down to 2D)
print("Running t-SNE (this might take a minute)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# 6. Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=embeddings_2d[:, 0], 
    y=embeddings_2d[:, 1], 
    hue=labels, 
    palette="viridis", 
    s=15, 
    alpha=0.7
)
plt.title("The Scribal Landscape: GlyphCLR Embedding Clusters")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("glyph_clusters.png")
plt.show()
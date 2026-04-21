import torch
import numpy as np
import os
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from src.dataset import GlyphCLRDataset
from torchvision import transforms

# 1. Setup
device = "cpu" # Baseline doesn't need GPU
root_dir = "data/glyphs"

# 2. Simple Flattening Transform
# We resize to 64x64 to make the pixel math faster
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = GlyphCLRDataset(root_dir=root_dir, transform=transform)

pixel_data = []
labels = []

print("Reading pixels for baseline...")
for img_path in dataset.image_paths:
    image = Image.open(img_path).convert("L")
    tensor = transform(image)
    # Flatten the 64x64 image into a 4096-length vector
    pixel_data.append(tensor.numpy().flatten())
    labels.append(img_path.split('/')[-2])

pixel_data = np.array(pixel_data)

# 3. Fit k-NN (k=5 for Top-5 Retrieval)
print("Computing pixel similarities...")
knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(pixel_data)

# 4. Evaluate Accuracy
# We check: are the 5 closest pixel-matches the same script?
distances, indices = knn.kneighbors(pixel_data)

correct = 0
total = len(labels)

for i in range(total):
    # indices[i][0] is the image itself, so we look at [1:6]
    neighbor_indices = indices[i][1:]
    neighbor_labels = [labels[idx] for idx in neighbor_indices]
    
    # If the majority of neighbors are the same script, count as "correct"
    # (Or you can check if any of top-5 match a specific target)
    if labels[i] in neighbor_labels:
        correct += 1

print(f"\n--- Baseline Results ---")
print(f"Pixel k-NN Top-5 Accuracy: {(correct/total)*100:.2f}%")
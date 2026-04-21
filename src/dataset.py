import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class GlyphCLRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # This grabs all .png files from all subfolders
        self.image_paths = glob.glob(os.path.join(root_dir, "*/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L") # Ensure grayscale
        
        # In Contrastive Learning, we return two different versions of the same image
        if self.transform:
            view1 = self.transform(image)
            view2 = self.transform(image)
        else:
            view1 = view2 = transforms.ToTensor()(image)
            
        return view1, view2

# Define the "Real World" augmentations
data_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ElasticTransform(alpha=50.0), # The "Handwritten" effect
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    # Normalize based on grayscale (0.5 mean/std is a good starting point)
    transforms.Normalize((0.5,), (0.5,))
])

# To test:
# dataset = GlyphCLRDataset(root_dir="data/glyphs", transform=data_transforms)
# loader = DataLoader(dataset, batch_size=32, shuffle=True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Initialize dataset
    # We point to data/glyphs because we run from the root GlyphCLR folder
    dataset = GlyphCLRDataset(root_dir="data/glyphs", transform=data_transforms)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. Grab one batch
    view1, view2 = next(iter(loader))

    # 3. Visualize the results
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        # View 1 (Augmented version 1)
        ax[0, i].imshow(view1[i].squeeze(), cmap='gray')
        ax[0, i].set_title(f"Pair {i} - View 1")
        ax[0, i].axis('off')
        
        # View 2 (Augmented version 2)
        ax[1, i].imshow(view2[i].squeeze(), cmap='gray')
        ax[1, i].set_title(f"Pair {i} - View 2")
        ax[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    print(f"Dataset loaded with {len(dataset)} images.")
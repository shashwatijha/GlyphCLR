import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import GlyphCLRDataset, data_transforms
from src.model import GlyphEncoder
import torch.nn.functional as F

# 1. Setup Device (Uses Mac GPU if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20

# 3. Data & Model
dataset = GlyphCLRDataset(root_dir="data/glyphs", transform=data_transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = GlyphEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. Contrastive Loss Function (Simplified InfoNCE)
def info_nce_loss(features, batch_size, temperature=0.5):
    # Features contains [view1_batch, view2_batch]
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0).to(device)
    
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    
    # Remove self-similarity from the diagonal
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)
    
    similarity_matrix = similarity_matrix / temperature
    return F.cross_entropy(similarity_matrix, labels)

# 5. Training Loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for view1, view2 in loader:
        # Standard SimCLR logic: concatenate views and pass through model
        images = torch.cat([view1, view2], dim=0).to(device)
        
        optimizer.zero_grad()
        features = model(images)
        
        # We need to handle the last batch if it's smaller than BATCH_SIZE
        curr_batch_size = view1.size(0)
        loss = info_nce_loss(features, curr_batch_size)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(loader):.4f}")

# 6. Save the Brain
torch.save(model.state_dict(), "glyph_encoder.pth")
print("Training complete. Model saved as glyph_encoder.pth")
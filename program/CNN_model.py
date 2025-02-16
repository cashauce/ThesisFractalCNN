import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# LOAD MRI IMAGES

DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed"))

class MRIDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_files = os.listdir(folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        img = Image.open(img_path)  # Load as-is
        img = torch.tensor(np.array(img), dtype=torch.float32)  # Convert to tensor
        img = img.permute(2, 0, 1) if len(img.shape) == 3 else img.unsqueeze(0)  # Ensure (C, H, W)
        return img

# Load dataset
dataset = MRIDataset(DATASET_PATH)

# Split dataset into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train images: {len(train_dataset)}, Test images: {len(test_dataset)}")

# DEFINE CNN MODEL

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Feature vector of 128
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)  # Feature vector
        return x

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

# TRAINING

criterion = nn.MSELoss()  # Autoencoder-style loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images in train_loader:
        images = images.to(device)

        # Forward pass
        features = model(images)
        loss = criterion(features, torch.zeros_like(features))  # Encourage compact features

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed!")

# EXTRACTING FEATURES

feature_vectors = []
model.eval()
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        features = model(images)
        feature_vectors.append(features.cpu().numpy())

feature_vectors = np.vstack(feature_vectors)
np.save("mri_features.npy", feature_vectors)
print("Features saved as 'mri_features.npy'!")

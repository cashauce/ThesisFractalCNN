import os
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import time
from program.util import getTime

# LOAD MRI IMAGES
class MRIDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_files = os.listdir(folder)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        img = Image.open(img_path).convert("L") 
        img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)  
        return img


# DEFINE CNN MODEL
class CNNModel(nn.Module):
    def __init__(self, input_size=256):  # Changed default input size to match image size
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # Calculate the flattened size
        self.input_size = input_size
        self.flattened_size = self._calculate_flattened_size()
        
        # Adjust fully connected layer
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc_dropout = nn.Dropout(0.5)

    def _calculate_flattened_size(self):
        dummy_input = torch.zeros(1, 1, self.input_size, self.input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))  # 128x128
        x = self.pool(F.relu(self.conv2(x)))           # 64x64
        x = self.pool(F.relu(self.conv3(x)))          # 32x32
        return x.numel()

    def forward(self, x):
        # Print shape for debugging
        # print(f"Input shape: {x.shape}")
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"After flatten shape: {x.shape}")
        
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        
        return x


# TRAINING FUNCTION
def train_cnn_model(dataset_path, output_path, num_epochs=10, batch_size=32):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MRIDataset(dataset_path)
    
    # Split dataset into train (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.MSELoss()  # Autoencoder-style loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    print("Training complete")

    # EXTRACTING FEATURES
    feature_vectors = []
    model.eval()
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            features = model(images)
            feature_vectors.append(features.cpu().numpy())

    feature_vectors = np.vstack(feature_vectors)
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, "mri_features.npy"), feature_vectors)
    
    end_time = time.time()
    total_time = getTime(end_time - start_time)
    print(f"CNN model training time: {total_time}")
    print(f"Features saved at {output_path}/mri_features.npy")

# Ensure script execution only happens when run directly
if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed"))
    output_path = "data/features"
    train_cnn_model(dataset_path, output_path)

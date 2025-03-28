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
from torchvision import transforms

# LOAD MRI IMAGES
class MRIDataset(Dataset):
    def __init__(self, folder, target_size=(256, 256), max_images=10000):
        self.folder = folder
        self.target_size = target_size
        print(f"Loading preprocessed images from: {folder}")
        
        # Get all image files from preprocessed folder
        self.image_files = [f for f in os.listdir(folder) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Limit the number of images
        if len(self.image_files) > max_images:
            print(f"Limiting dataset to {max_images} images")
            self.image_files = self.image_files[:max_images]
        
        if not self.image_files:
            raise ValueError(f"No images found in {folder}")
        
        print(f"Total preprocessed images loaded: {len(self.image_files)}")

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.folder, self.image_files[idx])
            img = Image.open(img_path).convert("L")
            img = self.transform(img)
            
            # Only verify tensor dimensions without printing
            if img.shape != (1, self.target_size[0], self.target_size[1]):
                return None
            
            return img
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None


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

        # Add decoder layers for reconstruction
        self.decoder_fc = nn.Linear(128, self.flattened_size)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def _calculate_flattened_size(self):
        dummy_input = torch.zeros(1, 1, self.input_size, self.input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))  # 128x128
        x = self.pool(F.relu(self.conv2(x)))           # 64x64
        x = self.pool(F.relu(self.conv3(x)))          # 32x32
        return x.numel()

    def forward(self, x):
        # Encoder
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        # Get features
        features = F.relu(self.fc1(x))
        features = self.fc_dropout(features)
        
        # Decoder for reconstruction
        x = F.relu(self.decoder_fc(features))
        x = x.view(-1, 128, 32, 32)  # Reshape to match encoder output
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))  # Use tanh for normalized output
        
        return features, x


# TRAINING FUNCTION
def train_cnn_model(dataset_path, output_path, num_epochs=10, batch_size=32):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MRIDataset(dataset_path)
    
    # Filter out None values from dataset
    valid_indices = []
    for i in range(len(dataset)):
        if dataset[i] is not None:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        raise ValueError("No valid images found in the dataset!")
    
    print(f"Valid images found: {len(valid_indices)} out of {len(dataset)}")
    
    # Use only valid indices for training
    train_size = int(0.8 * len(valid_indices))
    test_size = len(valid_indices) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    criterion = nn.MSELoss()
    # Reduce initial learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )

    train_losses = []
    val_losses = []

    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 50)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 30)

        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)

            # Forward pass
            features, reconstructed = model(images)
            # Reduce loss scaling from 100 to 10
            loss = criterion(reconstructed, images) * 10

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Reduce clip value from 1.0 to 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # Print batch progress
            if (batch_idx + 1) % 5 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.6f}")

        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                features, reconstructed = model(images)
                val_loss += criterion(reconstructed, images).item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        print("\nEpoch Summary:")
        print(f"Training Loss: {avg_epoch_loss:.6f}")
        print(f"Validation Loss: {avg_val_loss:.6f}")
        print("=" * 50)

    # Save the trained model
    model_save_path = os.path.join(output_path, "cnn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    # EXTRACTING FEATURES
    feature_vectors = []
    model.eval()
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            features, _ = model(images)
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

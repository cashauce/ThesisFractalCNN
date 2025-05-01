import os
import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import time
from program.util import getTime, cnn_metrics_csv
from torchvision import transforms

class MRIDataset(Dataset):
    def __init__(self, folder, target_size=(256, 256), max_images=10000):
        self.folder = folder
        self.target_size = target_size
        print(f"Loading preprocessed images from: {folder}")
        
        self.image_files = [f for f in os.listdir(folder) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
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
            
            if img.shape != (1, self.target_size[0], self.target_size[1]):
                return None
            
            return img
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None

class CNNModel(nn.Module):
    def __init__(self, input_size=256):
        super(CNNModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.input_size = input_size
        self.flattened_size = self._calculate_flattened_size()
        
        self.feature_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, self.flattened_size),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def _calculate_flattened_size(self):
        x = torch.zeros(1, 1, self.input_size, self.input_size)
        x = self.encoder(x)
        return x.numel()

    def forward(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        features = self.feature_layers(flattened)

        decoded = self.decoder(features)
        decoded = decoded.view(-1, 128, self.input_size//8, self.input_size//8)
        reconstructed = self.decoder_conv(decoded)
        
        return features, reconstructed

def train_cnn_model(dataset_path, output_path, num_epochs=10, batch_size=32):
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MRIDataset(dataset_path)
    
    valid_indices = []
    for i in range(len(dataset)):
        if dataset[i] is not None:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        raise ValueError("No valid images found in the dataset!")
    
    print(f"Valid images found: {len(valid_indices)} out of {len(dataset)}")
    
    train_size = int(0.8 * len(valid_indices))
    test_size = len(valid_indices) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel().to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

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
            start_time = time.time()
            images = images.to(device)

            features, reconstructed = model(images)
            
            mse_loss = mse_criterion(reconstructed, images)
            l1_loss = l1_criterion(reconstructed, images)
            loss = (0.8 * mse_loss + 0.2 * l1_loss) * 5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss.item():.6f}")
            
            end_time = time.time()
            trainingTime = round((end_time - start_time), 4)
            cnn_metrics_csv(epoch+1, batch_count, loss.item(), trainingTime, "trained_cnn_CSV.csv")

        avg_epoch_loss = epoch_loss / batch_count
        train_losses.append(avg_epoch_loss)

        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for images in test_loader:
                images = images.to(device)
                features, reconstructed = model(images)
                val_loss += mse_criterion(reconstructed, images).item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)

        scheduler.step()

        print("\nEpoch Summary:")
        print(f"Training Loss: {avg_epoch_loss:.6f}")
        print(f"Validation Loss: {avg_val_loss:.6f}")
        print("=" * 50)

    model_save_path = os.path.join(output_path, "cnn_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

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

if __name__ == "__main__":
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "preprocessed"))
    output_path = "data/features"
    train_cnn_model(dataset_path, output_path)

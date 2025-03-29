import sys
import os
import numpy as np
import time
from program.util import save_csv
from skimage import io, img_as_ubyte, transform
from skimage.transform import AffineTransform, warp
from skimage.exposure import rescale_intensity, is_low_contrast
from skimage.io import imsave
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from skimage.color import rgb2gray
from program.CNN_model import CNNModel  # Import the custom CNNModel class
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Load pre-trained MobileNetV2 model for feature extraction
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cnn_model(model_path, device, input_size=64):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both new and old save formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        input_size = checkpoint.get('input_size', input_size)
        model = CNNModel(input_size=input_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Old format or direct state dict
        model = CNNModel(input_size=input_size).to(device)
        try:
            model.load_state_dict(checkpoint)
        except:
            print("Warning: Direct load failed, attempting to modify checkpoint...")
            # Adjust the checkpoint to match the current model's structure
            for key in list(checkpoint.keys()):
                if key not in model.state_dict() or checkpoint[key].shape != model.state_dict()[key].shape:
                    print(f"Removing mismatched layer: {key}")
                    del checkpoint[key]
            model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    return model

def load_image(file_path, target_size=(256, 256)):
    try:
        image = io.imread(file_path, as_gray=True)
        image = transform.resize(image, target_size, anti_aliasing=True)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Normalize to [0, 1]
        return image.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Error loading image at {file_path}: {e}")

# Partition image into smaller blocks
def partition_image(image, block_size):
    h, w = image.shape[:2]
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            
            if block.shape[:2] != (block_size, block_size):
                block = np.pad(block, ((0, block_size - block.shape[0]), (0, block_size - block.shape[1])), mode='edge')
            blocks.append(block)
    return blocks

# Apply affine transformation
def apply_affine_transformation(block, transformation):
    scale, rotation, tx, ty = transformation
    h, w = block.shape

    transformation_matrix = np.array([
        [scale * np.cos(rotation), -scale * np.sin(rotation), tx],
        [scale * np.sin(rotation), scale * np.cos(rotation), ty]
    ])

    y, x = np.indices((h, w))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())]) 

    transformed_coords = transformation_matrix @ coords
    x_transformed = transformed_coords[0, :].reshape(h, w)
    y_transformed = transformed_coords[1, :].reshape(h, w)

    x_transformed = np.clip(x_transformed, 0, w - 1).astype(np.int32)
    y_transformed = np.clip(y_transformed, 0, h - 1).astype(np.int32)

    transformed_block = block[y_transformed, x_transformed]
    return transformed_block

def extract_features(block, cnn_model, device):
    # Reshape and normalize block
    block_tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        features, _ = cnn_model(block_tensor)  # Get features from the encoder part
        features = features.view(features.size(0), -1)  # Flatten the features
        return features.squeeze().cpu().numpy()

def extract_features_batch(blocks, cnn_model, device, batch_size=64):
    n_blocks = len(blocks)
    features = []
    
    for i in range(0, n_blocks, batch_size):
        batch = blocks[i:i + batch_size]
        batch_tensor = torch.stack([torch.tensor(b, dtype=torch.float32).unsqueeze(0) for b in batch]).to(device)
        with torch.no_grad():
            batch_features, _ = cnn_model(batch_tensor)  # Get features from the encoder part
            batch_features = batch_features.view(batch_features.size(0), -1)  # Flatten the features
            features.append(batch_features.cpu().numpy())
    
    return np.vstack(features)

class KDNode:
    def __init__(self, point, index, left=None, right=None):
        self.point = point  # the feature vector
        self.index = index  # index of the original block
        self.left = left
        self.right = right

def build_kdtree(points, indices, depth=0):
    if len(points) == 0:
        return None

    k = points.shape[1]  # feature vector dimension
    axis = depth % k

    # Sort points and indices together based on the current axis
    sorted_indices = np.argsort(points[:, axis])
    points = points[sorted_indices]
    indices = indices[sorted_indices]
    
    median = len(points) // 2

    return KDNode(
        point=points[median],
        index=indices[median],
        left=build_kdtree(points[:median], indices[:median], depth + 1),
        right=build_kdtree(points[median + 1:], indices[median + 1:], depth + 1)
    )

def find_nearest_in_kdtree(node, target, best=None, best_dist=float('inf'), depth=0):
    if node is None:
        return best, best_dist

    k = len(target)
    axis = depth % k
    
    current_dist = np.sum((node.point - target) ** 2)
    
    if current_dist < best_dist:
        best = node
        best_dist = current_dist

    if target[axis] < node.point[axis]:
        first, second = node.left, node.right
    else:
        first, second = node.right, node.left

    best, best_dist = find_nearest_in_kdtree(first, target, best, best_dist, depth + 1)
    
    if abs(target[axis] - node.point[axis]) ** 2 < best_dist:
        best, best_dist = find_nearest_in_kdtree(second, target, best, best_dist, depth + 1)
    
    return best, best_dist

def encode_image_with_kdtree_manual(image, block_size=8, cnn_model=None, device=None):
    range_blocks = partition_image(image, block_size)
    domain_blocks = range_blocks  # Use same blocks for range and domain to reduce computation

    # Convert blocks to batch tensor for faster processing
    batch_tensor = torch.stack([torch.tensor(b, dtype=torch.float32).unsqueeze(0) for b in domain_blocks]).to(device)
    
    # Extract features for all blocks in one batch
    with torch.no_grad():
        all_features, _ = cnn_model(batch_tensor)
        all_features = all_features.cpu().numpy()
    
    # Split features into domain and range (they're the same in this case)
    domain_features = all_features
    range_features = all_features

    # Build KD-tree using domain features
    print("Building KD-tree...")
    domain_indices = np.arange(len(domain_blocks))
    kd_tree = build_kdtree(domain_features, domain_indices)

    encoded_data = []
    transformation = (1.0, 0.0, 1, 1)

    print("Finding best matches using KD-tree...")
    with tqdm(total=len(range_blocks), desc="Encoding Image", unit="block", colour="red") as pbar:
        for idx, feature in enumerate(range_features):
            # Find best match using KD-tree
            best_node, _ = find_nearest_in_kdtree(kd_tree, feature)
            best_index = best_node.index
            
            # Store mapping without applying transformation yet
            encoded_data.append((best_index, transformation))
            pbar.update(1)

    return encoded_data, domain_blocks

# Decode the image
def decode_image(encoded_data, domain_blocks, image_shape, block_size=8, output_file=None, output_path='data/compressed/fractal'):
    os.makedirs(output_path, exist_ok=True)
    reconstructed_image = np.zeros(image_shape, dtype=np.float64)
    h, w = image_shape
    idx = 0

    # Decode the data from the raw binary format and include transformation info
    decoded_data = [(int(entry[0]), entry[1]) for entry in encoded_data]

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if idx < len(decoded_data):
                best_index, transformation = decoded_data[idx]
                transformed_block = apply_affine_transformation(domain_blocks[best_index], transformation)
                reconstructed_image[i:i + block_size, j:j + block_size] = transformed_block
                idx += 1

    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    if is_low_contrast(reconstructed_image):
        reconstructed_image = rescale_intensity(reconstructed_image, in_range='image', out_range=(0, 1))

    reconstructed_image = img_as_ubyte(reconstructed_image)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imsave(output_file, reconstructed_image)

def process_image_batch(images, image_files, original_path, output_path, block_size, cnn_model, device):
    """Process all images in two phases"""
    all_kd_data = []  # Store KD-trees and features for all images
    
    # Phase 1: Build KD-trees for all images
    print("\nPhase 1: Building KD-trees for all images...")
    for idx, (image, image_file) in enumerate(zip(images, image_files), 1):
        print(f"\nProcessing KD-tree for image {idx}/{len(images)}: {image_file}")
        
        # Partition and get features
        blocks = partition_image(image, block_size)
        batch_tensor = torch.stack([torch.tensor(b, dtype=torch.float32).unsqueeze(0) for b in blocks]).to(device)
        
        with torch.no_grad():
            features, _ = cnn_model(batch_tensor)
            features = features.cpu().numpy()
        
        # Build KD-tree
        domain_indices = np.arange(len(blocks))
        kd_tree = build_kdtree(features, domain_indices)
        
        # Store data for phase 2
        all_kd_data.append({
            'kd_tree': kd_tree,
            'features': features,
            'blocks': blocks,
            'image': image,
            'file': image_file
        })
    
    # Phase 2: Process transformations for all images
    print("\nPhase 2: Processing transformations for all images...")
    transformation = (1.0, 0.0, 1, 1)
    
    for idx, data in enumerate(all_kd_data, 1):
        print(f"\nProcessing transformations for image {idx}/{len(images)}: {data['file']}")
        
        # Start encoding time measurement
        start_time = time.time()
        
        # Encode using pre-built KD-tree
        encoded_data = []
        with tqdm(total=len(data['features']), desc="Encoding Image", unit="block", colour="red") as pbar:
            for feature in data['features']:
                best_node, _ = find_nearest_in_kdtree(data['kd_tree'], feature)
                encoded_data.append((best_node.index, transformation))
                pbar.update(1)
        
        encoding_time = round(time.time() - start_time, 4)
        
        # Save compressed image
        compressed_file = f"compressed_{os.path.splitext(data['file'])[0]}.jpg"
        output_file = os.path.join(output_path, compressed_file)
        
        # Start decoding time measurement
        start_time = time.time()
        decode_image(encoded_data, data['blocks'], data['image'].shape, 
                    block_size, output_file=output_file, output_path=output_path)
        decoding_time = round(time.time() - start_time, 4)
        
        # Save metrics with rounded times
        image_path = os.path.join(original_path, data['file'])
        save_csv(data['image'], image_path, output_file, data['file'], 
                compressed_file, encoding_time, decoding_time)

def run_enhanced_compression(original_path, output_path, limit, block_size=8):
    cnn_model_path = "data/features/cnn_model.pth"
    cnn_model = load_cnn_model(cnn_model_path, device, input_size=block_size)

    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])[:limit]
    print(f"Compressing {limit} image/s in '{original_path}' using enhanced fractal compression...")
    
    # Load all images first
    images = []
    valid_files = []
    for image_file in image_files:
        try:
            image_path = os.path.join(original_path, image_file)
            image = load_image(image_path)
            images.append(image)
            valid_files.append(image_file)
        except ValueError as e:
            print(f"Error loading {image_file}: {e}")
            continue
    
    # Process all images in batch
    process_image_batch(images, valid_files, original_path, output_path, block_size, cnn_model, device)
    
    print(f"***Finished compressing {len(valid_files)} image/s***")
    sys.exit(1)

def modify_checkpoint(model_path, new_model):
    checkpoint = torch.load(model_path, map_location="cpu")
    new_state_dict = new_model.state_dict()

    # Remove mismatched layers
    for key in list(checkpoint.keys()):
        if key not in new_state_dict or checkpoint[key].shape != new_state_dict[key].shape:
            print(f"Removing mismatched layer: {key}")
            del checkpoint[key]

    # Load the modified checkpoint into the new model
    new_model.load_state_dict(checkpoint, strict=False)
    return new_model

# Example usage
# cnn_model = CNNModel().to(device)
# cnn_model = modify_checkpoint("data/features/cnn_model.pth", cnn_model)

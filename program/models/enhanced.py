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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_cnn_model(model_path, device, input_size=64):
    model = CNNModel(input_size=input_size).to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # Adjust the checkpoint to match the current model's structure
    checkpoint_state_dict = checkpoint
    model_state_dict = model.state_dict()

    for key in checkpoint_state_dict.keys():
        if key in model_state_dict and checkpoint_state_dict[key].shape != model_state_dict[key].shape:
            print(f"Resizing layer: {key} from {checkpoint_state_dict[key].shape} to {model_state_dict[key].shape}")
            if "weight" in key:
                checkpoint_state_dict[key] = torch.nn.functional.interpolate(
                    checkpoint_state_dict[key].unsqueeze(0).unsqueeze(0),
                    size=model_state_dict[key].shape,
                    mode="nearest"
                ).squeeze(0).squeeze(0)
            elif "bias" in key:
                checkpoint_state_dict[key] = torch.zeros_like(model_state_dict[key])

    # Load the adjusted checkpoint
    model.load_state_dict(checkpoint_state_dict, strict=False)
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
    block_tensor = torch.tensor(block, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        return cnn_model(block_tensor).squeeze().cpu().numpy()

def extract_features_batch(blocks, cnn_model, device, batch_size=64):
    n_blocks = len(blocks)
    features = []
    
    # Convert blocks to tensor in batches
    for i in range(0, n_blocks, batch_size):
        batch = blocks[i:i + batch_size]
        batch_tensor = torch.stack([torch.tensor(b, dtype=torch.float32).unsqueeze(0) for b in batch]).to(device)
        with torch.no_grad():
            batch_features = cnn_model(batch_tensor).cpu().numpy()
        features.append(batch_features)
    
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
    domain_blocks = partition_image(image, block_size)

    # Extract CNN features in batches for better performance
    print("Extracting features for domain blocks...")
    domain_features = extract_features_batch(domain_blocks, cnn_model, device)
    
    # Build KD-tree using domain features
    print("Building KD-tree...")
    domain_indices = np.arange(len(domain_blocks))
    kd_tree = build_kdtree(domain_features, domain_indices)

    encoded_data = []
    transformation = (1.0, 0.0, 1, 1)

    print("Finding best matches using KD-tree...")
    start_time = time.time()
    with tqdm(total=len(range_blocks), desc="Encoding Image", unit="block", colour="green") as pbar:
        for idx, block in enumerate(range_blocks):
            # Extract features for current block
            feature = extract_features(block, cnn_model, device)
            
            # Find best match using KD-tree
            best_node, _ = find_nearest_in_kdtree(kd_tree, feature)
            best_index = best_node.index
            
            # Apply transformation
            transformed_block = apply_affine_transformation(domain_blocks[best_index], transformation)
            encoded_data.append((best_index, transformation))
            pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    bps = len(range_blocks) / elapsed_time if elapsed_time > 0 else 0
    bps = round((bps), 4)

    return encoded_data, domain_blocks, bps

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

# Function to compress and evaluate images in a folder using fractal compression
def run_enhanced_compression(original_path, output_path, limit, block_size=8):
    cnn_model_path = "data/features/cnn_model.pth"  # Path to the pre-trained CNN model
    cnn_model = load_cnn_model(cnn_model_path, device, input_size=block_size)  # Use block_size as input_size

    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    image_files = image_files[:limit]  # Limit the number of images to process
    print(f"Compressing {limit} image/s in '{original_path}' using enhanced fractal compression...")

    for idx, image_file in enumerate(image_files, start=1):
        process_single_image(image_file, original_path, output_path, block_size, cnn_model, device)

    print(f"***Finished compressing {limit} image/s***")
    sys.exit(1)

def process_single_image(image_file, original_path, output_path, block_size, cnn_model, device):
    image_path = os.path.join(original_path, image_file)
    compressed_file = f"compressed_{os.path.splitext(image_file)[0]}.jpg"
    output_file = os.path.join(output_path, compressed_file)
    print(f"Processing {image_file}...")

    try:
        image = load_image(image_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    start_time = time.time()
    encoded_data, domain_blocks, bps = encode_image_with_kdtree_manual(image, block_size, cnn_model, device)
    end_time = time.time()
    encodingTime = round((end_time - start_time), 4)

    start_time = time.time()
    decode_image(encoded_data, domain_blocks, image.shape, block_size, output_file=output_file, output_path=output_path)
    end_time = time.time()
    decodingTime = round((end_time - start_time), 4)

    save_csv(image, image_path, output_file, image_file, compressed_file, encodingTime, decodingTime, bps, output_path)

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

import sys
import os
import time
import torch
import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from program.util import multiRun_csv, evaluate_compression
from program.CNN_model import CNNModel
from skimage import io, img_as_ubyte, transform
from skimage.exposure import rescale_intensity, is_low_contrast
from skimage.io import imsave
from tqdm import tqdm
from scipy.interpolate import make_interp_spline

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

def encode_image_with_kdtree(image, block_size=8, cnn_model=None, device=None):
    range_blocks = partition_image(image, block_size)
    domain_blocks = range_blocks  # Use same blocks for both to reduce computation

    # Extract all features at once in a single batch
    start_time = time.perf_counter()
    batch_tensor = torch.stack([torch.tensor(b, dtype=torch.float32).unsqueeze(0) for b in domain_blocks]).to(device)
    with torch.no_grad():
        all_features, _ = cnn_model(batch_tensor)
        all_features = all_features.view(all_features.size(0), -1).cpu().numpy()
    inference_time = round((time.perf_counter() - start_time) * 1000, 4)

    # Build KD-tree using domain features
    start_time = time.perf_counter()
    domain_indices = np.arange(len(domain_blocks))
    kd_tree = build_kdtree(all_features, domain_indices)
    buildingTree_time = round((time.perf_counter() - start_time) * 1000, 4)

    # Search for nearest neighbors
    encoded_data = []
    transformation = (1.0, 0.0, 1, 1)  # Fixed transformation
    total_search_time = 0
    
    start_encoding = time.perf_counter()
    with tqdm(total=len(range_blocks), desc="Encoding Image", unit="block", colour="green") as pbar:
        for feature in all_features:  # Use pre-computed features
            search_start = time.perf_counter()
            best_node, _ = find_nearest_in_kdtree(kd_tree, feature)
            total_search_time += time.perf_counter() - search_start
            
            encoded_data.append((best_node.index, transformation))
            pbar.update(1)

    nearestSearch_time = round((total_search_time / len(range_blocks)) * 1000, 4)
    bps = round(len(range_blocks) / (time.perf_counter() - start_encoding), 4)

    return encoded_data, domain_blocks, bps, buildingTree_time, nearestSearch_time, inference_time

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
    output_path = "data/compressed/test"
    method = "proposed"
    cnn_model_path = "data/features/cnn_model.pth"  # Path to the pre-trained CNN model

    if not os.path.exists(cnn_model_path):
        # https://drive.google.com/file/d/1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW/view?usp=drive_link
        file_id = "1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW"
        print(f"\n\nDownloading the CNN model with extracted features...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", cnn_model_path, quiet=False)
    
    cnn_model = load_cnn_model(cnn_model_path, device, input_size=block_size)  # Use block_size as input_size

    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"\n\nCompressing {limit} image/s in '{original_path}' using enhanced fractal compression...")

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    processed_count = 0  # Count of newly compressed images
    compression_data = []  # List to store compression metrics for plotting
    
    for image_file in image_files:
        if processed_count >= limit:
            break  # Stop when we have compressed 'limit' new images

        base_filename = f"{method}_compressed_{os.path.splitext(image_file)[0]}.jpg"
        output_file = os.path.join(output_path, base_filename)

        # Check if the file already exists and generate a new filename with a number
        counter = 2
        while os.path.exists(output_file):
            base_filename = f"{method}_compressed_{os.path.splitext(image_file)[0]}_{counter}.jpg"
            output_file = os.path.join(output_path, base_filename)
            counter += 1

        print(f"[Processing {processed_count+1}/{limit}] {image_file}...")
        image_path = os.path.join(original_path, image_file)
        image = load_image(image_path)

        start_time = time.perf_counter()
        encoded_data, domain_blocks, bps, buildingTree_time, nearestSearch_time, inference_time = encode_image_with_kdtree(image, block_size, cnn_model, device)
        end_time = time.perf_counter()
        encodingTime = round((end_time - start_time), 4)

        start_time = time.perf_counter()
        decode_image(encoded_data, domain_blocks, image.shape, block_size, output_file=output_file, output_path=output_path)
        end_time = time.perf_counter()
        decodingTime = round((end_time - start_time), 4)

        # Collect data for graph plotting
        _, _, _, psnr, ssim = evaluate_compression(image, image_path, output_file)

        compression_data.append({
            'Method': method,
            'Image': image_file,
            'Encoding Time (s)': encodingTime,
            'Decoding Time (s)': decodingTime,
            'PSNR (dB)': psnr,
            'SSIM': ssim
        })

        multiRun_csv(
                method, image, image_path, output_file, image_file, base_filename,
                buildingTree_time, nearestSearch_time, inference_time, encodingTime, decodingTime, bps, "singleRun_CSV.csv"
            )
        processed_count += 1

    # Plot the metrics after compression
    plot_compression_metrics(compression_data)

    """# Delete the cnn file after processing
    if os.path.exists(cnn_model_path):
        os.remove(cnn_model_path)"""
        
    print(f"***Finished compressing {limit} image/s***")
    sys.exit(1)






def plot_compression_metrics(compression_data):
    # Convert list of dictionaries into DataFrame
    df = pd.DataFrame(compression_data)

    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)

    # Define metrics and titles
    metrics = ["Encoding Time (s)", "Decoding Time (s)", "PSNR (dB)", "SSIM"]
    titles = [
        "Encoding Time Comparison",
        "Decoding Time Comparison",
        "PSNR (dB) Comparison",
        "SSIM Comparison"
    ]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        for method in df["Method"].unique():
            method_data = df[df["Method"] == method]
            y = method_data[metric].dropna()
            x = np.arange(len(y))

            # Special linestyle for Proposed method
            line_kwargs = {"label": method}
            if method == "Proposed":
                line_kwargs["linestyle"] = "--"

            if len(y) > 3 and len(np.unique(x)) > 3:
                try:
                    x_new = np.linspace(x.min(), x.max(), 300)
                    spline = make_interp_spline(x, y, k=3)
                    y_smooth = spline(x_new)
                    ax.plot(x_new, y_smooth, **line_kwargs)
                except Exception as e:
                    print(f"[Warning] Spline failed for {method} - {metric}: {e}")
                    ax.plot(x, y, **line_kwargs)
            else:
                ax.plot(x, y, **line_kwargs)

        ax.set_title(titles[i])
        ax.set_xlabel("Images Compressed")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend(title="Method", loc='best')

    # Final layout adjustment
    plt.tight_layout()
    plt.show()








"""# multi testing function
# Function to compress and evaluate images in a folder using fractal compression
def run_enhanced_compression(original_path, output_path, limit, block_size=8):
    # https://drive.google.com/file/d/1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW/view?usp=drive_link
    #file_id = "1ZEJl6nB2GBOLIuzd3TSWwjVL2Obf-LXW"
    cnn_model_path = "data/features/cnn_model.pth"  # Path to the pre-trained CNN model
    #print(f"\n\nDownloading the CNN model with extracted features...")
    #gdown.download(f"https://drive.google.com/uc?id={file_id}", cnn_model_path, quiet=False)
    cnn_model = load_cnn_model(cnn_model_path, device, input_size=block_size)  # Use block_size as input_size

    glioma_original_path = "data/dataset/glioma"
    pituitary_original_path = "data/dataset/pituitary"
    output_path = "data/compressed/multiTest"
    print(f"Compressing all glioma and pituitary images using proposed fractal compression...")

    glioma = [
        "glioma_0132.jpg", "glioma_0990.jpg", "glioma_0322.jpg", "glioma_0296.jpg", "glioma_0102.jpg",
        "glioma_0840.jpg", "glioma_0557.jpg", "glioma_0386.jpg", "glioma_0769.jpg", "glioma_0159.jpg",
        "glioma_0934.jpg", "glioma_0171.jpg", "glioma_0111.jpg", "glioma_0918.jpg", "glioma_0558.jpg"
    ]

    pituitary = [
        "pituitary_0048.jpg", "pituitary_0681.jpg", "pituitary_0287.jpg", "pituitary_0798.jpg", "pituitary_0598.jpg",
        "pituitary_0274.jpg", "pituitary_0873.jpg", "pituitary_0971.jpg", "pituitary_0816.jpg", "pituitary_0217.jpg",
        "pituitary_0106.jpg", "pituitary_0249.jpg", "pituitary_0498.jpg", "pituitary_0218.jpg", "pituitary_0499.jpg"
    ]

    selected_images = glioma + pituitary
    method = "proposed"
    total_runs = 5

    for testRuns in range(1, total_runs + 1):
        print(f"\n>>> Starting run {testRuns} of {total_runs}...\n")

        for idx, image_file in enumerate(selected_images, start=1):
            if "glioma" in image_file:
                image_path = os.path.join(glioma_original_path, image_file)
            else:
                image_path = os.path.join(pituitary_original_path, image_file)

            compressed_file = f"{method}_{testRuns}_compressed_{os.path.splitext(image_file)[0]}.jpg"
            output_file = os.path.join(output_path, compressed_file)
            print(f"[Run {testRuns}] [Image {idx}/{len(selected_images)}] Processing {image_file}...")

            image = load_image(image_path)

            start_time = time.perf_counter()
            encoded_data, domain_blocks, bps, buildingTree_time, nearestSearch_time, inference_time = encode_image_with_kdtree(image, block_size, cnn_model, device)
            end_time = time.perf_counter()
            encodingTime = round((end_time - start_time), 4)

            start_time = time.perf_counter()
            decode_image(encoded_data, domain_blocks, image.shape, block_size, output_file=output_file, output_path=output_path)
            end_time = time.perf_counter()
            decodingTime = round((end_time - start_time), 4)

            multiRun_csv(
                method, testRuns, 
                image, image_path, output_file, image_file, compressed_file,
                buildingTree_time, nearestSearch_time, inference_time, encodingTime, decodingTime, bps, "multiTest_CSV.csv"
            )

    print(f"\n*** Finished all {total_runs} runs for {len(selected_images)} images ***")
    sys.exit(1)"""


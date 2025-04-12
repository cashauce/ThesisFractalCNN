import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from program.util import multiRun_csv, evaluate_compression
from skimage import io, img_as_ubyte, transform
from skimage.transform import AffineTransform, warp
from skimage.exposure import rescale_intensity, is_low_contrast
from skimage.io import imsave
from tqdm import tqdm
from scipy.interpolate import make_interp_spline


def load_image(file_path, target_size=(256, 256)):
    image = io.imread(file_path, as_gray=True)
    image = transform.resize(image, target_size, anti_aliasing=True) 
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) 
    return image.astype(np.float32)

# KD-tree node class
class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point  # the block vector
        self.left = left    # left child
        self.right = right  # right child

# build KD-tree manually
def build_kdtree(points, depth=0):
    if not points:
        return None

    k = len(points[0])  # block vector length
    axis = depth % k

    points.sort(key=lambda x: x[axis])
    median = len(points) // 2 

    # create node and construct subtrees
    return KDNode(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

# calculate euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

# find nearest neighbor in the KD-tree
def kdtree_nearest_neighbor(node, target, depth=0, best=None):
    if node is None:
        return best

    k = len(target)
    axis = depth % k

    # update best if the current node is closer
    if best is None or euclidean_distance(target, node.point) < euclidean_distance(target, best):
        best = node.point

    next_branch = None
    opposite_branch = None
    if target[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    # search the next branch
    best = kdtree_nearest_neighbor(next_branch, target, depth + 1, best)

    # check the other branch if necessary
    if abs(target[axis] - node.point[axis]) < euclidean_distance(target, best):
        best = kdtree_nearest_neighbor(opposite_branch, target, depth + 1, best)

    return best

# partition image into smaller blocks
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

# apply affine transformation
# hindi pa nagagamit sa encoding yung affine transformation 
def apply_affine_transformation(block, transformation):
    scale, rotation, tx, ty = transformation
    h, w = block.shape
    #center_y, center_x = h // 2, w // 2

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
    #print(transformed_block)

    return transformed_block


def find_best_match_kdtree_manual(block, kd_tree, domain_blocks, transformation):
    block_vector = block.flatten()  
    best_vector = kdtree_nearest_neighbor(kd_tree, block_vector)
    best_index = [np.array_equal(best_vector, b.flatten()) for b in domain_blocks].index(True)

    #transformed_block = apply_affine_transformation(domain_blocks[best_index], transformation)
    transformed_block = None

    return best_index, transformation, transformed_block


# encode image with KD-tree
def encode_image_with_kdtree_manual(image, block_size=8):
    range_blocks = partition_image(image, block_size)
    domain_blocks = partition_image(image, block_size)

    # flatten domain blocks and build KD-tree
    start_time = time.perf_counter()
    domain_vectors = [block.flatten() for block in domain_blocks]
    kd_tree = build_kdtree(domain_vectors)
    buildingTree_time = round((time.perf_counter() - start_time) * 1000, 4)

    encoded_data = []
    total_search_time = 0

    # example transformation (rotation, scaling, translation)
    transformation = (1.0, 0.0, 1, 1)  # no scaling, no rotation, and translation by (1, 1)

    with tqdm(total=len(range_blocks), desc="Encoding Image", unit="block", colour="red") as pbar:
        for block in range_blocks:
            search_start = time.perf_counter()
            best_index, _ , transformed_block = find_best_match_kdtree_manual(block, kd_tree, domain_blocks, transformation)
            total_search_time += time.perf_counter() - search_start
            encoded_data.append((best_index, transformation))  # store best index and transformation
            pbar.update(1)

    
    nearestSearch_time = round((total_search_time / len(range_blocks)) * 1000, 4)

    return encoded_data, domain_blocks, buildingTree_time, nearestSearch_time



# decode the image
def decode_image(encoded_data, domain_blocks, image_shape, block_size=8, output_file=None, output_path='data/compressed/fractal'):
    os.makedirs(output_path, exist_ok=True)
    reconstructed_image = np.zeros(image_shape, dtype=np.float64)
    h, w = image_shape
    idx = 0

    # decode the data from the raw binary format and include transformation info
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

    # save the reconstructed image
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imsave(output_file, reconstructed_image)


# Function to compress and evaluate images in a folder using fractal compression
def run_kd_only_compression(original_path, output_path, limit, block_size=8):
    output_path = "data/compressed/test"
    method = "kd-tree-only"
    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"\n\nCompressing {limit} image/s in '{original_path}' using kd-tree only compression...")

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
        encoded_data, domain_blocks, buildingTree_time, nearestSearch_time = encode_image_with_kdtree_manual(image, block_size)
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
                buildingTree_time, nearestSearch_time, 0, encodingTime, decodingTime, 0, "singleRun_CSV.csv"
            )
        processed_count += 1

    # Plot the metrics after compression
    plot_compression_metrics(compression_data)
        
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






"""# function to compress and evaluate images in a folder using fractal compression
def (original_path, output_path, limit, block_size=8):
    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    image_files = image_files[:limit]
    print(f"Comprerun_kd_only_compressionssing {limit} image/s in '{original_path}' using kd-tree only fractal compression...")

    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(original_path, image_file)
        compressed_file = f"compressed_{os.path.splitext(image_file)[0]}.jpg"
        output_file = os.path.join(output_path, compressed_file)
        print(f"[Image {idx}/{limit}] Processing {image_file}...")

        image = load_image(image_path)

        start_time = time.perf_counter()
        encoded_data, domain_blocks, buildingTree_time, nearestSearch_time = encode_image_with_kdtree_manual(image, block_size)
        end_time = time.perf_counter()
        encodingTime = round((end_time-start_time), 4)

        start_time = time.perf_counter()
        decode_image(encoded_data, domain_blocks, image.shape, block_size, output_file=output_file, output_path=output_path)
        end_time = time.perf_counter()
        decodingTime = round((end_time-start_time), 4)

        multiRun_csv(
                method, testRuns, 
                image, image_path, output_file, image_file, compressed_file,
                buildingTree_time, nearestSearch_time, 0, encodingTime, decodingTime, 0, "multiTest_CSV.csv"
            )

    print(f"***Finished compressing {limit} image/s***")
    sys.exit(1)"""



"""# multi testing function
def run_kd_only_compression(original_path, output_path, limit, block_size=8):
    glioma_original_path = "data/dataset/glioma"
    pituitary_original_path = "data/dataset/pituitary"
    output_path = "data/compressed/multiTest"
    print(f"Compressing all glioma and pituitary images using kd-tree only fractal compression...")

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
    method = "kd-tree-only"
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
            encoded_data, domain_blocks, buildingTree_time, nearestSearch_time = encode_image_with_kdtree_manual(image, block_size)
            end_time = time.perf_counter()
            encodingTime = round((end_time - start_time), 4)

            start_time = time.perf_counter()
            decode_image(encoded_data, domain_blocks, image.shape, block_size, output_file=output_file, output_path=output_path)
            end_time = time.perf_counter()
            decodingTime = round((end_time - start_time), 4)

            multiRun_csv(
                method, testRuns, 
                image, image_path, output_file, image_file, compressed_file,
                buildingTree_time, nearestSearch_time, 0, encodingTime, decodingTime, 0, "multiTest_CSV.csv"
            )

    print(f"\n*** Finished all {total_runs} runs for {len(selected_images)} images ***")
    sys.exit(1)"""


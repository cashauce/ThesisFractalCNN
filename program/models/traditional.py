import sys
import os
import numpy as np
import time
from skimage import io, transform, img_as_ubyte
from skimage.transform import AffineTransform, warp
from skimage.exposure import rescale_intensity, is_low_contrast
from skimage.io import imsave
from program.util import multiRun_csv, evaluate_compression
from tqdm import tqdm
import zlib
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline


def load_image(file_path, target_size=(256, 256)):
    image = io.imread(file_path, as_gray=True)
    image = transform.resize(image, target_size, anti_aliasing=True) 
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) 
    return image.astype(np.float32)


# partition image into smaller blocks
def partition_image(image, block_size):
    h, w = image.shape[:2]
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape[:2] != (block_size, block_size):
                block = transform.resize(block, (block_size, block_size), mode='reflect', anti_aliasing=True)
            blocks.append(block)
    return np.array(blocks)


# apply affine transformation using optimized method
def apply_affine_transformation(block, transformation):
    scale, rotation, tx, ty = transformation
    transform = AffineTransform(scale=(scale, scale), rotation=rotation, translation=(tx, ty))
    return warp(block, transform.inverse, mode='reflect', preserve_range=True)


# adjust mean and variance using vectorized operations
def adjust_mean_variance(block, target_mean, target_var):
    mean, var = np.mean(block), np.var(block)
    scale = np.sqrt(target_var / (var + 1e-8)) if var > 0 else 0
    return np.clip(scale * (block - mean) + target_mean, 0, 1)


# find the best match between blocks
def self_similarity_search(block, domain_blocks):
    min_error = float('inf')
    best_block = None
    best_transformation = None

    block_mean, block_var = np.mean(block), np.var(block)

    for domain_block in domain_blocks:
        for scale in [1.0]:  # fixed scale to 1.0
            for rotation in [0, np.pi/2, np.pi]:  # rotations
                for tx, ty in [(0, 0), (1, 1)]:  # translations
                    transformation = (scale, rotation, tx, ty)
                    transformed_block = apply_affine_transformation(domain_block, transformation)
                    
                    transformed_block = adjust_mean_variance(transformed_block, block_mean, block_var)
                    
                    error = np.sum((block - transformed_block) ** 2)
                    if error < min_error:
                        min_error = error
                        best_block = transformed_block
                        best_transformation = transformation
    return best_block, best_transformation


# convert affine transformations to compressed representation
def affine_transformation(transformations):
    transformed_str = ','.join(map(str, transformations))
    return zlib.compress(transformed_str.encode())


# encode a single block 
def encode_block(args):
    block, domain_blocks = args
    return self_similarity_search(block, domain_blocks)


# encode image
def encode_image(image, block_size=8):
    range_blocks = partition_image(image, block_size)
    domain_blocks = partition_image(image, block_size)

    encoded_data = []

    start_time = time.perf_counter()
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(encode_block, [(block, domain_blocks) for block in range_blocks]), 
                            total=len(range_blocks), desc="Encoding Image", unit="block", colour="red"))
        
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    bps = len(range_blocks) / elapsed_time if elapsed_time > 0 else 0
    bps = round((bps), 4)

    for best_block, transformation in results:
        compressed_transformation = affine_transformation(transformation)
        encoded_data.append((best_block, compressed_transformation))

    return encoded_data, elapsed_time, bps


# decode the image
def decode_image(encoded_data, image_shape, block_size=8, output_file=None, output_path='data/compressed/fractal'):
    os.makedirs(output_path, exist_ok=True)
    reconstructed_image = np.zeros(image_shape, dtype=np.float64)
    h, w = image_shape
    idx = 0

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if idx < len(encoded_data):
                best_block, _ = encoded_data[idx] 
                reconstructed_image[i:i+block_size, j:j+block_size] = best_block
                idx += 1

    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    if is_low_contrast(reconstructed_image):
        reconstructed_image = rescale_intensity(reconstructed_image, in_range='image', out_range=(0, 1))

    reconstructed_image = img_as_ubyte(reconstructed_image)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imsave(output_file, reconstructed_image)


# Function to compress and evaluate images in a folder
def run_traditional_compression(original_path, output_path, limit, block_size=8):
    method = "traditional"
    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    print(f"Compressing {limit} image(s) in '{original_path}' using fractal compression...")

    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists

    processed_count = 0  # Count of newly compressed images
    compression_data = []  # List to store compression metrics for plotting
    
    for image_file in image_files:
        if processed_count >= limit:
            break  # Stop when we have compressed 'limit' new images

        base_filename = f"compressed_{os.path.splitext(image_file)[0]}.jpg"
        output_file = os.path.join(output_path, base_filename)

        # Check if the file already exists and generate a new filename with a number
        counter = 2
        while os.path.exists(output_file):
            base_filename = f"compressed_{os.path.splitext(image_file)[0]}_{counter}.jpg"
            output_file = os.path.join(output_path, base_filename)
            counter += 1

        print(f"[Processing {processed_count+1}/{limit}] {image_file}...")
        image_path = os.path.join(original_path, image_file)
        image = load_image(image_path)

        start_time = time.perf_counter()
        encoded_data, nearestSearch_time, bps = encode_image(image, block_size)
        encodingTime = round(time.perf_counter() - start_time, 4)

        start_time = time.perf_counter()
        decode_image(encoded_data, image.shape, block_size, output_file=output_file, output_path=output_path)
        decodingTime = round(time.perf_counter() - start_time, 4)

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
                0, nearestSearch_time, 0, encodingTime, decodingTime, 0, "singleRun_CSV.csv"
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






"""# multi testing function
def run_traditional_compression(original_path, output_path, limit, block_size=8):
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
    method = "traditional"
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
            encoded_data, nearestSearch_time, bps = encode_image(image, block_size)
            end_time = time.perf_counter()
            encodingTime = round((end_time - start_time), 4)

            start_time = time.perf_counter()
            decode_image(encoded_data, image.shape, block_size, output_file=output_file, output_path=output_path)
            end_time = time.perf_counter()
            decodingTime = round((end_time - start_time), 4)

            multiRun_csv(
                method, testRuns, 
                image, image_path, output_file, image_file, compressed_file,
                0, nearestSearch_time, 0, encodingTime, decodingTime, 0, "multiTest_CSV.csv"
            )

    print(f"\n*** Finished all {total_runs} runs for {len(selected_images)} images ***")
    sys.exit(1)"""


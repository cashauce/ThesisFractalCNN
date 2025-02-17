import sys
import os
import numpy as np
import time
from skimage import io, transform, img_as_ubyte
from skimage.transform import AffineTransform, warp
from skimage.exposure import rescale_intensity, is_low_contrast
from skimage.io import imsave
from program.util import evaluate_compression, getTime
from tqdm import tqdm
import zlib


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
            # handle incomplete blocks
            if block.shape[:2] != (block_size, block_size):
                block = transform.resize(block, (block_size, block_size), mode='reflect', anti_aliasing=True)
            blocks.append(block)
    return np.array(blocks)


# apply affine transformation
def apply_affine_transformation(block, transformation):
    scale, rotation, tx, ty = transformation
    center_y, center_x = block.shape[0] // 2, block.shape[1] // 2
    transformed_block = np.zeros_like(block)
    y, x = np.indices(block.shape)
    x_shifted = x + tx
    y_shifted = y + ty
    x_rotated = (x_shifted - center_x) * np.cos(rotation) - (y_shifted - center_y) * np.sin(rotation) + center_x
    y_rotated = (x_shifted - center_x) * np.sin(rotation) + (y_shifted - center_y) * np.cos(rotation) + center_y
    # scaling
    x_scaled = x_rotated * scale
    y_scaled = y_rotated * scale
    
    x_scaled = np.clip(x_scaled, 0, block.shape[1] - 1).astype(np.int32)
    y_scaled = np.clip(y_scaled, 0, block.shape[0] - 1).astype(np.int32)

    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            transformed_block[i, j] = block[y_scaled[i, j], x_scaled[i, j]]

    return transformed_block


def adjust_mean_variance(block, target_mean, target_var):
    mean, var = np.mean(block), np.var(block)
    if var > 0:
        adjusted_block = ((block - mean) * np.sqrt(target_var / (var + 1e-8)) + target_mean).clip(0, 1)
    else:
        adjusted_block = np.full_like(block, target_mean)
    return adjusted_block


# find the best match between blocks 
def self_similarity_search(block, domain_blocks):
    min_error = float('inf')
    best_block = None
    best_transformation = None

    block_mean, block_var = np.mean(block), np.var(block)
    
    for domain_block in domain_blocks:
        for scale in [1.0, 1.2, 0.8]:  # scales
            for rotation in [0, np.pi/8, -np.pi/8]:  # rotations
                for tx, ty in [(0, 0), (1, 1), (-1, -1)]:  # translations
                    transformation = (scale, rotation, tx, ty)
                    transformed_block = apply_affine_transformation(domain_block, transformation)
                    
                    transformed_block = adjust_mean_variance(transformed_block, block_mean, block_var)
                    
                    error = np.sum((block - transformed_block) ** 2)
                    if error < min_error:
                        min_error = error
                        best_block = transformed_block
                        best_transformation = transformation
    return best_block, best_transformation


def affine_transformation(transformations):
    transformed_str = ','.join(map(str, transformations)) 
    compressed = zlib.compress(transformed_str.encode()) 
    return compressed


# encode image 
def encode_image(image, block_size=8):
    range_blocks = partition_image(image, block_size)
    domain_blocks = partition_image(image, block_size)  # using same size for domain blocks

    encoded_data = []
    
    with tqdm(total=len(range_blocks), desc="Encoding Image", unit="block", colour="red") as pbar:
        for block in range_blocks:
            best_block, transformation = self_similarity_search(block, domain_blocks)
            
            compressed_transformation = affine_transformation(transformation)
            encoded_data.append((best_block, compressed_transformation))
            pbar.update(1)
    return encoded_data


# decode the image
def decode_image(encoded_data, image_shape, block_size=8, output_file=None, output_path='data/compressed/fractal'):
    os.makedirs(output_path, exist_ok=True)
    reconstructed_image = np.zeros(image_shape, dtype=np.float64)
    h, w = image_shape
    idx = 0
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if idx < len(encoded_data):
                best_block, compressed_transformation = encoded_data[idx]
                reconstructed_image[i:i+block_size, j:j+block_size] = best_block
                idx += 1

    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    if is_low_contrast(reconstructed_image):
        reconstructed_image = rescale_intensity(reconstructed_image, in_range='image', out_range=(0, 1)) 
    
    reconstructed_image = img_as_ubyte(reconstructed_image)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imsave(output_file, reconstructed_image)


# Function to compress and evaluate images in a folder using traditional fractal compression
def run_traditional_compression(original_path, output_path, limit, block_size=8):
    image_files = sorted([f for f in os.listdir(original_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
    image_files = image_files[:limit]
    print(f"Compressing {limit} image/s in '{original_path}' using fractal compression...")

    for idx, image_file in enumerate(image_files, start=1):
        image_path = os.path.join(original_path, image_file)
        output_file = os.path.join(output_path, f"compressed_{os.path.splitext(image_file)[0]}.jpg")
        print(f"[Image {idx}/{limit}] Processing {image_file}...")

        image = load_image(image_path)

        start_time = time.time()
        encoded_data = encode_image(image, block_size)
        end_time = time.time()
        total_time = getTime(end_time-start_time)
        print(f"Encoding time: {total_time}")

        start_time = time.time()
        decode_image(encoded_data, image.shape, block_size, output_file=output_file, output_path=output_path)
        end_time = time.time()
        total_time = getTime(end_time-start_time)
        print(f"Decoding time: {total_time}")

        evaluate_compression(image, image_path, output_file)

    print(f"***Finished compressing {limit} image/s***")
    sys.exit(1)

import os
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# pang evaluate lang to para makita yung mga metrics. hindi to yung final na gagamitin
def evaluate_compression(image, original_image_path, compressed_image_path):
    original_image = image
    compressed_image = io.imread(compressed_image_path, as_gray=True) / 255.0
    original_image = np.clip(original_image, 0, 1)
    compressed_image = np.clip(compressed_image, 0, 1)

    # PSNR and SSIM
    PSNR = psnr(original_image, compressed_image, data_range=1.0)
    print(f"\tPeak Signal-to-Noise Ratio (PSNR): {PSNR:.2f} dB")

    SSIM = ssim(original_image, compressed_image, data_range=1.0)
    print(f"\tStructural Similarity Index (SSIM): {SSIM:.4f}")

    # file sizes
    original_size = os.path.getsize(original_image_path)
    compressed_size = os.path.getsize(compressed_image_path) 

    # compression ratio
    if compressed_size != 0:
        CR = original_size / compressed_size
        CR_ratio = f"1:{original_size // compressed_size}"
    else:
        CR = float('inf')
        CR_ratio = "Infinity:1"

    print(f"\tCompression Ratio: {CR:.2f} ({CR_ratio})")
    print(f"\t\tOriginal File Size: {original_size} bytes")
    print(f"\t\tCompressed File Size: {compressed_size} bytes")


def getTime(total_time):
    if total_time < 60:
        return f"{total_time:.4f} seconds"
    elif total_time < 3600:
        minutes = total_time // 60
        seconds = total_time % 60
        return f"{int(minutes)} minutes and {seconds:.4f} seconds"
    else:
        hours = total_time // 3600
        remaining_time = total_time % 3600
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        return f"{int(hours)} hours, {int(minutes)} minutes, and {seconds:.4f} seconds"


#get = getTime(10000)
#print(get)
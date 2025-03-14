import csv
import os
import numpy as np
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim


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
    

# pang evaluate lang to para makita yung mga metrics. hindi to yung final na gagamitin
def evaluate_compression(image, original_image_path, compressed_image_path):
    original_image = image
    compressed_image = io.imread(compressed_image_path, as_gray=True) / 255.0
    original_image = np.clip(original_image, 0, 1)
    compressed_image = np.clip(compressed_image, 0, 1)

    # PSNR and SSIM
    PSNR = round(psnr(original_image, compressed_image, data_range=1.0), 4)
    #print(f"\tPeak Signal-to-Noise Ratio (PSNR): {PSNR:.2f} dB")

    SSIM = round(ssim(original_image, compressed_image, data_range=1.0), 4)
    #print(f"\tStructural Similarity Index (SSIM): {SSIM:.4f}")

    # file sizes
    original_size = os.path.getsize(original_image_path)
    compressed_size = os.path.getsize(compressed_image_path) 

    # compression ratio
    if compressed_size != 0:
        #CR = original_size / compressed_size
        CR_ratio = f"1:{original_size // compressed_size}"
    else:
        #CR = float('inf')
        CR_ratio = "Infinity:1"

    original_size = round(original_size / 1024, 2)  
    compressed_size = round(compressed_size / 1024, 2)  

    return original_size, compressed_size, CR_ratio, PSNR, SSIM


def save_csv(image, original_image_path, compressed_image_path, original_image, compressed_image, encodingTime, decodingTime, bps,  data_folder):
    os.makedirs(data_folder+"/csv", exist_ok=True)

    csv_filename = os.path.join(data_folder+"/csv", "compression_metrics.csv")

    # Check if the image already exists in the CSV file
    if os.path.isfile(csv_filename):
        with open(csv_filename, mode="r", newline="") as file:
            reader = csv.reader(file)
            existing_images = {row[0] for row in reader}  # Collect existing original image names

        if original_image in existing_images:
            print(f"Skipping {original_image}, already recorded.")
            return  # Skip saving if the image is already recorded

    # evaluate metrics
    original_size, compressed_size, cr, psnr, ssim = evaluate_compression(image, original_image_path, compressed_image_path)

    # prepare data for CSV
    data = [original_image, original_size, compressed_image, compressed_size, cr, encodingTime, decodingTime, psnr, ssim, bps]

    # Write to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["originalImage", "originalImage_size (KB)", "compressedImage", "compressedImage_size (KB)",
                             "compressionRatio", "encodingTime (s)", "decodingTime (s)", "PSNR (dB)", "SSIM", "blocks (blocks/s)"])

        # Write the data row
        writer.writerow(data)

    print(f"Metrics saved to {csv_filename}")
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


def cnn_metrics_csv(epoch, batch, trainingLoss, trainingTime, csvFile_name):
    os.makedirs("data/csv", exist_ok=True)
    csv_filename = os.path.join("data/csv", csvFile_name)
    data = [epoch, batch, trainingLoss, trainingTime]
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["Epoch", "Batch", "Training Loss", "Training Time (s)"])

        # Write the data row
        writer.writerow(data)


def compression_traditional_csv(image, original_image_path, compressed_image_path, original_image, compressed_image, encodingTime, decodingTime, bps,  csvFile_name):
    os.makedirs("data/csv", exist_ok=True)
    csv_filename = os.path.join("data/csv", csvFile_name)

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
    data = [original_image, original_size, original_image_path, 
            compressed_image, compressed_size, compressed_image_path, 
            cr, encodingTime, decodingTime, psnr, ssim, bps]

    # Write to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["Original Image", "Original Image Size (KB)", "Original Image Path",
                             "Compressed Image", "Compressed Image Size (KB)", "Compressed Image Path", 
                             "Compression Ratio", "Encoding Time (s)", "Decoding Time (s)", "PSNR (dB)", "SSIM", "Blocks (blocks/s)"])

        # Write the data row
        writer.writerow(data)

    print(f"Metrics saved to {csv_filename}")


def compression_hybrid_csv(image, original_image_path, compressed_image_path, original_image, compressed_image, 
                             buildingTree_time, nearestSearch_time, inference_time, encodingTime, decodingTime, bps,  csvFile_name):
    os.makedirs("data/csv", exist_ok=True)

    csv_filename = os.path.join("data/csv", csvFile_name)

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
    data = [original_image, original_size, original_image_path, 
            compressed_image, compressed_size, compressed_image_path, cr, 
            buildingTree_time, nearestSearch_time, inference_time, encodingTime, decodingTime, 
            psnr, ssim, bps]

    # Write to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writerow(["Original Image", "Original Image Size (KB)", "Original Image Path",
                             "Compressed Image", "Compressed Image Size (KB)", "Compressed Image Path", "Compression Ratio", 
                             "Build Tree Time (ms)", "Nearest Search Time (ms)", "CNN Inference Time (ms)", "Encoding Time (s)", "Decoding Time (s)", 
                             "PSNR (dB)", "SSIM", "Blocks (blocks/s)"])

        # Write the data row
        writer.writerow(data)

    print(f"Metrics saved to {csv_filename}")
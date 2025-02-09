import os
import numpy as np
from skimage import io, transform, img_as_ubyte
from skimage.util import random_noise
from skimage.exposure import adjust_gamma
from skimage.transform import rotate
from tqdm import tqdm

# function to preprocess a single image with augmentation
def preprocess_image(file_path, target_size=(256, 256), augmentations=[]):
    """Load an image, resize it to target size, normalize pixel values, and apply augmentations."""
    image = io.imread(file_path, as_gray=True)  # load image in grayscale
    image_resized = transform.resize(image, target_size, mode='reflect', anti_aliasing=True)  # resize
    image_normalized = (image_resized - np.min(image_resized)) / (np.max(image_resized) - np.min(image_resized))  # normalize to [0, 1]
    
    # augmentations
    augmented_images = {}
    for name, aug in augmentations:
        augmented_image = aug(image_normalized)
        augmented_images[name] = augmented_image

    return {"original": image_normalized, **augmented_images}  # include the original preprocessed image


def add_noise(image):
    """Add random noise to the image."""
    return random_noise(image, mode='gaussian', var=0.01)

def adjust_brightness(image, gamma=0.8):
    """Adjust the brightness of the image."""
    return adjust_gamma(image, gamma=gamma)

def rotate_image(image, angle=15):
    """Rotate the image by a specific angle."""
    return rotate(image, angle, mode='reflect', resize=False)

def flip_image(image):
    """Flip the image horizontally."""
    return np.fliplr(image)

# function to preprocess and augment the first N images
def preprocess_images(input_folder, output_folder, limit, target_size=(256, 256)):
    """Preprocess and augment the first N images in a dataset, saving them to an output folder."""
    os.makedirs(output_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(input_folder))

    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])[:limit]
    print(f"Found {len(image_files)} images in '{input_folder}'. Preprocessing and augmenting the first {limit} images.")

    augmentations = [
        ("noise", add_noise),
        ("brightness", lambda img: adjust_brightness(img, gamma=1.2)),
        ("flipped", flip_image),
        ("rotated", lambda img: rotate_image(img, angle=15))
    ]
    
    with tqdm(total=len(image_files) * (len(augmentations) + 1), desc="Processing Images", unit="image", colour="blue") as pbar:
        for idx, image_file in enumerate(image_files):
            input_path = os.path.join(input_folder, image_file)
            base_name, ext = os.path.splitext(image_file)

            processed_images = preprocess_image(input_path, target_size, augmentations)

            # save the original and augmented images
            for aug_name, processed_image in processed_images.items():
                suffix = f"{aug_name}"
                output_path = os.path.join(output_folder, f"{base_name}_{suffix}_{folder_name}{ext}")
                io.imsave(output_path, img_as_ubyte(processed_image)) 
                pbar.update(1)

    print(f"Preprocessing and augmentation completed. Processed images saved to '{output_folder}'.")
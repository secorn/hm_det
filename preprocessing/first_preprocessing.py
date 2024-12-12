import pydicom as dicom
import matplotlib.pyplot as plt
import os
import shutil
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import pandas as pd
import random

### Important ###
# First update the variables in .env to fit your local directory structure #

# Load required variables from the .env file #
load_dotenv()
output_path = os.getenv("output_path")
img_path = os.getenv("img_path")


### Helper Functions ###

def transform_in_hu(img):
    """
    From the .dcm file, the required information (slope + intercept) is extracted
    and then applied to transform the raw pixel values into Hounsefield Units (standard scale in CT images)
    """
    #Check if RescaleSlope and RescaleIntercept exist in the DICOM metadata
    if hasattr(img, 'RescaleSlope') and hasattr(img, 'RescaleIntercept'):
        slope = img.RescaleSlope
        intercept = img.RescaleIntercept
    else:
        slope = 1 # Default value if not present
        intercept = 0  # Default value if not present

    pixel_array = img.pixel_array
    img_hu = pixel_array * slope + intercept
    return img_hu

def apply_window(img, window_center, window_width):
    """
    Apply a window to highlight areas of the CT scan; here: brain area
    """
    lower_bound = window_center - (window_width / 2)
    upper_bound = window_center + (window_width / 2)
    img = np.clip(img, lower_bound, upper_bound)
    return img

def normalizer(img):
    """
    Apply Min-Max-Scaling to normalize the image pixel values
    """
    minimum = np.min(img)
    maximum = np.max(img)
    img = (img - minimum ) / (maximum - minimum)
    return img

### Finally, Process the images ###

if __name__ == "__main__":

    # Ensure the output directory exists
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    # Create the directory fresh
    os.makedirs(output_path)

    def process_images(k=100, window_center=40, window_width=80):
        """
        Draws a random sample of size k (default=100) from all .dcm images
        Then applies all necessary pre-processing steps 1) HU conversion 2) Application of brain
        window 3) Normalization. Finally, saves processed images as PNG to the <output_path> directory
        """

        all_files = [f for f in os.listdir(img_path) if f.endswith('.dcm')]
        selected_files = random.sample(all_files, min(k, len(all_files)))

        for filename in selected_files:

            # Check if the file is a DICOM file (assuming .dcm extension)
            image_path = os.path.join(img_path, filename)  # Full file path
            img = dicom.dcmread(image_path)  # Reads the DICOM file

            img_hu = transform_in_hu(img)
            img_window = apply_window(img_hu, window_center, window_width)
            img_norm = normalizer(img_window) * 255 # bring pixels to 0-255 range (necessary for PNG)
            img_norm = img_norm.astype(np.uint8)  # Convert to 8-bit unsigned integer

            # Save as PNG
            output_file = os.path.join(output_path, filename.replace('.dcm', '.png'))

            # Ensure the file ends with .png
            if not output_file.endswith('.png'):
                output_file += '.png'  # Add .png if it's missing

            img_pil = Image.fromarray(img_norm)
            img_pil.save(output_file)

        return f"✅ {k} Images were processed, converted to .png and saved in {output_path}"

    process_images(k = 500)

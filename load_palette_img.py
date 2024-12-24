from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image_with_grayscale_indices(image_path, palette=None):
    """
    Loads an image and converts it to grayscale indices.
    If the image is in "P" (palette) mode, it extracts indices based on the palette.
    If the image is in grayscale mode ("L"), it directly loads the pixel values.

    Args:
        image_path (str): Path to the image file.
        palette (np.ndarray, optional): A 2D numpy array where each row represents
                                         an RGB color in the palette (required for palette images).

    Returns:
        np.ndarray: A 2D numpy array where each pixel value corresponds to grayscale indices.
    """
    # Open the image
    image = Image.open(image_path)
    
    if image.mode == 'P':
        # Ensure a palette is provided for palette-based images
        if palette is None:
            raise ValueError("Palette is required for 'P' (palette) mode images.")
        # Convert the image to a numpy array of indices
        grayscale_indices = np.array(image)
    elif image.mode == 'L':
        # For grayscale images, directly return the pixel values
        grayscale_indices = np.array(image)
    else:
        raise ValueError("The input image must be in 'P' (palette) mode or 'L' (grayscale) mode.")
    
    return grayscale_indices

# Specify palette (same as defined in Cutie/cutie/utils/palette.py)
palette = np.array([
    [  0,   0,   0],            # Background         (No color)
    [  0,   0, 255],            # Azygos             (Blue)   
    [255,   0,   0],            # Aorta              (Red)
    [160, 100, 160],            # Lung               (Pink)
    [255, 160,   0],            # Esophagus          (Orange)
    [255,   0, 157],            # Pericardium        (Purple)
    [255, 255, 255],            # Airways            (White)
    [255, 255,   0],            # Nerves             (Yellow)
    [100, 80,   0],            # Hook               (Orange)
    [128,   0,   0],            # Forceps            (Red)
    [  0, 128,   0],            # Suction/irrigation (Green)
    [  0, 255, 255],            # Vessel sealer      (Cyan)
    [  0, 255,   0]             # Thoracic duct      (Green)
])

# Specify image path
image_path = r'C:\Users\20182054\Documents\Cutie\workspace\P0003video2_00-06-37_0009923\masks\P0003video2_00-06-30_0009738.png' # Palette image
image_path = r'C:\Users\20182054\Documents\Cutie\data\RAMIE_15s\P0003video2_00-10-25_0015614\masks\P0003video2_00-10-25_0015614.png' # Grayscale image

# Load image using function
image = load_image_with_grayscale_indices(image_path, palette)

# Display the image in grayscale
plt.imshow(image, cmap='gray')

# Print unique class values in image
print(np.unique(image))
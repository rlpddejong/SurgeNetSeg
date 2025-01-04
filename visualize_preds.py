import cv2
import os
import numpy as np

propagated_mask_path = './data/RAMIE/propagated_masks/'


folders = [folder for folder in os.listdir(propagated_mask_path) if os.path.isdir(os.path.join(propagated_mask_path, folder))]

def overlay_mask_with_multiple_classes(image, mask, color_map, alpha_map):
    """
    Overlays a multi-class mask on the image with different transparencies for each class.
    
    Parameters:
    - image (np.array): The original image (height x width x 3)
    - mask (np.array): The segmentation mask (height x width). Values represent class labels.
    - color_map (dict): Dictionary mapping class label to RGB color.
    - alpha_map (dict): Dictionary mapping class label to alpha transparency level (0 to 1).
    
    Returns:
    - result (np.array): The image with the overlaid multi-class mask.
    """
    
    # Ensure image and mask are numpy arrays
    image = np.array(image)
    mask = np.array(mask)

    # Create a copy of the image to start with (so we don't modify the original)
    overlay = image.copy()

    # Iterate through all unique class labels in the mask
    for class_label in np.unique(mask):
        # Skip background class (usually class 0)
        if class_label in [0]: # [0,4,8,9,10,11,12]:
            continue
        
        # Get the color and transparency for this class
        color = color_map.get(class_label, (255, 255, 255))  # Default to white if not specified
        alpha = alpha_map.get(class_label, 0.5)  # Default to 50% transparency if not specified

        # Create a mask for the current class
        class_mask = (mask == class_label)
        
        # Create an image of the same size as the input image, colored for this class
        class_overlay = np.zeros_like(image)
        class_overlay[class_mask] = color
        
        # Blend the class overlay with the image using the specified alpha value
        # Adjust alpha blending formula: image = alpha * class_overlay + (1-alpha) * original image
        overlay[class_mask] = cv2.addWeighted(overlay[class_mask], 1 - alpha, class_overlay[class_mask], alpha, 0)

    return overlay

# Color map (class label to color)
color_map = {
    1:  (255, 0, 0),           # Azygos (Blue)
    2:  (0, 0, 255),           # Aorta (Red)
    3:  (255, 192, 203),       # Lung (Pink)
    4:  (0, 140, 255),         # Esophagus (Orange)
    5:  (157,0,255),           # Pericardium (Purple)
    6:  (255, 255, 255),       # Airways (White)
    7:  (0, 255, 255),         # Nerves (Yellow)
    8:  (0, 166, 255),           # Hook
    9:  (0, 0, 128),            # Forceps
    10: (0, 128, 0),            # Suction / irrigation
    11: (255, 255, 0),              # Vessel sealer
    12: (0, 255, 0)           # Thoracic duct (Green)
}

# Alpha map (class label to transparency)
alpha_map = {
    1:  0.4,                    # Azygos (Blue) 
    2:  0.5,                    # Aorta (Red)
    3:  0.6,                    # Lung (Pink)
    4:  0.3,                    # Esophagus (Orange)   
    5:  0.5,                    # Pericardium (Purple)
    6:  0.5,                    # Airways (White)
    7:  0.3,                    # Nerves (Yellow)
    8:  0.5,                    # Hook
    9:  0.7,                    # Forceps
    10: 0.5,                    # Suction / irrigation
    11: 0.5,                    # Vessel sealer
    12: 0.5,                    # Thoracic duct (Green)
}

for folder in folders:

    # Paths to images and masks
    images_path = os.path.join("./data/RAMIE/images/", folder)
    masks_path = os.path.join("./data/RAMIE/propagated_masks/",folder)
    output_video = os.path.join("./data/RAMIE/propagated_videos/",folder+'.mp4')

    if not os.path.exists(output_video):

        print(images_path)
        print(masks_path)
        print(output_video)

        # List all images and masks (assuming filenames match)
        image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
        mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith('.png')])

        # Check that the number of images matches the number of masks
        if len(image_files) != len(mask_files):
            raise ValueError("Number of images and masks do not match!")

        # Initialize video writer
        sample_image = cv2.imread(os.path.join(images_path, image_files[0]))
        height, width, _ = sample_image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        fps = 25  # Frames per second
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Overlay masks on images and write to video
        for img_file, mask_file in zip(image_files, mask_files):
            # Read the image and mask
            img = cv2.imread(os.path.join(images_path, img_file))
            mask = cv2.imread(os.path.join(masks_path, mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Resize mask to match image dimensions (if necessary)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
            overlay = overlay_mask_with_multiple_classes(img, mask, color_map, alpha_map)

            # Write the frame to the video
            video_writer.write(overlay)
            #video_writer.write(img) # TEMP!

        # Release the video writer
        video_writer.release()
        print(f"Video saved to {output_video}")

import os
import shutil

# Define paths
in_path = r'C:\Users\20182054\Documents\Cutie_data\RAMIE_15s'
save_path = r'G:\datasets\RAMIE_879_video'

# Get all folder names in the input path
folder_names = [folder_name for folder_name in os.listdir(in_path)]
folder_names.sort()

for folder_name in folder_names:

    # Define image and mask paths
    img_path = os.path.join(in_path, folder_name, 'images')
    mask_path = os.path.join(in_path, folder_name, 'masks')

    # Count the number of png files in the mask folder
    file_names = [f for f in os.listdir(mask_path) if f.endswith('.png')]
    
    # Conditiopn used to only grab annotated files
    if len(file_names) == 375:

        # Define new image and mask paths
        new_img_folder_path = os.path.join(save_path, 'images', folder_name)
        new_mask_folder_path = os.path.join(save_path, 'masks', folder_name)

        # Create new directories for images and masks
        os.makedirs(new_img_folder_path, exist_ok=True)
        os.makedirs(new_mask_folder_path, exist_ok=True)

        for file_name in file_names:
            
            # Copy image and mask files to new directories
            shutil.copy(os.path.join(img_path, file_name), os.path.join(new_img_folder_path, file_name))
            shutil.copy(os.path.join(mask_path, file_name), os.path.join(new_mask_folder_path, file_name))
import os
import cv2
import numpy as np
import pickle

# USER NOTE: make these pickle files in the same version as pickle on server where training happens 
# to avoid compatibility issues (often numpy errors like missing numpy.core module)

# Validation
mask_folder_path = r'G:\datasets\RAMIE_879_RITM\Validation\masks'
pickle_file_path = r'G:\datasets\RAMIE_879_RITM\Validation\hannotation.pickle'

# # Train
# mask_folder_path = r'G:\datasets\RAMIE_879_RITM\Train\masks'
# pickle_file_path = r'G:\datasets\RAMIE_879_RITM\Train\hannotation.pickle'

# # Test
# mask_folder_path = r'G:\datasets\RAMIE_879_RITM\Test\masks'
# pickle_file_path = r'G:\datasets\RAMIE_879_RITM\Test\hannotation.pickle'

# Function to get unique values in a grayscale image
def get_unique_values(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Get unique values
    unique_values = np.unique(img)
    return unique_values

# Dictionary to hold the data
hlvis_annotation = {}

# Iterate over all PNG images in the folder
for filename in os.listdir(mask_folder_path):
    if filename.endswith(".png"):
        # Remove the extension from the filename
        base_filename = os.path.splitext(filename)[0]
        
        # Get the unique values in the grayscale image
        unique_values = get_unique_values(os.path.join(mask_folder_path, filename))
        
        # Assuming you want to count unique values as "num_instance_masks"
        num_instance_masks = len(unique_values)-1 # skip background by using -1
        
        # Construct the dictionary entry
        hlvis_annotation[base_filename] = {
            'num_instance_masks': num_instance_masks,
            'hierarchy': {}
        }
        
        # For each mask, create a hierarchy for illustration (replace with actual logic if needed)
        for i in range(0, num_instance_masks):
            hlvis_annotation[base_filename]['hierarchy'][i] = {
                'parent': None,
                'children': []
            }

# Save the dictionary as a pickle file
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(hlvis_annotation, pickle_file)

print(f"Data saved as {pickle_file_path}")

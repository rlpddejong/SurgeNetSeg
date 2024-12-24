import os
import cv2
import time
import shutil

def copy_files(source_folder, destination_folder):
    """
    Copy all files from source_folder to destination_folder.
    
    Args:
    - source_folder (str): Path to the source folder.
    - destination_folder (str): Path to the destination folder.
    """
    # Get list of files in source folder
    files = os.listdir(source_folder)

    # Iterate over files and copy them to destination folder
    for file in files:
        source_file_path = os.path.join(source_folder, file)
        

        curr_destination_folder = os.path.join(destination_folder, file.replace(".png", ""), 'masks')
        if not os.path.exists(curr_destination_folder):
            os.makedirs(curr_destination_folder)

        destination_file_path = os.path.join(curr_destination_folder, file)
        shutil.copy(source_file_path, destination_file_path)

# Define paths
data_path = r'G:\datasets\RAMIE_879\RAMIE_879_raw_12classes'
vid_path  = r'G:\RAMIE_videos_cut_crop_25fps\batch1'
save_path = r'C:\Users\20182054\Documents\Cutie_data\RAMIE_15s'

# Specify which frames to extract, e.g. -25 is 25 frames prior to annotated frame
x = 187 #75 # Number of frames prior and after, e.g. choosing 1 yields: [-1,0,1]
relative_frame_nrs = lst = list(range(-x, x + 1))

fps = 25

# Define relative paths
img_path = os.path.join(data_path,'images')
mask_path = os.path.join(data_path,'masks')

#save_img_path = os.path.join(save_path,'images')

# # Create save directories
# if not os.path.exists(save_img_path):
#         os.makedirs(save_img_path)

# Copy original files to new location
#copy_files(img_path,save_img_path)
copy_files(mask_path, save_path)

nr_frames = len(os.listdir(mask_path))

img_names = os.listdir(img_path)

from tqdm import tqdm  # Import tqdm for the progress bar

# Wrap `img_names` with tqdm to show progress
for img_name in tqdm(img_names, desc="Processing Images"):

    save_img_path = os.path.join(save_path, img_name.replace(".png", ""), 'images')
    
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

        # Extract relevant information from image name
        img_name_wo_file_ext = img_name.split('.')[0]
        vid_name = img_name_wo_file_ext.split('_')[0]
        timestamp = img_name_wo_file_ext.split('_')[1]
        frame_nr = img_name_wo_file_ext.split('_')[2]

        # Get video
        current_vid_path = os.path.join(vid_path, vid_name + '.mp4')
        vidcap = cv2.VideoCapture(current_vid_path)

        for relative_frame_nr in relative_frame_nrs:
            # Extract frame from video
            current_frame_nr = int(frame_nr) + int(relative_frame_nr)
            
            # Only extract frame when it is within the video
            if current_frame_nr >= 0:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_nr)
                success, image = vidcap.read() 
                
                # Break if end of video is reached
                if not success:
                    break
                #image = image[1:503, :]

                # Save frame
                seconds = round(current_frame_nr / fps)
                t = time.strftime('%H-%M-%S', time.gmtime(seconds))
                filename = vid_name + '_' + t + '_' + str(current_frame_nr).rjust(7, '0') + '.png'
                savename = os.path.join(save_img_path, filename)
                cv2.imwrite(savename, image)

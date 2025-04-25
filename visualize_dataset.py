import os
import cv2
import numpy as np
from PIL import Image

path = r'C:\Users\20182054\Documents\Cutie_data\RAMIE_15s'

folders = [
    'P0003video2_00-03-12_0004812',
    'P0003video2_00-49-14_0073846',
    'P0003video2_00-26-00_0039002',
    'P0003video2_00-10-25_0015614',
    'P0003video2_00-26-11_0039272',
    'P0003video2_00-01-18_0001959',
    'P0003video2_00-29-28_0044209',
    'P0003video2_00-51-45_0077621',
    'P0003video2_00-06-37_0009923',
    'P0003video2_00-51-03_0076567',
    'P0003video2_00-23-42_0035554',
    'P0005video1_00-40-17_0060418',
    'P0006video3_00-16-38_0024945',
    'P0009video1_02-12-03_0198069',
    'P0010video3_01-05-33_0098330',
    'P0011video4_00-49-09_0073736',
    'P0012video1_01-51-10_0166738',
    'P0013video2_00-05-54_0008852',
    'P0014video4_00-03-19_0004971',
    'P0016video1_00-51-00_0076493',
    'P0017video1_00-06-06_0009151',
    'P0018video1_00-06-42_0010046',
    'P0019video1_00-45-07_0067664',
    'P0020video5_01-06-03_0099068',
    'P0021video1_00-33-29_0050218',
    'P0024video1_00-09-17_0013933',
    'P0026video1_00-07-55_0011864',
    'P0034video1_02-17-30_0206260'
]

color_map = {1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 192, 203), 4: (0, 140, 255), 
             5: (157, 0, 255), 6: (255, 255, 255), 7: (0, 255, 255), 8: (0, 166, 255), 
             9: (0, 0, 128), 10: (0, 128, 0), 11: (255, 255, 0), 12: (0, 255, 0)}

alpha_map = {0: 0.0, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.3, 5: 0.5, 6: 0.5, 7: 0.3, 8: 0.5, 
             9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5}

fps = 25  # Set frame rate

def apply_color_map(pred, frame):
    #pred = pred.astype(int)  # Ensure integer type
    pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        mask = (pred == class_id)
        pred_colored[mask] = color
        alpha = alpha_map.get(class_id, 0.5)
        
        if np.any(mask):
            frame[mask] = cv2.addWeighted(frame[mask], 1 - alpha, pred_colored[mask], alpha, 0)

    return frame

for folder in folders:
    mask_path = os.path.join(path, folder, 'masks')
    image_path = os.path.join(path, folder, 'images')
    output_video_path = os.path.join(path, f'{folder}.mp4')
    
    files = sorted(os.listdir(mask_path))
    
    frame_width, frame_height = None, None
    video_writer = None
    
    for file in files:
        mask_file = os.path.join(mask_path, file)
        img_file = os.path.join(image_path, file)
        
        if not os.path.exists(img_file):
            continue
        
        # Read image and mask
        image = cv2.imread(img_file)
        #mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = Image.open(mask_file)  # Replace with your image file
        mask = mask.convert("P")  # Ensure it is in palette mode
        mask = np.array(mask)

        if image is None or mask is None:
            continue
        
        # Initialize video writer if not already initialized
        if video_writer is None:
            frame_height, frame_width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Convert mask to color
        #mask_colored = cv2.applyColorMap(mask, color_map)
        
        # Overlay mask on image
        #overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)
        
        overlay = apply_color_map(mask, image)

        # Write frame to video
        video_writer.write(overlay)
        
    if video_writer:
        video_writer.release()

print("Videos saved for each folder.")

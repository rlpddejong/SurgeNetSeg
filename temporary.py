import cv2
import numpy as np
import os

folder_path = r'C:\Users\20182054\Documents\Cutie_data\annotated_videos'  # Change this to your folder path
file_paths = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_paths.append(os.path.join(root, file))

video_files = sorted(file_paths)

# Read the first video to get the properties (e.g., width, height, FPS)
cap = cv2.VideoCapture(video_files[0])
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Output video size (6 columns and 4 rows)
tile_width = frame_width * 6
tile_height = frame_height * 4

# Video writer to output the tiled video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(os.path.join(folder_path,"output.mp4"), fourcc, fps, (tile_width, tile_height))

# Open all videos
caps = [cv2.VideoCapture(video) for video in video_files]

while True:
    # Read a frame from each video
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # If any video ends, stop processing
    if len(frames) < 24:
        break

    # Resize each frame to match the target resolution
    resized_frames = [cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC) for frame in frames]

    # Stack the frames into a grid (6x4)
    rows = []
    for i in range(4):  # 4 rows
        row = np.hstack(resized_frames[i*6:(i+1)*6])  # Stack 6 frames horizontally
        rows.append(row)

    # Stack all rows vertically
    tiled_frame = np.vstack(rows)

    # Write the tiled frame to the output video
    out.write(tiled_frame)

# Release all resources
for cap in caps:
    cap.release()
out.release()

print("Tiling complete. Output video saved as 'output.mp4'.")

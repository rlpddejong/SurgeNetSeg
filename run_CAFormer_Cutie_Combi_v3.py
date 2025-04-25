import cv2
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from MetaFormer import MetaFormerFPN
from torchvision import transforms as T
from collections import deque

from torchvision.transforms.functional import to_tensor
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

# Paths
weights_path = r'C:\Users\20182054\OneDrive - TU Eindhoven\PhD\code\intrasurge\runtime_verification_experiment\CAFormerS18_RAMIE_SurgeNet.pth'
video_path = r'C:\Users\20182054\OneDrive - TU Eindhoven\PhD\code\intrasurge\runtime_verification_experiment\clips\P0018video1_clip_15.mp4'
out_folder = r'C:\Users\20182054\OneDrive - TU Eindhoven\PhD\code\intrasurge\runtime_verification_experiment\out'
cutie_weights_path = r'C:\Users\20182054\OneDrive - TU Eindhoven\PhD\code\Cutie_RAMIE\weights\RAMIE_24_IT11600_l133.pth'

# Color map per structure (BGR)
color_map = {1: (0, 0, 255), 2: (255, 0, 0), 3: (203, 192, 255), 4: (255, 140, 0), 
             5: (255, 0, 157), 6: (255, 255, 255), 7: (255, 255, 0), 8: (255, 166, 0), 
             9: (128, 0, 0), 10: (0, 128, 0), 11: (0, 255, 255), 12: (0, 255, 0)}

alpha_map = {0: 0.0, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.3, 5: 0.5, 6: 0.5, 7: 0.3, 8: 0.5, 
             9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5}

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MetaFormerFPN(num_classes=13, pretrained='SurgNet').to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()

# Create a Tkinter window
root = tk.Tk()
root.title("Webcam Video Feed")
w, h = 256,256 #640, 480
root.geometry(f"{w}x{h}")

# Create a label in the Tkinter window
label = tk.Label(root)
label.pack()

# Open the webcam
cap = cv2.VideoCapture(1)

mean, std = torch.tensor([0.4927, 0.2927, 0.2982]), torch.tensor([0.2680, 0.2320, 0.2343])
t_norm = T.Normalize(mean=mean, std=std)

# Preprocess function (moved to a separate thread)
def preprocess_frame(frame):
    image_rgb = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
    image_tensor = T.ToTensor()(image_rgb).unsqueeze(0).to(device)
    return t_norm(image_tensor)

# Predict function
def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        return output.cpu().squeeze().numpy()

# Postprocessing function
def postprocess(pred, frame):
    #pred = np.argmax(pred, axis=0).astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_colored = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        mask = (pred == class_id)
        pred_colored[mask] = color

        alpha = alpha_map.get(class_id, 0.5)
        if np.any(mask):
            frame[mask] = cv2.addWeighted(frame[mask], 1 - alpha, pred_colored[mask], alpha, 0)

    return frame

ti = 0 # TODO: remove this

cutie = get_default_model()
cutie.load_weights(torch.load(cutie_weights_path))
processor = InferenceCore(cutie, cfg=cutie.cfg)
processor.max_internal_size = 480

objects = [1,2,3,4,5,6,7,8,9,10,11,12]

# Circular buffer to store the last 50 pred_caformer_inp
buffer_size = 25
pred_caformer_buffer = deque(maxlen=buffer_size)
image_buffer = deque(maxlen=buffer_size)

@torch.inference_mode()
@torch.cuda.amp.autocast()
# Function to update the frame in the Tkinter window
def update():
    global ti

    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)

        # Create fresh copies of the frame for each prediction
        frame_caformer = np.copy(frame)
        frame_cutie = np.copy(frame)
        frame_average = np.copy(frame)

        # Preprocessing
        image_tensor = preprocess_frame(frame)

        # MetaFormer Prediction
        pred_caformer = model(image_tensor)
        pred_caformer_logits = pred_caformer.squeeze()
        pred_caformer_inp = pred_caformer_logits.argmax(dim=0)
        pred_caformer = pred_caformer_inp.cpu().numpy()

        # Postprocessing for MetaFormer
        out_caformer = postprocess(pred_caformer, frame_caformer)

        # CUTIE Prediction
        print(ti)
        image = to_tensor(frame).cuda().float()

        # Add the current prediction to the buffer (circular)
        pred_caformer_buffer.append(pred_caformer_inp)
        image_buffer.append(image)

        total_pred_cutie_logits = torch.zeros((13,256,256)).cuda()

        if len(pred_caformer_buffer) == buffer_size:
            for i in range(10):
                pred_caformer_buf = pred_caformer_buffer[i]
                image_buf = image_buffer[i].squeeze(0)
                pred_cutie_logits = processor.step(image_buf, pred_caformer_buf, objects=objects)
                pred_cutie_logits = processor.step(image)
                total_pred_cutie_logits += pred_cutie_logits
        else:
            pred_cutie_logits = processor.step(image, pred_caformer_inp, objects=objects)

        pred_cutie_logits = total_pred_cutie_logits

        pred_cutie = processor.output_prob_to_mask(pred_cutie_logits)
        pred_cutie = pred_cutie.cpu().numpy().astype(np.uint8)

        # Postprocessing for CUTIE
        out_cutie = postprocess(pred_cutie, frame_cutie)

        # Combine Predictions
        pred_caformer_probs = torch.softmax(pred_caformer_logits, dim=0)
        pred_cutie_probs = torch.softmax(pred_cutie_logits, dim=0)
        pred_combined_probs = pred_caformer_probs + pred_cutie_probs
        pred_average = pred_combined_probs.argmax(dim=0).cpu().numpy().astype(np.uint8)

        # Postprocessing for Average Prediction
        out_average = postprocess(pred_average, frame_average)

        # Combine the three outputs for visualization
        out = np.hstack((out_caformer, out_cutie, out_average))

        # Update Tkinter Label with out_caformer (or any other output)
        img = Image.fromarray(out)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        ti +=1

    root.after(10, update)


# Start the video loop
update()

# Run the Tkinter loop
root.mainloop()

# Release resources after closing the window
cap.release()
cv2.destroyAllWindows()

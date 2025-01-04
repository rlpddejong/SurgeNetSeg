import os
import cv2
import pickle
import numpy as np
from PIL import Image

def encode_masks(masks):
    layers = [np.zeros(masks[0].shape, dtype=np.uint8)]
    layers_objs = [[]]
    objs_mapping = [(None, None)] * len(masks)
    ordered_masks = sorted(list(enumerate(masks)), key=lambda x: x[1].sum())[::-1]
    for global_id, obj_mask in ordered_masks:
        for layer_indx, (layer_mask, layer_objs) in enumerate(zip(layers, layers_objs)):
            if len(layer_objs) >= 255:
                continue
            if np.all(layer_mask[obj_mask] == 0):
                layer_objs.append(global_id)
                local_id = len(layer_objs)
                layer_mask[obj_mask] = local_id
                objs_mapping[global_id] = (layer_indx, local_id)
                break
        else:
            new_layer = np.zeros_like(layers[-1])
            new_layer[obj_mask] = 1
            objs_mapping[global_id] = (len(layers), 1)
            layers.append(new_layer)
            layers_objs.append([global_id])

    layers = [cv2.imencode('.png', x)[1] for x in layers]
    return layers, objs_mapping

def process_and_save_png_to_pickle(input_png_path, output_pickle_path):
    # Step 1: Load the PNG as a NumPy array
    semantic_mask = np.array(Image.open(input_png_path))
    
    # Step 2: Extract unique IDs and generate binary masks
    unique_ids = np.unique(semantic_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Exclude background if it's 0
    masks = [(semantic_mask == obj_id) for obj_id in unique_ids]

    with open(output_pickle_path, 'wb') as f:
            pickle.dump(encode_masks(masks), f)

if __name__ == '__main__':

    # Validation set
    original_path = r'G:\datasets\RAMIE_879_RITM\Validation\masks'
    new_path = r'G:\datasets\RAMIE_879_RITM\Validation\masks_pickle'
    
    # Train set
    original_path = r'G:\datasets\RAMIE_879_RITM\Train\masks'
    new_path = r'G:\datasets\RAMIE_879_RITM\Train\masks_pickle'

    # Test set
    original_path = r'G:\datasets\RAMIE_879_RITM\Test\masks'
    new_path = r'G:\datasets\RAMIE_879_RITM\Test\masks_pickle'


    os.makedirs(new_path, exist_ok=True)

    for file in os.listdir(original_path):
        input_png = os.path.join(original_path, file)
        output_pickle = os.path.join(new_path, file.replace('.png', '.pickle'))
        process_and_save_png_to_pickle(input_png, output_pickle)

    # # Test the pickle file
    # test_path = r'G:\datasets\RAMIE_879_RITM\Validation\masks_pickle\P0003video2_00-01-18_0001959.pickle'
    # with open(test_path, 'rb') as f:
    #      encoded_layers, objs_mapping = pickle.load(f)
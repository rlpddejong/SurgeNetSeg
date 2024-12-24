import os

import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

from hydra.core.global_hydra import GlobalHydra


@torch.inference_mode()
@torch.cuda.amp.autocast()
def main():

    # Define image and mask folders
    image_folder = './data/RAMIE/images/'
    mask_folder = './data/RAMIE/masks/'

    

    mask_names = os.listdir(mask_folder)

    for mask_name in mask_names:
        
        # Get filename w/o extension
        filename_wo_ext = mask_name.replace('.png','')
        
        # Get paths
        mask_path = os.path.join(mask_folder, mask_name)
        image_path = os.path.join(image_folder, filename_wo_ext)

        images = os.listdir(image_path)

        # Get and create save directory
        save_path = image_path.replace('/images/', '/propagated_masks/')
        
        # Only continue if not propagated previously
        if not os.path.exists(save_path): 
            
            os.makedirs(save_path)

            # Get mask and images upto mask and reverse
            images_prior = images[:images.index(mask_name)+1] 
            images_prior.sort(reverse=True)

            # Get mask and images after
            images_after = images[images.index(mask_name):] 
            images_after.sort(reverse=False)

            for images in [images_prior, images_after]:
                
                # obtain the Cutie model with default parameters -- skipping hydra configuration
                # Clear the GlobalHydra instance if it is already initialized
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
                cutie = get_default_model()
                # Typically, use one InferenceCore per video
                processor = InferenceCore(cutie, cfg=cutie.cfg)
                # the processor matches the shorter edge of the input to this size
                # you might want to experiment with different sizes, -1 keeps the original size
                processor.max_internal_size = 480

                # Load mask
                mask = Image.open(mask_path)
                mask_type = mask.mode if mask.mode in ['P', 'L'] else None
                if mask_type == 'P':
                    palette = mask.getpalette()

                # Get number of objects
                objects = np.unique(np.array(mask))
                objects = objects[objects != 0].tolist() # background "0" does not count as an object

                mask = torch.from_numpy(np.array(mask)).cuda()

                for ti, image_name in enumerate(images):
                    # load the image as RGB; normalization is done within the model
                    image = Image.open(os.path.join(image_path, image_name))
                    image = to_tensor(image).cuda().float()

                    if ti == 0:
                        # Propagate using initial mask
                        output_prob = processor.step(image, mask, objects=objects)
                    else:
                        # Otherwise, we propagate the mask from memory
                        output_prob = processor.step(image)

                    # convert output probabilities to an object mask
                    mask = processor.output_prob_to_mask(output_prob)

                    # visualize prediction
                    mask = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
                    
                    # Save as mmseg
                    if mask_type == 'P':
                        mask.putpalette(palette)
                    mask.save(os.path.join(save_path, image_name), format="PNG")

main()

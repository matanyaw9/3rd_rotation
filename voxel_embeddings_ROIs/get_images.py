import os 
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, PngImagePlugin
import torch


sys.path.append('/home/jonathak/VisualEncoder/Voxels_Prediction')
from predict_voxels_jonathan import get_images_for_prediction


def save_as_png(array, save_path, metadata=None):   
        # 1) Convert torch tensor → numpy
        if isinstance(array, torch.Tensor):
            array = array.cpu().numpy()

        # 2) Reorder channels if needed
        if array.ndim == 3 and array.shape[0] == 3:
            array = array.transpose(1, 2, 0)

        # 3) Convert dtype to uint8
        if array.dtype in (np.float32, np.float64):
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255).round().astype(np.uint8)
        elif array.dtype == np.uint8:
            pass
        else:
            raise ValueError(f"Unsupported array dtype: {array.dtype}")

         # 4) Build the PIL Image
        img = Image.fromarray(array)

        # 5) Attach metadata if provided
        if metadata:
            pnginfo = PngImagePlugin.PngInfo()
            for key, val in metadata.items():
                pnginfo.add_text(str(key), str(val))
            img.save(save_path, pnginfo=pnginfo)
        else:
            print('save_as_png was called with no metadata. Saving without metadata.')
            img.save(save_path)
    
print("Starts saving images\n")

images = get_images_for_prediction(image_type='excluded', subjects=[1])
images = images.permute(0, 2, 3, 1)
image_save_path = '/home/matanyaw/DIP_decoder/excluded_images'
os.makedirs(image_save_path, exist_ok=True)
for img_idx, image in enumerate(images):

    save_as_png(images[img_idx], f'{image_save_path}/{img_idx}.png')


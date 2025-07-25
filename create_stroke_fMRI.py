import os
import sys
import torch
import numpy as np
from plotly.subplots import make_subplots

from create_diff_brain_map_for_stroke import create_diff_brain_map_for_stroke

sys.path.append('/home/jonathak/VisualEncoder/Voxels_Prediction')
from predict_single_image_voxels import SingleImageVoxelPredictor 

sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps/Diff_maps')
from create_diff_brain_map import create_diff_brain_map

sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps')
from NIPS_utils import *
from create_full_brain_map import create_full_brain_map


class StrokeVoxelPredictor:
    """
    Class for zeroing out ("stroking") certain voxels in an fMRI map, based on
    thresholded difference maps. 
    """
    def __init__(self, gpu="0"):
        # GPU settings moved to constructor because they typically stay the same
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def create_stroked_voxels(
        self,
        img_idx,
        sub,
        threshold,
        original_voxels_path,
        masked_voxels_path,
        only_indices=False
    ):
        """
        Zeroes out certain voxels in the fMRI map based on thresholded difference maps.

        Args:
            img_idx (int): Index of the image whose voxels to modify.
            sub (int): Subject identifier, e.g. 1 or 2, etc.
            stroke_hemisphere (str): 'lh' or 'rh' (left or right hemisphere).
            threshold (float): Threshold above which voxels are zeroed.
            original_voxels_path (str): Path to the .pt file containing original voxel activations.
            masked_voxels_path (str): Path to the .pt file containing masked voxel activations.

        Returns:
            np.ndarray: The voxels (for that image and subject hemisphere) after zeroing out
                        any that exceed the threshold.
        """
        hemispheres = ['lh', 'rh']
        lh_start_idx, lh_end_idx = get_hemisphere_indices(sub, 'lh')
        rh_start_idx, rh_end_idx = get_hemisphere_indices(sub, 'rh')
        
        stroke_indices = {}
        
        # Creating the subject object and its boolean fsaverage mask
        
        data_dir = data_dir_navve
        parent_submission_dir = '/home/jonathak/VisualEncoder/Results/parent_submission_dir'
        args = argObj(data_dir, parent_submission_dir, sub)
        
        for hemisphere in hemispheres:
            
            roi_dir = os.path.join(
                args.data_dir, 'roi_masks',
                hemisphere[0] + 'h.all-vertices_fsaverage_space.npy'
            )
            fsaverage_all_vertices = np.load(roi_dir)

            # Creating the avg diff map and extracting stroke location
            mean_diff_map, _ = create_diff_brain_map_for_stroke(
                sub,
                hemisphere,
                voxel_map_path=masked_voxels_path,
                original_voxel_map_path=original_voxels_path,
                abs=True,
                std_normalize=False
            )

            # Getting the stroke indices

            mean_diff_map_subjects_space = mean_diff_map[fsaverage_all_vertices==1]
            stroke_indices[hemisphere] = np.where(mean_diff_map_subjects_space > threshold)[0]
            
            if hemisphere == 'lh':
                stroke_indices[hemisphere] = stroke_indices[hemisphere] + lh_start_idx
            else:
                stroke_indices[hemisphere] = stroke_indices[hemisphere] + rh_start_idx

        # Converting stroke_indices to a single list
        stroke_indices = [idx for hemisphere in hemispheres for idx in stroke_indices[hemisphere]]
        stroke_indices = np.array(stroke_indices)
        
        if only_indices:
            # print(f"Num of stroke indices: {len(stroke_indices)} out of {rh_end_idx - lh_start_idx}")
            return stroke_indices
        
        else:
            # Loading the original voxels and "stroking" them
            original_voxels = torch.load(original_voxels_path).numpy()
            img_original_voxels = original_voxels[img_idx, lh_start_idx:rh_end_idx]

            img_voxels_w_stroke = img_original_voxels.copy()
            img_voxels_w_stroke[stroke_indices] = 0

            return img_voxels_w_stroke


if __name__ == "__main__":
    """
    Below is example plotting code, or anything else you'd like to do to test
    this functionality when running the file directly. If you import
    StrokeVoxelZeroer from elsewhere, this block will NOT run automatically.
    """
    # For demonstration, the arguments are hardcoded here. In real usage, 
    # you might pass them from e.g. command-line args, function calls, etc.

    predictor = StrokeVoxelPredictor(gpu="0")
    img_idx = 20
    sub = 1
    stroke_hemisphere = 'rh'
    threshold = 0.8
    original_voxels_path = '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/original_voxels.pt'
    masked_voxels_path = '/home/jonathak/VisualEncoder/Results/all_masks_excluded_imgs/quadrant_mask_quad-top_left_mask_mean_voxels.pt'

    stroked_voxels = predictor.create_stroked_voxels(
        img_idx,
        sub,
        stroke_hemisphere,
        threshold,
        original_voxels_path,
        masked_voxels_path
    )

    # Example usage of "stroked_voxels," or any code you want for plotting, 
    # verifying, etc. (This might replicate your existing commented-out block.)
    print("Stroked voxels shape:", stroked_voxels.shape)

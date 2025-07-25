import os
import torch
import sys
import numpy as np
import sys
sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps')
from NIPS_utils import *

def create_diff_brain_map_for_stroke(sub, hemisphere, voxel_map_path, original_voxel_map_path, abs = False, std_normalize = False):
    '''
    Create a brain map of all voxels for a given hemisphere of a given subject (1 or 2)
    '''
    # Configure GPU settings
    gpu = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

    # Assert subject and hemisphere are valid
    assert sub in [1,2], "Subject index must be between 1 and 2"
    assert hemisphere in ['lh', 'rh'], "Hemisphere must be 'lh' or 'rh'"

    # Define data dir paths

    data_dir = data_dir_navve
    parent_submission_dir = '/home/jonathak/VisualEncoder/Results/parent_submission_dir'

    # Create an argObj object for the subject

    args = argObj(data_dir, parent_submission_dir, sub)
    
    # Load the voxel ativation maps
    
    voxels = torch.load(voxel_map_path).numpy() # transformed voxels
    original_voxels = torch.load(original_voxel_map_path).numpy() # original voxels
    
    # Take only the voxels that are in the hemisphere
    start_idx, end_idx = get_hemisphere_indices(sub, hemisphere)
    voxels = voxels[:,start_idx:end_idx]
    original_voxels = original_voxels[:,start_idx:end_idx]
    
    # Get mean and std of diff map
    
    diff_mean, diff_std = get_diff_mean_std_maps(transformed_voxels = voxels, original_voxels = original_voxels, abs = abs, std_normalize = std_normalize)
    
    # Normalize the voxel map
    # voxels = (voxels - np.min(voxels)) / (np.max(voxels) - np.min(voxels))

    # Load the brain surface map of all vertices

    roi_dir = os.path.join(args.data_dir, 'roi_masks',
    hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Create the map for the relevant vertices only and fill it with the voxel map
    
    fsaverage_response_mean = np.ones(len(fsaverage_all_vertices))*np.min(diff_mean)
    fsaverage_response_std = np.zeros(len(fsaverage_all_vertices))
    
    # print(f'The shape of the diff maps is mean: {diff_mean.shape} and std: {diff_std.shape}')
    # print(f'The shape of the fsaverage_response_mean is {fsaverage_response_mean[np.where(fsaverage_all_vertices)[0]].shape}')
    # print(f'The shape of the fsaverage_response_std is {fsaverage_response_std[np.where(fsaverage_all_vertices)[0]].shape}')
    
    assert (fsaverage_response_mean[np.where(fsaverage_all_vertices)[0]].shape == diff_mean.shape 
            and fsaverage_response_std[np.where(fsaverage_all_vertices)[0]].shape == diff_std.shape
            ), "The shape of the voxel map and the fsaverage_response are not the same"
    
    fsaverage_response_mean[np.where(fsaverage_all_vertices)[0]] = diff_mean 
    fsaverage_response_std[np.where(fsaverage_all_vertices)[0]] = diff_std

    return fsaverage_response_mean, fsaverage_response_std



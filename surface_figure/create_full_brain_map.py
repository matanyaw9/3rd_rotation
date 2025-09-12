import os
import torch
import sys
import numpy as np
sys.path.append('/home/matanyaw/VisualEncoder/Analysis/Brain_maps/')
from NIPS_utils import *

roi_colorscale = [
    [0.0, "lightgray"],  # 0 → background
    [1.0, "red"]         # 1 → ROI
]



def create_full_brain_map(sub, hemisphere, voxels, transformation_title, image_handling = 'mean', engine = 'plotly'):
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
    
    # Take only the voxels that are in the hemisphere
    start_idx, end_idx = get_hemisphere_indices(sub, hemisphere)
    voxels = voxels[:,start_idx:end_idx]

    # If there is more than one image in the voxel map, take the average
    if voxels.shape[0] > 1:
        if image_handling == 'mean':
            voxels = np.mean(voxels, axis=0)
        elif image_handling == 'std':
            voxels = np.std(voxels, axis=0)
        elif image_handling == 'pick_random':
            voxels = voxels[np.random.randint(voxels.shape[0])]
        elif image_handling == 'pick_first':
            voxels = voxels[0]
        else:
            raise ValueError(f"Invalid image handling: {image_handling}. Please choose 'mean' or 'max'.")
    elif voxels.shape[0] == 1:
        voxels = voxels.squeeze(0)
        
    # Normalize the voxel map
    #voxels = (voxels - np.min(voxels)) / (np.max(voxels) - np.min(voxels))

    # Load the brain surface map of all vertices
    roi_dir = os.path.join(args.data_dir, 'roi_masks', 
                           hemisphere[0]+'h.all-vertices_fsaverage_space.npy')
    fsaverage_all_vertices = np.load(roi_dir)

    # Create the map for the relevant vertices only and fill it with the voxel map
    # voxels_nonzero = voxels[voxels != 0]
    # out_value = 0.0
    # fsaverage_response = np.full(len(fsaverage_all_vertices), out_value, dtype=float)
    # if voxels_nonzero.size != 0:
    #     # No data for this hemisphere -> flat baseline map
    #     # (Optional) don’t bother writing voxels since they’re zero
    #     # out_value = voxels_nonzero.min()
    #     assert fsaverage_response[np.where(fsaverage_all_vertices)[0]].shape == voxels.shape, "The shape of the voxel map and the fsaverage_response are not the same"
    #     fsaverage_response[np.where(fsaverage_all_vertices)[0]] = 1.0
    
    # # fsaverage_response = np.ones(len(fsaverage_all_vertices)) * out_value
    # # assert fsaverage_response[np.where(fsaverage_all_vertices)[0]].shape == voxels.shape, "The shape of the voxel map and the fsaverage_response are not the same"
    # # fsaverage_response[np.where(fsaverage_all_vertices)[0]] = voxels 

    # Create the map for the relevant vertices only and fill it with the voxel map
    
    fsaverage_response = np.zeros(len(fsaverage_all_vertices)) 
    assert fsaverage_response[np.where(fsaverage_all_vertices)[0]].shape == voxels.shape, "The shape of the voxel map and the fsaverage_response are not the same"
    fsaverage_response[np.where(fsaverage_all_vertices)[0]] = voxels 

    # Create the title for the brain map
    if hemisphere == 'lh':
        hemisphere_title = 'Left Hemisphere'
    else:
        hemisphere_title = 'Right Hemisphere'
    
    # If no title is provided, keep it as None
    if transformation_title:
        title = f'{transformation_title}, Subject {sub}, {hemisphere_title}'
    else:
        title = None
    
    # Create the brain map
    if engine == 'surface_view':
        map = create_brain_map(sub,hemisphere,colorbar=True,fsaverage_response=fsaverage_response,title=title,cmap=None,symmetric_cmap = False)
        return map
    
    elif engine == 'plotly':

            # Create the Plotly 3D surface trace (Mesh3d)
            mesh = create_plotly_surface(
                fsaverage_response=fsaverage_response,
                hemisphere=hemisphere,
                cmap='Reds',
                # cmap = roi_colorscale,
                showscale=True  # later you can disable in a composite figure if needed
            )
            
            return mesh
        
            # # Optionally, create a standalone Figure
            # fig = go.Figure(data=[mesh])
            # fig.update_layout(
            #     title=title,
            #     scene=dict(aspectmode='data')
            # )

            # return fig  # or return mesh if you prefer just the trace



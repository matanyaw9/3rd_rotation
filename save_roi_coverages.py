import torch
import numpy as np
import os
import sys
# sys.path.append('/home/matanyaw/DIP_decoder/voxel_embeddings_ROIs')
from voxel_embeddings_ROIs import ROI_coverage, ROI_tsne_funcs
# Getting my modules
sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps')
from NIPS_utils import get_hemisphere_indices, get_roi_indices, get_roi_indices_per_hemisphere


# Setting up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Appending Roman's path
sys.path.append('/home/romanb/PycharmProjects/BrainVisualReconst/')

# Loading the model
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model = torch.load('/home/jonathak/VisualEncoder/Voxels_Prediction/model_ch128.pth').eval().cuda()

# Testing voxel embeddings
voxel_embeddings = model.voxel_embed # Has shape [315997, 256]

# Getting subject 1 indices
COVERAGES_DIR = '/home/matanyaw/DIP_decoder/data/roi_coverages'

subject = 1
lh_start, lh_end = get_hemisphere_indices(subject, 'lh')
rh_start, rh_end = get_hemisphere_indices(subject, 'rh')    
sub_indices = np.arange(lh_start, rh_end)

voxel_embeddings = voxel_embeddings[sub_indices]

ROI_names = ROI_coverage.get_roi_names(subject=subject)

predefined_ROI_indices = {}

# Creating a dictionary of ROI indices (iterating over copy because we remove ROIs that don't exist)
for ROI in ROI_names.copy():
    
    roi_indices = get_roi_indices(subject, ROI)
    
    if roi_indices is None:
        ROI_names.remove(ROI)
    else:
        predefined_ROI_indices[ROI] = roi_indices

roi_coverage_configs = []
roi_coverage_configs.append(ROI_coverage.InferRoiCoverageConfig(voxel_embeddings=voxel_embeddings, 
                                                                predefined_ROI_indices_dict=predefined_ROI_indices,
                                                                center_method=None, 
                                                                metric=None, 
                                                                discrimination_method='predefined'))

roi_coverage_configs.append(ROI_coverage.InferRoiCoverageConfig(voxel_embeddings=voxel_embeddings, 
                                                                predefined_ROI_indices_dict=predefined_ROI_indices,
                                                                center_method='mean', 
                                                                metric='cosine', 
                                                                discrimination_method='nearest_voxels'))

roi_coverage_configs.append(ROI_coverage.InferRoiCoverageConfig(voxel_embeddings=voxel_embeddings, 
                                                                predefined_ROI_indices_dict=predefined_ROI_indices,
                                                                center_method='mean', 
                                                                metric='cosine', 
                                                                discrimination_method='nearest_center'))

roi_coverage_configs.append(ROI_coverage.InferRoiCoverageConfig(voxel_embeddings=voxel_embeddings, 
                                                                predefined_ROI_indices_dict=predefined_ROI_indices,
                                                                center_method='meanshift', 
                                                                metric='cosine', 
                                                                discrimination_method='nearest_voxels'))

roi_coverage_configs.append(ROI_coverage.InferRoiCoverageConfig(voxel_embeddings=voxel_embeddings, 
                                                                predefined_ROI_indices_dict=predefined_ROI_indices,
                                                                center_method='meanshift', 
                                                                metric='cosine', 
                                                                discrimination_method='nearest_center'))


for coverage in roi_coverage_configs:
    print("Infering ROI Coverage:", coverage.name)
    coverage.infer_roi_coverage()
    path = os.path.join(COVERAGES_DIR, coverage.name + '.pkl')
    coverage.save(path)

print('Done saving all ROI coverages.')


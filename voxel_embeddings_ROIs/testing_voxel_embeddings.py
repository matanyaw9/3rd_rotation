import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.manifold import TSNE

# Getting my modules
sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps')
from NIPS_utils import get_hemisphere_indices, get_roi_indices

# Setting up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Appending Roman's path
sys.path.append('/home/romanb/PycharmProjects/BrainVisualReconst/')

# Loading the model
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model = torch.load('/home/jonathak/VisualEncoder/Voxels_Prediction/model_ch128.pth').eval().cuda()

# Testing voxel embeddings
voxel_embeddings = model.voxel_embed # Has shape [315997, 256]

# Getting ROI indices

sub = 1

ROIs_bodies = ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies']
ROIs_faces = ['OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces']
ROIs_places = ['OPA', 'PPA', 'RSC']
ROIs_words = ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']
ROIs_visual = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']

ROIs = ROIs_bodies + ROIs_faces + ROIs_places + ROIs_words + ROIs_visual

ROI_indices = {}

for ROI in ROIs:
    ROI_indices[ROI] = get_roi_indices(sub, ROI)

# Getting voxel embeddings for each ROI

ROIs_voxel_embeddings = {}

for ROI in ROIs:
    ROIs_voxel_embeddings[ROI] = voxel_embeddings[ROI_indices[ROI]]

print(ROIs_voxel_embeddings['EBA'].shape)

# Getting the t-SNE embeddings

ROIs_voxel_embeddings_tsne = {}

tsne = TSNE(n_components=2, random_state=42, perplexity=30)

for ROI in ROIs:
    tsne_embeddings = tsne.fit_transform(ROIs_voxel_embeddings[ROI])
    ROIs_voxel_embeddings_tsne[ROI] = tsne_embeddings

# Plotting voxel embeddings for a specific ROI

ROI = 'EBA'

# Create the scatter plot
plt.figure(figsize=(10, 10))
plt.scatter(ROIs_voxel_embeddings_tsne[ROI][:, 0], 
           ROIs_voxel_embeddings_tsne[ROI][:, 1],
           alpha=0.6)

# Set title and labels with larger font sizes
plt.title(f't-SNE Embeddings for {ROI}', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=14)
plt.ylabel('t-SNE Dimension 2', fontsize=14)

# Increase tick label sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Display the plot
plt.show()




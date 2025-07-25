import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance

"""
This script creates a file with a dictionary of ROIs and their voxel indices for a given subject.
"""

# Getting my modules
sys.path.append('/home/jonathak/VisualEncoder/Analysis/Brain_maps')
from NIPS_utils import get_hemisphere_indices, get_roi_indices, get_roi_indices_per_hemisphere


# Setting up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Appending Roman's path
sys.path.append('/home/romanb/PycharmProjects/BrainVisualReconst/')


def summary_roi_coverage(roi_indices, sub_indices):
    # turn sub_indices into a 1D numpy array
    if isinstance(sub_indices, torch.Tensor):
        sub_indices = sub_indices.cpu().numpy()
    else:
        sub_indices = np.array(sub_indices)

    # build a list of numpy arrays for all ROI-assigned voxels
    roi_arrays = []
    for roi, inds in roi_indices.items():
        if isinstance(inds, torch.Tensor):
            roi_arrays.append(inds.cpu().numpy())
        else:
            roi_arrays.append(np.array(inds))

    # now do your stats
    indices_with_roi = np.concatenate(roi_arrays)
    indices_with_roi_unique = np.unique(indices_with_roi)
    not_in_any_roi = np.setdiff1d(sub_indices, indices_with_roi_unique)

    print("\nSummary of ROI coverage:")
    print(f"Total number of voxels: {sub_indices.shape[0]}")
    print(f"Total number of voxels in ROIs: {indices_with_roi.shape[0]}")
    print(f"Unique voxels in ROIs: {indices_with_roi_unique.shape[0]}")
    print(f"Duplicities in ROIs: {indices_with_roi.shape[0] - indices_with_roi_unique.shape[0]}")
    print(f"Voxels not in any ROI: {not_in_any_roi.shape[0]}")
    return not_in_any_roi

def find_voxels_with_no_roi(sub_indices, ROI_indices):

    all_indices = np.concatenate(list(ROI_indices.values()))
    all_indices_unique = np.unique(all_indices)
    not_in_any_roi = np.setdiff1d(sub_indices, all_indices_unique)
    
    return not_in_any_roi


def get_average_distance(roi_indices, voxel_embeddings, center_point, metric='euclidean'):
    """
    Calculate the average distance of voxel embeddings in the ROI from the center of mass.
    
    Parameters:
    - roi_indices (Tensor): 1D tensor of voxel indices for the ROI
    - voxel_embeddings (Tensor): [num_voxels, embedding_dim]
    
    Returns:
    - float: Average distance from the center of mass
    """
    voxels = voxel_embeddings[roi_indices]
    center_point_dict = {"roi": center_point}
    distances, _ = infer_distances(voxels, center_point_dict, metric=metric)

    return distances.mean().item()
    

def infer_by_center_of_mass(predefined_ROI_indices, voxel_embeddings, top_k=None, threshold=None, use_angle=False):
    if threshold is None and top_k is None:
        raise ValueError("Either threshold or top_k must be provided.")
    if threshold is not None and top_k is not None:
        raise ValueError("Only one of threshold or top_k should be provided.")
    
    voxels = voxel_embeddings[predefined_ROI_indices]
    center_of_mass = voxels.mean(dim=0)

        # compute distances
    if use_angle:
        # normalize embeddings and center
        distances, _ = infer_distances(voxel_embeddings, {'roi': center_of_mass}, metric='cosine')  # [N, 1]

    else:
        distances = torch.norm(voxel_embeddings - center_of_mass, dim=1)    # Euclidean

    if threshold is not None:
        inferred_ROI_indices = torch.where(distances < threshold)[0]

    elif top_k is not None:
        inferred_ROI_indices = torch.topk(distances, k=top_k, dim=0, largest=False).indices

    return inferred_ROI_indices.cpu().numpy()


def infer_center_by_meanshift(predefined_ROI_indices: torch.Tensor,
                             voxel_embeddings: torch.Tensor,
                             quantile: float = 0.2,
                             n_samples: int = 500) -> torch.Tensor:
    """
    Infer the densest center (mode) of the ROI using Mean-Shift.

    Args:
      predefined_ROI_indices: 1D LongTensor of voxel indices for the ROI
      voxel_embeddings:      [num_voxels, embedding_dim] FloatTensor

    Returns:
      center: FloatTensor of shape [embedding_dim], the densest cluster center
    """

    # 1) Pull out ROI embeddings as a NumPy array
    X = voxel_embeddings[predefined_ROI_indices].detach().cpu().numpy()
    # 2) Estimate bandwidth (you can tweak quantile)
    bw = estimate_bandwidth(X,
                            quantile=quantile,
                            n_samples=min(len(X), n_samples))
    if bw <= 0:
        raise ValueError(f"Bandwidth came out non-positive: {bw}")

    # 3) Run Mean-Shift
    ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
    labels = ms.fit_predict(X)
    centers = ms.cluster_centers_

    # 4) Find the largest cluster
    counts = np.bincount(labels)
    best = counts.argmax()
    densest_center = centers[best]

    # 5) Convert back to torch, on the same device as voxel_embeddings
    device = voxel_embeddings.device
    return torch.from_numpy(densest_center).to(device).float()



def assign_voxels_to_rois(voxel_embeddings: torch.Tensor,
                          centers_dict: dict,
                          ROIs: list) -> (torch.Tensor, dict):
    """
    Assign each voxel to the nearest ROI center.

    Args:
      voxel_embeddings: [N, D] FloatTensor of all voxel embeddings.
      centers_dict:     {roi_name: center Tensor of shape [D]}.
      ROIs:             List of roi_names in the same order you’ll use for centers.

    Returns:
      labels:          LongTensor of shape [N], where labels[i] = j means voxel i → ROIs[j].
      roi_to_indices:  dict mapping each roi_name -> LongTensor of voxel indices assigned to it.
    """
    device = voxel_embeddings.device
    # 1) Stack your centers into [R, D]
    centers = torch.stack([centers_dict[roi] for roi in ROIs], dim=0).to(device)  # [R, D]

    # 2) Compute pairwise distances [N, R]
    #    Using torch.cdist (broadcasted Euclidean)
    dists = torch.cdist(voxel_embeddings, centers, p=2)  # [N, R]

    # 3) Find nearest center
    labels = torch.argmin(dists, dim=1)  # [N], values in 0..R-1

       # 4) Build reverse index
    roi_to_indices = {
        roi: torch.nonzero(labels == idx, as_tuple=False).squeeze(1)
        for idx, roi in enumerate(ROIs)
    }


    return labels, roi_to_indices


class ROIInferenceConfig:
    """
    Configuration object for ROI inference.

    Attributes:
        center_method (str): Method to find ROI center ('mean', 'meanshift', etc.)
        distance_method (str): Method to compute distance ('euclidean', 'angle', etc.)
        discrimination_method (str): How to discriminate ROI voxels ('threshold', 'top_k', etc.)
        params (dict): Additional parameters for methods.
    """
    def __init__(self, center_method='mean', metric='euclidean', discrimination_method='threshold', threshold_dict=None, top_k_dict=None, print_run=True):
        self.center_method = center_method
        self.distance_method = metric
        self.discrimination_method = discrimination_method
        self.threshold_dict = threshold_dict or {}
        self.top_k_dict = top_k_dict or {}
        self.print_run = print_run

    def __repr__(self):
        return (f"ROIInferenceConfig(center_method={self.center_method}, "
                f"distance_method={self.distance_method}, "
                f"discrimination_method={self.discrimination_method}, "
                f"params={self.params})")
    

def infer_roi_dict(voxel_embeddings, predefined_ROI_indices_dict, config: ROIInferenceConfig):
    """
    Infer the ROI based on the given configuration.

    Args:
        voxel_embeddings: [num_voxels, embedding_dim] FloatTensor of all voxel embeddings.
        predefined_ROI_indices: 1D LongTensor of voxel indices for the predefined ROI.
        config: ROIInferenceConfig object with inference settings.

    Returns:
        inferred_ROI_indices: 1D NumPy array of inferred voxel indices for the ROI.
    """

    centers = infer_centers(voxel_embeddings, predefined_ROI_indices_dict, center_method=config.center_method)
    distances, center_names = infer_distances(voxel_embeddings, centers, metric=config.distance_method)
    inferred_ROI_indices = infer_roi_indices(
        centers, distances, discrimination_method=config.discrimination_method,
        threshold_dict=config.threshold_dict, top_k_dict=config.top_k_dict
    )
    return inferred_ROI_indices

def infer_centers(voxel_embeddings, predefined_ROI_indices_dict, center_method='mean'):
    """
    Infer the center of the ROI based on the given configuration.

    Args:
        voxel_embeddings: [num_voxels, embedding_dim] FloatTensor of all voxel embeddings.
        predefined_ROI_indices: 1D LongTensor of voxel indices for the predefined ROI.
        config: ROIInferenceConfig object with inference settings.

    Returns:
        center: FloatTensor of shape [embedding_dim], the inferred center of the ROI.
    """
    inferred_centers = {}

    for ROI in predefined_ROI_indices_dict:
            if center_method == 'mean':
                voxels = voxel_embeddings[predefined_ROI_indices_dict[ROI]]
                center_of_mass = voxels.mean(dim=0)

            elif center_method == 'meanshift':
                center_of_mass = infer_center_by_meanshift(predefined_ROI_indices_dict[ROI], voxel_embeddings)
            
            else: 
                raise ValueError(f"Unknown center method: {center_method}")
            
            inferred_centers[ROI] = center_of_mass

    return inferred_centers

def infer_cosine_distances(voxel_embeddings: torch.Tensor, centers: torch.Tensor, eps=1e-8):
    device = voxel_embeddings.device
    # Normalize embeddings and centers
    centers = centers.to(voxel_embeddings.device)
    center_norm = centers / centers.norm(dim=1, keepdim=True).clamp_min(min=eps)  # [R, D]
    emb_norm = voxel_embeddings / voxel_embeddings.norm(dim=1, keepdim=True).clamp_min(min=eps)  # [N, D]
    # Compute cosine distances
    cos_sim   = emb_norm @ center_norm.t()
    distances = (1 - cos_sim).clamp(min=0.0, max=2.0)

    return distances


def infer_distances(voxel_embeddings, centers, metric='euclidean'):
    """
    Infer distances of all voxels to the given centers.

    Args:
        voxel_embeddings: [N, D] FloatTensor.
        centers: dict mapping names → [D]-tensors.
        metric: str or callable, optional (See torch.cdist documentation for options).
    Returns:
        distances: [N, R] Tensor of distances.
        center_names: list of length R.
    """
    device = voxel_embeddings.device
    center_names = list(centers.keys())
    center_tensors = torch.stack([centers[n] for n in center_names], dim=0).to(device)
    if metric == 'cosine':
        distances = infer_cosine_distances(voxel_embeddings, center_tensors)
    elif metric == 'euclidean':
        # Using torch.cdist for Euclidean distance
        distances = torch.cdist(voxel_embeddings, center_tensors, p=2)
    return distances, center_names
    


def infer_roi_indices(centers, distances, discrimination_method='threshold', threshold_dict={}, top_k_dict={}, print_run=True):
    """
    Infer ROI for each ROI indices based on distances to centers.

    Args:
        voxel_embeddings: [num_voxels, embedding_dim] FloatTensor of all voxel embeddings.
        centers: dict mapping ROI names to their inferred centers.
        distances: Tensor of shape [num_voxels, num_ROIs] with distances to each center.
        discrimination_method: Method to discriminate ROI voxels ('threshold', 'top_k', 'nearest_center').
        params: Additional parameters for the discrimination method.

    Returns:
        inferred_ROI_indices: dict mapping ROI names to 1D NumPy arrays of inferred voxel indices.
    """
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_TOP_K = 100
    inferred_ROI_indices = {}

    if discrimination_method == 'nearest_center':
        # For 'nearest_center', we simply assign each voxel to the nearest center
        voxel_assignments = distances.argmin(dim=1)
        for roi_idx, roi_name in enumerate(centers.keys()):
            indices = torch.where(voxel_assignments == roi_idx)[0].cpu().numpy()
            inferred_ROI_indices[roi_name] = indices
        return inferred_ROI_indices
    
    for roi_idx, roi_name in enumerate(centers.keys()):
        if discrimination_method == 'threshold':
            if len(threshold_dict) == 0:
                print(f"Using default threshold {DEFAULT_THRESHOLD} for all ROIs.")
            threshold = threshold_dict.get(roi_name, DEFAULT_THRESHOLD)
            
            print(f"Using threshold {threshold} for ROI {roi_name}")
            indices = torch.where(distances[:, roi_idx] < threshold)[0].cpu().numpy()
        
        elif discrimination_method == 'top_k':
            if len(top_k_dict) == 0:
                print(f"Using default top_k {DEFAULT_TOP_K} for all ROIs.")
            top_k = top_k_dict.get(roi_name, DEFAULT_TOP_K)
            if print_run:
                print(f"Using top_k {top_k} for ROI {roi_name}")
            indices = torch.topk(distances[:, roi_idx], k=top_k, largest=False).indices.cpu().numpy()
        
        else:
            raise ValueError(f"Unknown discrimination method: {discrimination_method}")
        
        inferred_ROI_indices[roi_name] = indices

    return inferred_ROI_indices

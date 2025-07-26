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

class RoiInferConfig:
    """
    Configuration object for ROI inference.

    Attributes:
        center_method (str): Method to find ROI center ('mean', 'meanshift')
        distance_method (str): Method to compute distance ('euclidean', 'cosine', etc.)
        discrimination_method (str): How to discriminate ROI voxels ('nearest_center', 'nearest_voxels', 'avg_distance')
        params (dict): Additional parameters for methods.
    """
    def __init__(self, 
                 voxel_embeddings: torch.Tensor,
                 predefined_ROI_indices_dict: dict,
                 center_method='mean', 
                 metric='euclidean', 
                 discrimination_method='nearest_voxels',
                 ):
        self.voxel_embeddings = voxel_embeddings
        self.predefined_ROI_indices_dict = predefined_ROI_indices_dict

        self.center_method = center_method
        self.metric = metric
        self.discrimination_method = discrimination_method
        self.ROI_names = list(predefined_ROI_indices_dict.keys())
        
        self.roi_centers = None
        self.inferred_ROI_indices_dict = None

    def __repr__(self):
        return (f"RoiInferConfig(center_method={self.center_method}, "
                f"distance_method={self.distance_method}, "
                f"discrimination_method={self.discrimination_method}, "
                )
    

    def infer_roi_dict(self):
        """The Main function - Infer the ROI indices based on the given configuration.
        """
        self.infer_centers()
        distances = self.infer_distances()
        self.infer_roi_indices(distances)
        return self.inferred_ROI_indices_dict

    
    def infer_centers(self):
        """Infer the centers of all ROIs based on the voxel embeddings.
        """
        inferred_centers = {}
        for ROI in self.predefined_ROI_indices_dict:
                if self.center_method == 'mean':
                    voxels = self.voxel_embeddings[self.predefined_ROI_indices_dict[ROI]]
                    center_of_mass = voxels.mean(dim=0)

                elif self.center_method == 'meanshift':
                    center_of_mass = infer_center_by_meanshift(self.predefined_ROI_indices_dict[ROI], self.voxel_embeddings)

                else:
                    raise ValueError(f"Unknown center method: {self.center_method}")
                
                inferred_centers[ROI] = center_of_mass
        self.roi_centers = inferred_centers
        return inferred_centers



    def infer_distances(self):
        """
        Infer distances from voxel embeddings to the centers of all ROIs.
        """
        device = self.voxel_embeddings.device
        # center_names = list(self.roi_centers.keys())
        center_tensors = torch.stack([self.roi_centers[n] for n in self.ROI_names], dim=0).to(device)
        if self.metric == 'cosine':
            distances = infer_cosine_distances(self.voxel_embeddings, center_tensors)
        elif self.metric == 'euclidean':
            # Using torch.cdist for Euclidean distance
            distances = torch.cdist(self.voxel_embeddings, center_tensors, p=2)
        return distances
    


    def infer_roi_indices(self, distances):
        """
        Infer the indices of voxels belonging to each ROI based on the distances to the centers.
        This method uses the configured discrimination method to determine how to assign voxels to ROIs.
        """
        inferred_ROI_indices = {}

        if self.discrimination_method == 'nearest_center':
            # For 'nearest_center', we simply assign each voxel to the nearest center
            voxel_assignments = distances.argmin(dim=1)
            for roi_idx, roi_name in enumerate(self.roi_centers.keys()):
                indices = torch.where(voxel_assignments == roi_idx)[0].cpu().numpy()
                inferred_ROI_indices[roi_name] = indices

        elif self.discrimination_method == 'avg_distance':
            inferred_ROI_indices = infer_by_avg_distance(self, distances)

        elif self.discrimination_method == 'nearest_voxels':
            inferred_ROI_indices = infer_by_nearest_voxels(self, distances)
        else:
            raise ValueError(f"Unknown discrimination method: {self.discrimination_method}")
        self.inferred_ROI_indices_dict = inferred_ROI_indices
        return inferred_ROI_indices
    

def infer_by_avg_distance(inferConfig: RoiInferConfig, distances):
    inferred_indices = {}
    for i, ROI in enumerate(inferConfig.ROI_names):
        predifined_indices = inferConfig.predefined_ROI_indices_dict[ROI]
        avg_dist = distances[predifined_indices, i].mean()
        print(f"Average distance for ROI '{ROI}': {avg_dist:.4f}")
        mask = distances[:, i] < avg_dist
        chosen = torch.where(mask)[0]
        inferred_indices[ROI] = chosen.cpu().numpy()
    return inferred_indices


def infer_by_nearest_voxels(inferConfig: RoiInferConfig, distances):
    inferred_indices = {}

    for i, ROI in enumerate(inferConfig.ROI_names):
        predifined_indices = inferConfig.predefined_ROI_indices_dict[ROI]
        roi_size = len(predifined_indices)
        relevant_distances = distances[:, i]
        inferred_indices[ROI] = torch.topk(relevant_distances, k=roi_size, dim=0, largest=False).indices.cpu().numpy()
    return inferred_indices

        
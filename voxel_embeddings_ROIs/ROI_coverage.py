import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# from sklearn.manifold import TSNE
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

def get_roi_names(subject=1):
    # Getting ROI indices
    ROIs_bodies = ['EBA', 'FBA-1', 'FBA-2', 'mTL-bodies']
    ROIs_faces = ['OFA', 'FFA-1', 'FFA-2', 'mTL-faces', 'aTL-faces']
    ROIs_places = ['OPA', 'PPA', 'RSC']
    ROIs_words = ['OWFA', 'VWFA-1', 'VWFA-2', 'mfs-words', 'mTL-words']
    ROIs_visual = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4']

    ROI_names = ROIs_bodies + ROIs_faces + ROIs_places + ROIs_words + ROIs_visual

    for ROI in ROI_names.copy():
        roi_indices = get_roi_indices(subject, ROI)

        if roi_indices is None:
            ROI_names.remove(ROI)
    return ROI_names
    



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
    

def infer_center_by_meanshift(predefined_ROI_indices: torch.Tensor,
                             voxel_embeddings: torch.Tensor,
                             metric='eucledian',
                             with_labels=False,
                             quantile: float = 0.2,
                             n_samples: int = 500) -> torch.Tensor:
    """
    Infer the densest center (mode) of the ROI using Mean-Shift.

    Args:
        predefined_ROI_indices: Tensor of shape [N] with indices of the voxels in the ROI.
        voxel_embeddings:      N, D] FloatTensor of all voxel embeddings.
        metric:                euclidean' or 'cosine' - distance metric to use.
        with_labels:           If true - will return also the voxel labels
        quantile:              Quantile for bandwidth estimation.
        n_samples:             Number of samples to use for bandwidth estimation.

    Returns:
      center: FloatTensor of shape [embedding_dim], the densest cluster center
    """

    # 1) Pull out ROI embeddings as a NumPy array
    if metric == 'cosine':
        X = voxel_embeddings[predefined_ROI_indices]
        X = F.normalize(X, dim=1).detach().cpu().numpy()
    else:
        X = voxel_embeddings[predefined_ROI_indices].detach().cpu().numpy()

    # 2) Estimate bandwidth (you can tweak quantile)
    bw = estimate_bandwidth(X,
                            quantile=quantile,
                            n_samples=min(len(X), n_samples))
    if bw <= 0:
        raise ValueError(f"Bandwidth came out non-positive: {bw}")

    # 3) Run Mean-Shift
    bin_seeding=True
    if metric == 'cosine': bin_seeding = False
    ms = MeanShift(bandwidth=bw, bin_seeding=bin_seeding, n_jobs=-1)
    labels = ms.fit_predict(X)
    centers = ms.cluster_centers_

    # 4) Find the largest cluster
    counts = np.bincount(labels)
    best = counts.argmax()
    densest_center = centers[best]

    # 5) Convert back to torch, on the same device as voxel_embeddings
    device = voxel_embeddings.device

    if with_labels:
        return torch.from_numpy(densest_center).to(device).float(), labels

    else:
        return torch.from_numpy(densest_center).to(device).float()


def infer_cosine_distances(voxel_embeddings: torch.Tensor, centers: torch.Tensor, eps=1e-8):
    # return distances
    device = voxel_embeddings.device
    centers = centers.to(device)

    # Use F.normalize to unit‐length each row, with epsilon for stability
    emb_norm    = F.normalize(voxel_embeddings, p=2, dim=1, eps=eps)  # [N, D]
    center_norm = F.normalize(centers,          p=2, dim=1, eps=eps)  # [R, D]

    # Cosine similarity then convert to distance
    cos_sim   = emb_norm @ center_norm.t()                 # [N, R]
    distances = (1 - cos_sim).clamp(min=0.0, max=2.0)       # [N, R]

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

class InferRoiCoverageConfig:
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
                 name=None,
                 ):
        self.voxel_embeddings = voxel_embeddings
        self.predefined_ROI_indices_dict = predefined_ROI_indices_dict

        self.center_method = center_method
        self.metric = metric
        self.discrimination_method = discrimination_method
        self.ROI_names = list(predefined_ROI_indices_dict.keys())

        
        self.roi_centers = None
        self.inferred_ROI_indices_dict = None
        self.ROIless_indices = None
        self.name = name
        
        if discrimination_method == 'predefined':
            self.inferred_ROI_indices_dict = predefined_ROI_indices_dict
            self.name = 'predefined'
        if self.name is None:
            self.make_default_name()

    def __repr__(self):
        return (f"RoiInferConfig(center_method={self.center_method}, "
                f"distance_method={self.distance_method}, "
                f"discrimination_method={self.discrimination_method}, "
                )
    

    def make_default_name(self):
        """This name will be used for storing in a file"""
        name = ''
        if self.center_method=='meanshift':
            name += 'ms'
        else:
            name += self.center_method
        name += '_'
        if self.metric == 'euclidean':
            name += 'euc'
        elif self.metric == 'cosine':
            name += 'cos'
        else: 
            name += self.metric
        name += '_'
        name += self.discrimination_method
        self.name = name
        

    def infer_roi_coverage(self):
        """The Main function - Infer the ROI indices based on the given configuration.
        Will only work if the discrimination method is not the 'predefined'.
        """
        if not self.discrimination_method == 'predefined':
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
                    center_of_mass = infer_center_by_meanshift(self.predefined_ROI_indices_dict[ROI], self.voxel_embeddings, metric=self.metric)

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
        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")
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
    
    def infer_roiless_indices(self, sub_indices):
        self.ROIless_indices = find_voxels_with_no_roi(sub_indices, self.inferred_ROI_indices_dict)


    def save_into_tezor(self, save_path):
        """This function is to save the information of which voxels belong to which ROI as a tenzor of 20 X 40K cells, binary.
        0 if the voxel is not in the ROI and 1 if it is. """
        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")

        # Create a binary tensor of shape [N, R] where N is the number of voxels and R is the number of ROIs
        num_voxels = self.voxel_embeddings.shape[0]
        num_rois = len(self.ROI_names)
        roi_tensor = torch.zeros((num_rois, num_voxels), dtype=torch.int8)

        for i, roi_name in enumerate(self.ROI_names):
            indices = self.inferred_ROI_indices_dict[roi_name]
            roi_tensor[i, indices] = 1

        # Save the tensor to a file
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, self.name + '.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(roi_tensor, save_path)
        
        return roi_tensor
    

def infer_by_avg_distance(inferConfig: InferRoiCoverageConfig, distances, print_info=False):
    inferred_indices = {}
    for i, ROI in enumerate(inferConfig.ROI_names):
        predifined_indices = inferConfig.predefined_ROI_indices_dict[ROI]
        avg_dist = distances[predifined_indices, i].mean()
        if print_info:
            print(f"Average distance for ROI '{ROI}': {avg_dist:.4f}")
        mask = distances[:, i] < avg_dist
        chosen = torch.where(mask)[0]
        inferred_indices[ROI] = chosen.cpu().numpy()
    return inferred_indices


def infer_by_nearest_voxels(inferConfig: InferRoiCoverageConfig, distances, k=None):
    """Every center claims the k nearest voxels to himself. 
    Multiple centers can claim the same voxel. If no K is given, by default k will be the ROI size."""
    inferred_indices = {}

    for i, ROI in enumerate(inferConfig.ROI_names):
        predifined_indices = inferConfig.predefined_ROI_indices_dict[ROI]
        roi_size = len(predifined_indices)
        if k is None:
            k = roi_size
        relevant_distances = distances[:, i]
        inferred_indices[ROI] = torch.topk(relevant_distances, k=k, dim=0, largest=False).indices.cpu().numpy()
    return inferred_indices


        
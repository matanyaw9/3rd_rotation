import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import nibabel as nib
from nilearn import datasets
# from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.sparse import coo_matrix, csgraph
import seaborn as sns
import pickle
from typing import List

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

NC = np.load("/home/romanb/data/datasets/NVD/tutorial_data/noise_ceiling/noise_ceiling.npy")


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

def list_pkl_files(directory: str) -> List[str]:
    """
    Return a sorted list of .pkl file paths in a directory.
    'predefined' files are placed first.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
    return sorted(files, key=lambda f: (0 if "predefined" in os.path.basename(f) else 1, f))

def load_coverages(directory, includes=[], equals=[], exclude=[]):
    """Loads many roi coverages, can specify for files that contain a word, excludes a word, or specific paths"""
    files = list_pkl_files(directory)
    if includes:
        files = [f for f in files if any(term in os.path.basename(f) for term in includes)]
    if equals:
        files = [f for f in files if any(os.path.basename(f) == term for term in equals)]
    if exclude:
        files = [f for f in files if not any(term in os.path.basename(f) for term in exclude)]

    coverages = [InferRoiCoverageConfig.load(f) for f in files]
    return coverages




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

def largest_surface_component(
    voxel_indices,        # 1D array/list of indices into the hemisphere-sliced voxel vector
    hemisphere,
    subject=1,                # 'lh' or 'rh'
    data_dir='/home/navvew/data/algonauts_2023_challenge_data/subj01',   # same data_dir you pass into argObj(...).data_dir
    mesh='fsaverage'           # 'fsaverage' | 'fsaverage6' | 'fsaverage5'
):
    """
    Returns the subset of hemi_voxel_indices that belong to the largest
    connectivity component on the fsaverage *surface* for the given hemisphere.
    """
    voxel_indices = np.asarray(voxel_indices, dtype=int)
    if voxel_indices.size == 0:
        return voxel_indices  # nothing to do
    
    # into hemisphere indexing
    lh_start, lh_end = get_hemisphere_indices(subject, 'lh')
    rh_start, rh_end = get_hemisphere_indices(subject, 'rh')

    start, end = get_hemisphere_indices(subject, hemisphere)

    hemi_voxel_indices = voxel_indices[(voxel_indices >= start) & (voxel_indices < end)] - start
    
    if len(hemi_voxel_indices) == 0:    # in case there are non of the voxels are in this hemisphere 
        return hemi_voxel_indices
    # 1) Load fsaverage mesh (inflated) and roi-mask that maps voxel indices -> surface vertices
    fsavg = datasets.fetch_surf_fsaverage(mesh)  # paths to GIFTI files
    mesh_file = fsavg['infl_left'] if hemisphere == 'lh' else fsavg['infl_right']
    gii = nib.load(mesh_file)
    coords = gii.darrays[0].data                 # (N_vertices, 3)
    faces  = gii.darrays[1].data.astype(np.int64)  # (N_faces, 3)

    mask_path = os.path.join(data_dir, 'roi_masks', f"{hemisphere[0]}h.all-vertices_fsaverage_space.npy")
    fsavg_mask = np.load(mask_path).astype(bool)  # length = N_vertices

    # Map hemi-local voxel indices to *global* vertex ids on the surface
    hemi_vertices = np.where(fsavg_mask)[0]      # vector of vertex ids that your model uses
    sel_vertices = hemi_vertices[hemi_voxel_indices]  # global vertex ids for the selected voxels

    # 2) Build sparse adjacency from faces (undirected)
    # Each triangle (a,b,c) gives edges (a-b, b-c, c-a) and their symmetric counterparts
    i = faces[:, [0,1,0,2,1,2]].ravel()
    j = faces[:, [1,0,2,0,2,1]].ravel()
    data = np.ones_like(i, dtype=np.uint8)
    nV = coords.shape[0]
    A = coo_matrix((data, (i, j)), shape=(nV, nV)).tocsr()

    # 3) Induce the subgraph on your selected vertices and find connected components
    subA = A[sel_vertices][:, sel_vertices]
    n_comp, labels = csgraph.connected_components(subA, directed=False)

    # 4) Keep largest component
    # np.bincount needs non-negative ints; find the label with max count
    largest_label = np.argmax(np.bincount(labels))
    keep_mask = (labels == largest_label)

    # Map back to the original *hemi voxel indices*
    return hemi_voxel_indices[keep_mask] + start

class InferRoiCoverageConfig:
    """
    Configuration object for ROI inference.

    Attributes:
        center_method (str): Method to find ROI center ('mean', 'meanshift')
        metric (str): Method to compute distance ('euclidean', 'cosine', etc.)
        discrimination_method (str): How to discriminate ROI voxels ('nearest_center', 'nearest_voxels', 'avg_distance')
        params (dict): Additional parameters for methods.
    """
    def __init__(self, 
                 voxel_embeddings: torch.Tensor,
                 predefined_ROI_indices_dict: dict,
                 subject:int=1,
                 center_method='mean', 
                 metric='euclidean', 
                 discrimination_method='nearest_voxels',
                 name=None,
                 ):
        self.voxel_embeddings = voxel_embeddings
        self.predefined_ROI_indices_dict = predefined_ROI_indices_dict
        self.subject = subject

        self.lh_start, self.lh_end = get_hemisphere_indices(subject, 'lh')
        self.rh_start, self.rh_end = get_hemisphere_indices(subject, 'rh') 
        self.center_method = center_method
        self.metric = metric
        self.discrimination_method = discrimination_method
        self.ROI_names = list(predefined_ROI_indices_dict.keys())

        
        self.roi_centers = None
        self.inferred_ROI_indices_dict = None
        self.name = name
        
        self._is_polished = False

        if discrimination_method == 'predefined':
            self.inferred_ROI_indices_dict = predefined_ROI_indices_dict
            self.name = 'predefined'
        if self.name is None:
            self.make_default_name()

    def __repr__(self):
        return (self.get_label())
    
    def __getitem__(self, roi_name):
        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")
        return self.inferred_ROI_indices_dict[roi_name]

    def copy(self): 
        return pickle.loads(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))

    def sub_indices(self):
        return np.arange(self.lh_start, self.rh_end)

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
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, InferRoiCoverageConfig):
            raise TypeError("Loaded object is not an InferRoiCoverageConfig instance.")
        return obj



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
    
    def get_roiless_indices(self):
        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")
        all_indices = np.concatenate(list(self.inferred_ROI_indices_dict.values()))
        all_indices_unique = np.unique(all_indices)
        roi_less = np.setdiff1d(self.sub_indices(), all_indices_unique)
        return roi_less
    

    def clear_islands(self):
        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")
        for roi in self.ROI_names:
            single_component_roi_indices_rh = largest_surface_component(self.inferred_ROI_indices_dict[roi], 'rh')
            single_component_roi_indices_lh = largest_surface_component(self.inferred_ROI_indices_dict[roi], 'lh')
            kept = np.concatenate([single_component_roi_indices_rh, single_component_roi_indices_lh])
            self.inferred_ROI_indices_dict[roi] = np.unique(kept)  # unique also sorts
        self.name += '_polished'
        self._is_polished = True
        return self
    
    
    def get_roi_size(self, ROI):
        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")
        if ROI not in self.ROI_names:
            raise ValueError(f"{ROI} is not a known ROI.")
        return len(self.inferred_ROI_indices_dict[ROI])
    
    
    def get_label(self):
        if 'predefined' in self.name:
            label = 'Predefined'
        else: 
            acronym = lambda words: ''.join([word[0].upper() for word in words.split('_')])
            label = self.center_method.capitalize() + ' '
            label += self.metric[:3].capitalize() + ' '
            label += acronym(self.discrimination_method) 
        if self._is_polished:
            label += ' Polished'
        return label

    def get_avg_SNR(self, roi_name:str='all', ndigits=3):
        """Returns the average Signal to Noise Ratio for ROI assigned voxels."""
        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")
        if roi_name not in self.ROI_names + ['all']:
            raise ValueError(f'{roi_name} is not in ROI_names! Try: {", ".join(["all"] + self.ROI_names)}')
        
        nc = NC[self.sub_indices()]
        if roi_name in self.ROI_names:
            indices = self.inferred_ROI_indices_dict[roi_name]
        elif roi_name == 'all':
            indices = np.concatenate(list(self.inferred_ROI_indices_dict.values()))
        return round(np.average(nc[indices]), ndigits)
        

    def into_numpy(self, ROIs=None):
        """
        Converts the inferred ROI indices into a numpy array representation.
        This function creates an array where each row corresponds to a specified ROI and each column corresponds to a voxel.
        The array contains 1s at positions where a voxel belongs to the ROI, and 0s elsewhere.
        Args:
            ROIs (list or str, optional): List of ROI names or a single ROI name to include in the array.
                If None, all known ROIs are used.
        Returns:
            np.ndarray: An array of shape (num_rois, num_voxels) with binary values indicating voxel membership in each ROI.
        """

        if self.inferred_ROI_indices_dict is None:
            raise ValueError("You need to run infer_roi_coverage() first to get the inferred ROI indices.")
        if ROIs is None:
            ROIs = self.ROI_names
        elif isinstance(ROIs, str):
            ROIs = [ROIs]
        
        for roi in ROIs:
            if roi not in self.ROI_names:
                raise ValueError(f"{roi} is not a known ROI.")
        num_voxels = self.voxel_embeddings.shape[0]
        num_rois = len(ROIs)
        roi_array = np.zeros((num_rois, num_voxels), dtype=np.int8)

        for i, roi_name in enumerate(ROIs):
            indices = self.inferred_ROI_indices_dict[roi_name]
            roi_array[i, indices] = 1
        return roi_array
    
    def plot_ROI_overlap(self, title=None):
        """
        Plots a heatmap where each cell[i, j] shows the percentage of ROI_i overlapping with ROI_j.
        Rows are ROI1, columns are ROI2.

        Parameters:
        - ROIs: list of ROI names
        - ROI_indices: dict mapping ROI name -> array of voxel indices
        - title: title for the plot
        """

        # Build overlap percentage matrix
        n = len(self.ROI_names)
        overlap_pct = np.zeros((n, n), dtype=float)
        for i, roi1 in enumerate(self.ROI_names):
            idx1 = self.inferred_ROI_indices_dict[roi1]
            size1 = self.get_roi_size(roi1)
            for j, roi2 in enumerate(self.ROI_names):
                idx2 = self.inferred_ROI_indices_dict[roi2]
                count = np.intersect1d(idx1, idx2).shape[0]
                overlap_pct[i, j] = (count / size1) * 100 if size1 > 0 else 0.0

        # Prepare labels with sizes
        xlabels = self.ROI_names
        ylabels = [f"{roi}\n(n={self.get_roi_size(roi)})" for roi in self.ROI_names]

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.set(style="white")
        ax = sns.heatmap(
            overlap_pct,
            annot=True, fmt='.1f', annot_kws={'size':9},
            cmap='Blues', xticklabels=xlabels, yticklabels=ylabels,
            cbar_kws={"label": "% of ROI1 overlapping ROI2", "shrink": .75},
            linewidths=0.5, linecolor='gray'
        )

        # Annotate number of ROI-less voxels if provided
        ROIless_indices_amount = self.get_roiless_indices()
        plt.gcf().text(0.99, 0.01, f'Voxels with no assigned ROI: {ROIless_indices_amount}', ha='right', va='bottom', fontsize=10)


        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        if title is None:
            title = self.get_label() + ' Overlaps'
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('ROI2', fontsize=12)
        plt.ylabel('ROI1', fontsize=12)
        plt.tight_layout()
        plt.show()


    

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


def infer_by_nearest_voxels(inferConfig: InferRoiCoverageConfig, distances, num_of_voxels=None):
    """Every center claims the k nearest voxels to himself. 
    Multiple centers can claim the same voxel. If no K is given, by default k will be the ROI size."""
    inferred_indices = {}

    for i, ROI in enumerate(inferConfig.ROI_names):
        predifined_indices = inferConfig.predefined_ROI_indices_dict[ROI]
        roi_size = len(predifined_indices)
        if num_of_voxels is None:
            k = roi_size
        else: 
            k = num_of_voxels
        relevant_distances = distances[:, i]
        inferred_indices[ROI] = torch.topk(relevant_distances, k=k, dim=0, largest=False).indices.cpu().numpy()
    return inferred_indices


        
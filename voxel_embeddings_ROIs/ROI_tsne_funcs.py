import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from openTSNE import TSNE as OTSNE

import os
import sys
from itertools import cycle
import ROI_actions
from ROI_actions import *


def plot_roi_intersection_tsne(roi1_indices, roi1_name, roi2_indices, roi2_name, voxel_embeddings_tsne):
    """
    Plots the t-SNE embeddings of two ROIs and their intersection.

    Parameters:
    - roi1_indices: Indices of the first ROI.
    - roi1_name: Name of the first ROI.
    - roi2_indices: Indices of the second ROI.
    - roi2_name: Name of the second ROI.
    - voxel_embeddings_tsne: t-SNE embeddings of the voxel embeddings.
    """
    plt.figure(figsize=(8, 6))

    intersection = np.intersect1d(roi1_indices, roi2_indices)
    roi1_only = np.setdiff1d(roi1_indices, intersection)
    roi2_only = np.setdiff1d(roi2_indices, intersection)

    # Plot first ROI
    plt.scatter(voxel_embeddings_tsne[roi1_only][:, 0], 
                voxel_embeddings_tsne[roi1_only][:, 1], 
                alpha=0.4, label=f'{roi1_name} ({len(roi1_indices)})', color='orange')

    # Plot second ROI
    plt.scatter(voxel_embeddings_tsne[roi2_only][:, 0], 
                voxel_embeddings_tsne[roi2_only][:, 1], 
                alpha=0.4, label=f'{roi2_name} ({len(roi2_indices)})', color='blue')

    # Highlight intersection
    if len(intersection) > 0:
        plt.scatter(voxel_embeddings_tsne[intersection][:, 0],
                    voxel_embeddings_tsne[intersection][:, 1],
                    alpha=1.0, color='green', label=f'Intersection ({len(intersection)})', s=10)

    plt.title(f't-SNE Intersection: {roi1_name} vs {roi2_name}', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_rois_tsne(voxel_embeddings_tsne, 
                       roi_index_dict, 
                       ROIs_to_plot, 
                       roi_centers_dict=None,
                       title="t-SNE of All ROI Embeddings", 
                       ROIless_indices=None):
    """
    Plots t-SNE embeddings of specified ROIs and optionally highlights voxels without any ROI.
    Parameters:
    - voxel_embeddings_tsne: t-SNE embeddings of the voxel embeddings.
    - roi_index_dict: Dictionary mapping ROI names to their indices.
    - ROIs_to_plot: List of ROI names to plot.
    - title: Title of the plot.
    - ROIless_indices: Indices of voxels that do not belong to any ROI (optional).
    """

    # Prepare colors
    cmap = cm.get_cmap('tab20', len(ROIs_to_plot))
    color_cycle = cycle([cmap(i) for i in range(cmap.N)])

    plt.figure(figsize=(10, 8))

    for roi_name in ROIs_to_plot:
        indices = roi_index_dict.get(roi_name)
        if indices is not None and len(indices) > 0:
            color = next(color_cycle)
            plt.scatter(voxel_embeddings_tsne[indices, 0],
                        voxel_embeddings_tsne[indices, 1],
                        label=f'{roi_name} ({len(indices)})',
                        alpha=0.6,
                        s=10,
                        color=color)
    if ROIless_indices is not None and len(ROIless_indices) > 0:

        plt.scatter(voxel_embeddings_tsne[ROIless_indices, 0],
                        voxel_embeddings_tsne[ROIless_indices, 1],
                        label=f'No-ROI Voxels ({len(ROIless_indices)})',
                        alpha=0.6,
                        s=10,
                        color='black')

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_rois_center_tsne(roi_config:InferRoiCoverageConfig, 
                          ROIs_to_plot=None, 
                          title='Rois plot',
                          voxel_embeddings_tsne=None,
                          roi_center_embeddings_tsne=None,
                          plot_roiless=True):
    """It's in a new function because together with the centers we have to do the tsne again 
    """
    
    if ROIs_to_plot is None:
        ROIs_to_plot = roi_config.ROI_names
    else: 
        ROIs_to_plot = [roi for roi in ROIs_to_plot if roi in roi_config.ROI_names]
    
    if voxel_embeddings_tsne is None or roi_center_embeddings_tsne is None:
        
        # Stack center embeddings
        roi_names = list(roi_config.ROI_names)

        roi_center_embeddings = np.vstack([roi_config.roi_centers[roi].detach().cpu().numpy() for roi in roi_names])

        all_embeddings = np.vstack([roi_config.voxel_embeddings.detach().cpu().numpy(), roi_center_embeddings])

        # tsne = TSNE(n_components=2, random_state=42, perplexity=30, metric="correlation")
        # voxel_embeddings_tsne = tsne.fit_transform(voxel_embeddings.detach().squeeze().cpu().numpy())

        # tsne = TSNE(n_components=2, perplexity=30, metric="cosine", random_state=42, n_jobs=-1)
        # voxel_embeddings_tsne = tsne.fit(voxel_embeddings.detach().squeeze().cpu().numpy())

        tsne = OTSNE(n_components=2, perplexity=30, metric="cosine", random_state=42, n_jobs=-1)
        all_tsne = tsne.fit(all_embeddings)

        voxel_embeddings_tsne = all_tsne[:len(roi_config.voxel_embeddings)]
        roi_center_embeddings_tsne = all_tsne[len(roi_config.voxel_embeddings):]

    cmap = cm.get_cmap('tab20', len(ROIs_to_plot))
    roi_color = {r: cmap(i) for i, r in enumerate(ROIs_to_plot)}

    plt.figure(figsize=(10, 8))

    
    # Plot voxels not assigned to any ROI
    if plot_roiless and roi_config.ROIless_indices is not None and len(roi_config.ROIless_indices) > 0:
        plt.scatter(voxel_embeddings_tsne[roi_config.ROIless_indices, 0],
                    voxel_embeddings_tsne[roi_config.ROIless_indices, 1],
                    label=f'No-ROI Voxels ({len(roi_config.ROIless_indices)})',
                    alpha=0.6,
                    s=10,
                    color='black')
        
    for roi_name in ROIs_to_plot:
        voxel_indices = roi_config.inferred_ROI_indices_dict.get(roi_name)
        if voxel_indices is not None and len(voxel_indices) > 0:

            # Plot voxels of the ROI
            plt.scatter(voxel_embeddings_tsne[voxel_indices, 0],
                        voxel_embeddings_tsne[voxel_indices, 1],
                        label=f'{roi_name} ({len(voxel_indices)})',
                        alpha=0.6,
                        s=10,
                        color=roi_color[roi_name])
    
    for roi_name in ROIs_to_plot:
        # Plot the center 
        i = ROIs_to_plot.index(roi_name)
        plt.scatter(roi_center_embeddings_tsne[i, 0],
                    roi_center_embeddings_tsne[i, 1],
                    color=roi_color[roi_name],
                    edgecolor='black',
                    marker='X',
                    s=80,
                    linewidths=1.5,
                    label=f'{roi_name} center')

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return voxel_embeddings_tsne, roi_center_embeddings_tsne

def plot_ROI_overlap_heatmap_percentage(ROI_indices, title, ROIless_indices_amount=None):
    """
    Plots a heatmap where each cell[i, j] shows the percentage of ROI_i overlapping with ROI_j.
    Rows are ROI1, columns are ROI2.

    Parameters:
    - ROIs: list of ROI names
    - ROI_indices: dict mapping ROI name -> array of voxel indices
    - title: title for the plot
    """
     # ensure everything is a CPU numpy array
    for roi in ROI_indices:
        idx = ROI_indices[roi]
        if torch.is_tensor(idx):
            ROI_indices[roi] = idx.cpu().numpy()
    # Compute ROI sizes
    ROI_names = list(ROI_indices.keys())
    roi_sizes = {roi_name: len(ROI_indices[roi_name]) for roi_name in ROI_names}

    # Build overlap percentage matrix
    n = len(ROI_names)
    overlap_pct = np.zeros((n, n), dtype=float)
    for i, roi1 in enumerate(ROI_names):
        idx1 = ROI_indices[roi1]
        size1 = roi_sizes[roi1]
        for j, roi2 in enumerate(ROI_names):
            idx2 = ROI_indices[roi2]
            count = np.intersect1d(idx1, idx2).shape[0]
            overlap_pct[i, j] = (count / size1) * 100 if size1 > 0 else 0.0

    # Prepare labels with sizes
    xlabels = ROI_names
    ylabels = [f"{roi}\n(n={roi_sizes[roi]})" for roi in ROI_names]

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
    if ROIless_indices_amount is not None:
        plt.gcf().text(0.99, 0.01, f'Voxels with no assigned ROI: {ROIless_indices_amount}', ha='right', va='bottom', fontsize=10)


    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('ROI2', fontsize=12)
    plt.ylabel('ROI1', fontsize=12)
    plt.tight_layout()
    plt.show()

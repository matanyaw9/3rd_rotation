#!/usr/bin/env python3
"""
Flexible brain-surface grid (Plotly) for ROI coverage meshes.

Key features:
- Single EDITOR ZONE to hard-code the exact grid (rows/cols & cells) OR use an
  automatic flow-by-columns mode with a simple filter + hemisphere list.
- No extra left label column; just a tight grid of scenes.
- Works with any number of columns.

Usage:
  python make_roi_grid.py --roi_name VWFA-1 --subject 1
"""

import argparse
import os
import sys
from typing import List, Tuple, Optional, Dict, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import datetime
import math

# Project imports
# import .voxel_embeddings_ROIs.ROI_coverage  # noqa: E402
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add DIP_decoder/
from voxel_embeddings_ROIs import ROI_coverage
from create_full_brain_map import create_full_brain_map  # noqa: E402

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = '/home/matanyaw/DIP_decoder/data/matanya_results'


# =========================
# EDITOR ZONE
# =========================
OUTPUT_DIR_NAME = f'html_files_{timestamp}'
# OUTPUT_DIR_NAME = f'nearest_voxels_vs_nearest_center'


# Choose ONE of the two modes: "manual" or "auto"

GRID_MODE = "manual"  # "manual" or "auto"

# ---- Manual mode ----
# Define your grid explicitly as a list of rows.
# Each cell is a dict describing which file and hemisphere to render.
# Matching is by filename (contains) OR by exact filename.
# Only one of "file_contains" or "file_equals" should be specified per cell.
# You can set label to "auto" for an auto generated label
#
# Example:
# MANUAL_GRID = [
#   [ {"file_contains": "predefined", "hemi": "lh", "label": "Predef LH"},
#     {"file_contains": "predefined", "hemi": "rh", "label": "Predef RH"} ],
#   [ {"file_contains": "nearest_voxels", "hemi": "lh"},
#     {"file_contains": "nearest_voxels", "hemi": "rh"} ],
# ]
# MANUAL_GRID: List[List[Dict[str, Any]]] = [
#     # Row 1
#     [
#         {"file_contains": "predefined", "hemi": "lh", "label": "Predefined · LH"},
#         {"file_contains": "predefined", "hemi": "rh", "label": "Predefined · RH"},
#     ],
#     # Row 2
#     [
#         {"file_contains": "nearest_voxels", "hemi": "lh", "label": "Nearest · LH"},
#         {"file_contains": "nearest_voxels", "hemi": "rh", "label": "Nearest · RH"},
#     ],
# ]

# # Comparing Different Metrics:
# MANUAL_GRID: List[List[Dict[str, Any]]] = [
#     # Row 1
#     [
#         {"file_contains": "predefined", "hemi": "lh", "label": "Predefined · LH"},
#         # {"file_contains": "predefined", "hemi": "rh", "label": "Predefined · RH"},
#     ],
#     # Row 2
#     [
#         {"file_equals": "mean_cos_nearest_voxels.pkl", "hemi": "lh", "label": "Mean Cos NV · LH"},
#         {"file_equals": "mean_euc_nearest_voxels.pkl", "hemi": "lh", "label": "Mean Euc NV  · LH"},
#     ],
#     # Row 3 
#     [
#         {"file_equals": "ms_cos_nearest_voxels.pkl", "hemi": "lh", "label": "Meanshift Cos NV · LH"},
#         {"file_equals": "ms_euc_nearest_voxels.pkl", "hemi": "lh", "label": "Meanshift Euc NV  · LH"},
#     ],
#     # Row 4
#     [
#         {"file_equals": "ms_cos_nearest_center.pkl", "hemi": "lh", "label": "Meanshift Cos NC · LH"},
#         {"file_equals": "ms_euc_nearest_center.pkl", "hemi": "lh", "label": "Meanshift Euc NC  · LH"},
#     ],
# ]

# # Nearest Voxel VS Nearest Center
# MANUAL_GRID: List[List[Dict[str, Any]]] = [
#     # Row 1
#     [
#         {"file_contains": "predefined", "hemi": "rh", "label": "auto"},
#         {"file_contains": "ms_cos_nearest_center.pkl", "hemi": "rh", "label": "auto"},
#     ],
#     # Row 2
#     [
#         {"file_equals": "mean_cos_nearest_voxels.pkl", "hemi": "rh", "label": "auto"},
#         {"file_equals": "ms_cos_nearest_voxels.pkl", "hemi": "rh", "label": "auto"},
#     ],
# ]

# polished vs unpolished
MANUAL_GRID: List[List[Dict[str, Any]]] = [
    # Row 1
    [
        {"file_contains": "predefined", "hemi": "rh", "label": "auto"},
        {"file_contains": "predefined", "hemi": "lh", "label": "auto"},
    ],
    # Row 2
    [
        {"file_equals": "mean_cos_nearest_voxels.pkl", "hemi": "rh", "label": "auto"},
        {"file_equals": "mean_cos_nearest_voxels_polished.pkl", "hemi": "rh", "label": "auto"},
    ],
        # Row 3
    [
        {"file_equals": "mean_cos_nearest_voxels.pkl", "hemi": "lh", "label": "auto"},
        {"file_equals": "mean_cos_nearest_voxels_polished.pkl", "hemi": "lh", "label": "auto"},
    ],
]


# ---- Auto mode ----
# If GRID_MODE == "auto", we:
#   1) load all .pkl coverages
#   2) keep those passing AUTO_FILTER(cov)
#   3) expand each kept coverage by HEMISPHERES (e.g., ["lh", "rh"] or ["lh"])
#   4) place them row-major into a grid with N_COLS columns (rows computed)
#
# You can also supply a CELL_LABEL function to control per-cell label text (or return "" for none).
def AUTO_FILTER(cov: "ROI_coverage.InferRoiCoverageConfig") -> bool:
    # Example: only nearest_voxels or predefined
    return cov.discrimination_method in {"nearest_voxels", "predefined"}

HEMISPHERES: List[str] = ["lh", "rh"]   # e.g. ["lh"] or ["lh", "rh"]
N_COLS: int = 3                          # auto mode: number of columns to use
# cell_label = lambda cov, roi, hemi: f"{cov.name} · size: {cov.get_roi_size(roi)} · {hemi.upper()}"

# ---- Shared presentation knobs ----
SHOW_COLORBAR: bool = False              # global toggle (on all cells)
BINARY_COLORSCALE = [[0, "white"], [1, "red"]]
C_MIN, C_MAX = 0, 1

# =========================
# END EDITOR ZONE
# =========================

# HELPER: file discovery
def list_pkl_files(directory: str) -> List[str]:
    """
    Return a sorted list of .pkl file paths in a directory.
    'predefined' files are placed first.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pkl')]
    return sorted(files, key=lambda f: (0 if "predefined" in os.path.basename(f) else 1, f))

def cell_label(cov: ROI_coverage.InferRoiCoverageConfig, roi, hemi):
    return f"{cov.get_label()} · Size: {cov.get_roi_size(roi)} · SNR: {cov.get_avg_SNR(roi, ndigits=2)} · {hemi.upper()}"

def scene_key_for(row: int, col: int, n_cols: int) -> str:
    """Return subplot scene key ('scene', 'scene2', ...) for (row, col) given n_cols."""
    idx = (row - 1) * n_cols + col
    return "scene" if idx == 1 else f"scene{idx}"


def build_manual_grid(
    args,
    roi_name: str,
    all_files: List[str],
) -> Tuple[List[List[go.Mesh3d]], List[List[str]]]:
    """
    Build a 2D list of Mesh3d (same shape as MANUAL_GRID), and parallel labels matrix.
    Each cell must resolve to exactly one file based on 'file_contains' or 'file_equals'.
    """
    meshes_grid: List[List[go.Mesh3d]] = []
    labels_grid: List[List[str]] = []

    for row in MANUAL_GRID:
        mesh_row: List[go.Mesh3d] = []
        label_row: List[str] = []
        for cell in row:
            hemi = cell.get("hemi")
            if hemi not in ("lh", "rh"):
                raise ValueError(f"Manual cell has invalid 'hemi': {hemi}")

            file_equals = cell.get("file_equals")
            file_contains = cell.get("file_contains")

            candidates = all_files
            if file_equals:
                candidates = [p for p in all_files if os.path.basename(p) == file_equals]
            elif file_contains:
                candidates = [p for p in all_files if file_contains in os.path.basename(p)]

            if not candidates:
                raise FileNotFoundError(
                    f"Manual cell not matched: file_equals={file_equals}, file_contains={file_contains}"
                )
            # If multiple candidates, pick the first after our predefined-first sort.
            chosen = candidates[0]

            cov = ROI_coverage.InferRoiCoverageConfig.load(chosen)
            voxels = cov.into_numpy(roi_name)
            mesh = create_full_brain_map(
                sub=args.subject,
                hemisphere=hemi,
                voxels=voxels,
                transformation_title=None,
                image_handling="mean",
                engine="plotly",
            )
            mesh.update(
                showscale=SHOW_COLORBAR,
                colorscale=BINARY_COLORSCALE,
                cmin=C_MIN,
                cmax=C_MAX,
            )
            mesh_row.append(mesh)
            label = cell.get("label", "")
            if label == 'auto':
                label = cell_label(cov, roi_name, hemi)
            label_row.append(label)

        meshes_grid.append(mesh_row)
        labels_grid.append(label_row)

    return meshes_grid, labels_grid


def build_auto_grid(
    args,
    roi_name: str,
    all_files: List[str],
) -> Tuple[List[List[go.Mesh3d]], List[List[str]]]:
    """
    Auto mode:
    - Filter coverages with AUTO_FILTER
    - Expand by HEMISPHERES
    - Row-major fill with N_COLS columns
    """
    selected_covs = []
    for path in all_files:
        cov = ROI_coverage.InferRoiCoverageConfig.load(path)
        if AUTO_FILTER(cov):
            selected_covs.append(cov)

    cells: List[Tuple[go.Mesh3d, str]] = []
    for cov in selected_covs:
        for hemi in HEMISPHERES:
            voxels = cov.into_numpy(roi_name)
            mesh = create_full_brain_map(
                sub=args.subject,
                hemisphere=hemi,
                voxels=voxels,
                transformation_title=None,
                image_handling="mean",
                engine="plotly",
            )
            mesh.update(
                showscale=SHOW_COLORBAR,
                colorscale=BINARY_COLORSCALE,
                cmin=C_MIN,
                cmax=C_MAX,
            )
            label = cell_label(cov, hemi)
            cells.append((mesh, label))

    if not cells:
        raise RuntimeError("Auto mode produced no cells. Adjust AUTO_FILTER / HEMISPHERES.")

    n_cols = max(1, int(N_COLS))
    n_rows = math.ceil(len(cells) / n_cols)

    # Fill grid row-major
    meshes_grid: List[List[go.Mesh3d]] = []
    labels_grid: List[List[str]] = []
    k = 0
    for r in range(n_rows):
        row_meshes: List[go.Mesh3d] = []
        row_labels: List[str] = []
        for c in range(n_cols):
            if k < len(cells):
                mesh, label = cells[k]
                row_meshes.append(mesh)
                row_labels.append(label)
                k += 1
        if row_meshes:  # last row might be short
            meshes_grid.append(row_meshes)
            labels_grid.append(row_labels)

    return meshes_grid, labels_grid


def create_single_html_file(args, roi_name: str):
    """Generate and save a grid of meshes (manual or auto)."""
    all_files = list_pkl_files(args.roi_dir)
    if not all_files:
        raise FileNotFoundError(f"No .pkl files found in: {args.roi_dir}")

    if GRID_MODE not in {"manual", "auto"}:
        raise ValueError("GRID_MODE must be 'manual' or 'auto'.")

    if GRID_MODE == "manual":
        meshes_grid, labels_grid = build_manual_grid(args, roi_name, all_files)
    else:
        meshes_grid, labels_grid = build_auto_grid(args, roi_name, all_files)

    n_rows = len(meshes_grid)
    n_cols = max(len(row) for row in meshes_grid) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        raise RuntimeError("Empty grid produced.")

    # Make subplot grid (no extra label column!)
    specs = [[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    # Add meshes
    for r, row in enumerate(meshes_grid, start=1):
        for c, mesh in enumerate(row, start=1):
            # skip missing cells in ragged last row
            if mesh is None:
                continue
            mesh.update(scene=scene_key_for(r, c, n_cols))
            fig.add_trace(mesh, row=r, col=c)

    # Scene cosmetics (hide axes etc.)
    total_scenes = n_rows * n_cols
    for idx in range(1, total_scenes + 1):
        key = "scene" if idx == 1 else f"scene{idx}"
        if key in fig.layout:
            fig.layout[key].update(aspectmode="data", bgcolor="white")
            for axis in ("xaxis", "yaxis", "zaxis"):
                getattr(fig.layout[key], axis).visible = False


    # Optional per-cell labels (top-center of each 3D scene, using paper coords)
    def _scene_center_top(fig, scene_key: str, dy: float = 0.02):
        dom = fig.layout[scene_key].domain
        # dom.x and dom.y are [start, end] within paper coords [0..1]
        x_center = 0.5 * (dom.x[0] + dom.x[1])
        y_top = dom.y[1] + dy
        return x_center, y_top

    ann = []
    for r in range(1, n_rows + 1):
        for c in range(1, len(meshes_grid[r - 1]) + 1):
            label = labels_grid[r - 1][c - 1] if labels_grid[r - 1] else ""
            if not label:
                continue
            scene_key = scene_key_for(r, c, n_cols)  # same helper you already use
            # Make sure this scene exists (ragged last row guard)
            if scene_key not in fig.layout or "domain" not in fig.layout[scene_key]:
                continue
            x, y = _scene_center_top(fig, scene_key, dy=0.02)
            ann.append(dict(
                x=x, y=y, xref="paper", yref="paper",
                text=label, showarrow=False, font=dict(size=14),
            ))

    if ann:
        # Extend, don't overwrite any existing annotations
        fig.update_layout(annotations=(fig.layout.annotations or []) + list(ann))

    # Final layout
    fig.update_layout(
        autosize=True,
        height=max(600, int(320 * n_rows)),
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        title=args.title or f"{roi_name} — {args.subject=}",
    )

    # Save HTML
    out_dir = args.output or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"{roi_name}_grid.html")
    fig.write_html(out_html)
    print(f"Saved: {out_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Flexible brain-surface grid of meshes for a given ROI (manual or auto grid)."
    )
    parser.add_argument("--roi_name", default="all",
                        help="ROI name to visualize (default: all ROIs)")
    parser.add_argument("--roi_dir", default="/home/matanyaw/data/roi_coverages",
                        help="Directory containing .pkl files of the ROI coverages")
    parser.add_argument("--subject", type=int, default=1, choices=[1, 2],
                        help="Subject index (default: 1)")
    DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, OUTPUT_DIR_NAME)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Directory where HTML will be saved (default: timestamped)")
    parser.add_argument("--title", default=None,
                        help="Figure title (default: auto)")

    args = parser.parse_args()

    if args.output is not None:
        if os.path.exists(args.output) and not os.path.isdir(args.output):
            raise ValueError(f"Output path {args.output} exists but is not a directory")
        os.makedirs(args.output, exist_ok=True)

    roi_names = ROI_coverage.get_roi_names(subject=args.subject)
    if args.roi_name.lower() == "all":
        rois_to_plot = roi_names
    elif args.roi_name in roi_names:
        rois_to_plot = [args.roi_name]
    else:
        raise ValueError(f"ROI '{args.roi_name}' not found for subject {args.subject}. "
                         f"Available: {', '.join(roi_names)} or 'all'")

    for roi_name in rois_to_plot:
        create_single_html_file(args, roi_name)


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() // 2))  # avoid oversubscription
    main()

#!/usr/bin/env python3
"""
Generate a 2-column (LH/RH) mesh grid of brain surfaces for a given ROI
across all `.pt` files in a directory, and export as an HTML file.
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import datetime

# Project imports
sys.path.append('/home/matanyaw/DIP_decoder/voxel_embeddings_ROIs')
import ROI_coverage  # noqa: E402
from create_full_brain_map import create_full_brain_map  # noqa: E402


def list_pt_files(directory: str) -> List[str]:
    """
    Return a sorted list of .pt file paths in a directory.
    Predefined files (containing 'predefined' in the name) come first.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
    return sorted(files, key=lambda f: (0 if "predefined" in os.path.basename(f) else 1, f))


def scene_name_for(rc: Tuple[int, int]) -> str:
    """Return subplot scene key ('scene', 'scene2', ...) for (row, col)."""
    r, c = rc
    idx = (r - 1) * 2 + c  # 2 columns
    return "scene" if idx == 1 else f"scene{idx}"


def create_single_html_file(args, roi_idx: int, roi_name: str):
    """Generate and save a grid of meshes for a single ROI across all .pt files."""
    pt_files = list_pt_files(args.tenzor_dir)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in: {args.tenzor_dir}")

    hemispheres = ["lh", "rh"]
    n_rows, n_cols = len(pt_files), 2

    # Build views: for each file × hemisphere
    views: List[go.Mesh3d] = []
    for voxel_map_path in pt_files:
        for hemisphere in hemispheres:
            mesh = create_full_brain_map(
                sub=args.subject,
                hemisphere=hemisphere,
                voxel_map_path=voxel_map_path,
                rows=[roi_idx],
                transformation_title=None,
                image_handling="mean",
                engine="plotly",
            )

            # Force binary colormap: 0 → white, 1 → red
            mesh.update(
                showscale=False,   # still suppress the colorbar if you don’t want it
                colorscale=[[0, "white"], [1, "red"]],
                cmin=0,
                cmax=1
            )
            views.append(mesh)

    # Create subplot grid
    specs = [[{"type": "scene"}, {"type": "scene"}] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows, cols=n_cols, specs=specs,
        horizontal_spacing=0.02, vertical_spacing=0.02
    )

    # Helper to add a mesh to subplot
    def add_mesh(mesh: go.Mesh3d, row: int, col: int, show_colorbar: bool = False):
        mesh.update(showscale=show_colorbar, scene=scene_name_for((row, col)))
        fig.add_trace(mesh, row=row, col=col)

    # Add all traces row by row
    k = 0
    for r in range(1, n_rows + 1):
        add_mesh(views[k], row=r, col=1, show_colorbar=False)  # left hemisphere
        k += 1
        add_mesh(views[k], row=r, col=2, show_colorbar=(args.show_colorbar and r == 1))  # right hemisphere
        k += 1

    # Layout adjustments
    total_scenes = n_rows * n_cols
    for idx in range(1, total_scenes + 1):
        key = "scene" if idx == 1 else f"scene{idx}"
        if key in fig.layout:
            fig.layout[key].update(aspectmode="data", bgcolor="white")
            for axis in ("xaxis", "yaxis", "zaxis"):
                getattr(fig.layout[key], axis).visible = False

    # Natural layout with reserved space for row labels
    label_frac, gutter_x, col_gap, gutter_y = 0.18, 0.02, 0.04, 0.02
    row_h = 1.0 / n_rows
    for r in range(1, n_rows + 1):
        y_top, y_bot = 1.0 - (r - 1) * row_h, 1.0 - r * row_h + gutter_y
        x_l0, x_l1 = label_frac, 0.5 - col_gap / 2
        x_r0, x_r1 = 0.5 + col_gap / 2, 1.0 - gutter_x
        for scene_key, domain in [
            (scene_name_for((r, 1)), dict(x=[x_l0, x_l1], y=[y_bot, y_top])),
            (scene_name_for((r, 2)), dict(x=[x_r0, x_r1], y=[y_bot, y_top])),
        ]:
            if scene_key not in fig.layout:
                fig.layout[scene_key] = {}
            fig.layout[scene_key].update(domain=domain, bgcolor="white")
            for axis in ("xaxis", "yaxis", "zaxis"):
                getattr(fig.layout[scene_key], axis).visible = False


    # Column headers
    left_header_x = label_frac + (0.5 - col_gap / 2 - label_frac) / 2
    right_header_x = (0.5 + col_gap / 2) + (1.0 - gutter_x - (0.5 + col_gap / 2)) / 2
    fig.update_layout(annotations=[
        dict(x=left_header_x, y=1.03, xref="paper", yref="paper",
             text="Left hemisphere", showarrow=False, font=dict(size=18)),
        dict(x=right_header_x, y=1.03, xref="paper", yref="paper",
             text="Right hemisphere", showarrow=False, font=dict(size=18)),
    ])

    # Row labels
    label_x = label_frac / 2.0
    row_labels = [
        dict(
            x=label_x,
            y=1.0 - (r - 0.5) / n_rows,
            xref="paper", yref="paper",
            xanchor="center", yanchor="middle",
            text=os.path.splitext(os.path.basename(path))[0],
            showarrow=False, font=dict(size=14),
        )
        for r, path in enumerate(pt_files, start=1)
    ]
    fig.update_layout(annotations=(fig.layout.annotations or []) + tuple(row_labels))

    # Final layout
    fig.update_layout(
        autosize=True,
        height=max(600, int(320 * n_rows)),
        margin=dict(l=10, r=10, t=90, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
    )

    # Save HTML
    out_dir = args.output or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_html = os.path.join(out_dir, f"{roi_name}_grid.html")
    fig.write_html(out_html)
    print(f"Saved: {out_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2-column (LH/RH) mesh grid of brain surfaces for a given ROI."
    )
    parser.add_argument("--roi_name", default="all",
                        help="ROI name to visualize (default: all ROIs)")
    parser.add_argument("--tenzor_dir", default="/home/matanyaw/data/roi_coverages_tenzors",
                        help="Directory containing .pt files")
    parser.add_argument("--subject", type=int, default=1, choices=[1, 2],
                        help="Subject index (default: 1)")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--output", default=f"/home/matanyaw/DIP_decoder/data/matanya_results/html_files_{timestamp}",
                        help="Directory where HTML will be saved (default: timestamped)")
    parser.add_argument("--title", default=None,
                        help="Figure title (default: auto)")
    parser.add_argument("--show_colorbar", action="store_true",
                        help="Show a colorscale on the first row, right hemisphere")
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
        roi_idx = roi_names.index(roi_name)
        create_single_html_file(args, roi_idx, roi_name)


if __name__ == "__main__":
    torch.set_num_threads(max(1, os.cpu_count() // 2))  # avoid oversubscription
    main()

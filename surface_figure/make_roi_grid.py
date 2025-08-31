#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Project imports
sys.path.append('/home/matanyaw/DIP_decoder/voxel_embeddings_ROIs')
import ROI_coverage  # noqa: E402
from create_full_brain_map import create_full_brain_map  # noqa: E402


def list_pt_files(directory: str) -> List[str]:
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
    files.sort()
    return files


def scene_name_for(rc: Tuple[int, int]) -> str:
    """Return subplot scene key ('scene', 'scene2', ...) for (row, col)."""
    r, c = rc
    idx = (r - 1) * 2 + c  # 2 columns
    return "scene" if idx == 1 else f"scene{idx}"


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2-column (LH/RH) mesh grid of brain surfaces for a given ROI across .pt files in a directory."
    )
    parser.add_argument("roi_name", help="ROI name to visualize (must exist for the chosen subject)")
    parser.add_argument("--tenzor-dir", default="/home/matanyaw/data/roi_coverages_tenzors", help="Directory containing .pt files")
    parser.add_argument("--subject", type=int, default=1, choices=[1, 2], help="Subject index (default: 1)")
    parser.add_argument("--output", default=None, help="Path to save HTML (default: <cwd>/<roi_name>_grid.html)")
    parser.add_argument("--title", default=None, help="Figure title (default: auto)")
    parser.add_argument("--show-colorbar", action="store_true",
                        help="Show a colorscale on the first row, right hemisphere (default: off)")
    args = parser.parse_args()

    pt_files = list_pt_files(args.tenzor_dir)
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in: {args.tenzor-dir}")

    # Validate ROI and get its row index
    roi_names = ROI_coverage.get_roi_names(subject=args.subject)
    if args.roi_name not in roi_names:
        raise ValueError(f"ROI '{args.roi_name}' not found for subject {args.subject}. "
                         f"Available: {', '.join(roi_names)}")
    roi_idx = roi_names.index(args.roi_name)

    hemispheres = ['lh', 'rh']
    n_rows = len(pt_files)
    n_cols = 2

    # Build views: for each file (row) × hemisphere (2 columns)
    views: List[go.Mesh3d] = []
    for voxel_map_path in pt_files:
        for hemisphere in hemispheres:
            mesh = create_full_brain_map(
                sub=args.subject,
                hemisphere=hemisphere,
                voxel_map_path=voxel_map_path,
                rows=[roi_idx],
                transformation_title=None,
                image_handling='mean',
                engine='plotly'
            )
            # Ensure no colorbar on traces by default; we can re-enable one later
            mesh.update(showscale=False)
            views.append(mesh)

    # Create subplot grid (each cell is a 3D scene)
    specs = [[{'type': 'scene'}, {'type': 'scene'}] for _ in range(n_rows)]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    # Helper to add a mesh to a specific subplot and bind it to the correct scene name
    def add_mesh(mesh: go.Mesh3d, row: int, col: int, show_colorbar: bool = False):
        mesh = mesh.update(showscale=show_colorbar)
        mesh = mesh.update(scene=scene_name_for((row, col)))
        fig.add_trace(mesh, row=row, col=col)

    # Add all traces row-major
    k = 0
    for r in range(1, n_rows + 1):
        # left hemisphere
        add_mesh(views[k], row=r, col=1, show_colorbar=False)
        k += 1
        # right hemisphere; optionally show colorbar only on the first row RH
        add_mesh(views[k], row=r, col=2, show_colorbar=(args.show_colorbar and r == 1))
        k += 1

    # # Column headers
    # annotations = [
    #     dict(x=0.25, y=1.04, xref="paper", yref="paper", xanchor='center',
    #          text="Left hemisphere", showarrow=False, font=dict(size=18)),
    #     dict(x=0.75, y=1.04, xref="paper", yref="paper", xanchor='center',
    #          text="Right hemisphere", showarrow=False, font=dict(size=18)),
    # ]

    # Row labels (filenames without extension), positioned along the left side
    # y position for row r (1..n_rows) centered within its band:
    # top of row r in paper coords ≈ 1 - (r-1)/n_rows; bottom ≈ 1 - r/n_rows
    # center y ≈ 1 - (r-0.5)/n_rows
    # for r, path in enumerate(pt_files, start=1):
    #     name = os.path.splitext(os.path.basename(path))[0]
    #     y = 1.0 - (r - 0.5) / n_rows
    #     annotations.append(dict(
    #         x=-0.02, y=y, xref="paper", yref="paper", xanchor='right',
    #         text=name, showarrow=False, font=dict(size=14)
    #     ))

    # # Title
    # fig_title = args.title or f"ROI: {args.roi_name} (Subject {args.subject})"
    # fig.update_layout(
    #     title=dict(text=fig_title, x=0.5, y=0.98, xanchor='center', yanchor='top', font=dict(size=22)),
    #     annotations=annotations,
    #     # margin=dict(l=120, r=40, t=80, b=40),
    #     paper_bgcolor='white',
    #     plot_bgcolor='white'
    # )
    # fig.update_layout(
    #     width=1000,                   # total width
    #     height=400 * n_rows,          # auto scale by number of rows
    #     autosize=False,
    #     margin=dict(l=200, r=40, t=80, b=40)  # more left margin for labels
    # )   

    


    # Make each scene pretty: data aspect mode + hide axes
    # scenes are 'scene', 'scene2', ..., up to n_rows * n_cols
    total_scenes = n_rows * n_cols
    for idx in range(1, total_scenes + 1):
        key = "scene" if idx == 1 else f"scene{idx}"
        if key in fig.layout:
            fig.layout[key].update(aspectmode='data', bgcolor='white')
            fig.layout[key].xaxis.visible = False
            fig.layout[key].yaxis.visible = False
            fig.layout[key].zaxis.visible = False

    # --- Natural layout: reserve a left strip for row labels, then 2 equal columns ---

    # fractions of the figure width
    label_frac = 0.18   # reserved for row labels (tweak 0.15–0.22)
    gutter_x   = 0.02   # small outer gutter on the right
    col_gap    = 0.04   # gap between LH and RH scenes
    gutter_y   = 0.02   # small vertical gutter inside each row

    row_h = 1.0 / n_rows
    for r in range(1, n_rows + 1):
        # row vertical domain (top to bottom)
        y_top = 1.0 - (r - 1) * row_h
        y_bot = 1.0 - r * row_h + gutter_y

        # column horizontal domains
        x_l0 = label_frac
        x_l1 = 0.5 - col_gap / 2
        x_r0 = 0.5 + col_gap / 2
        x_r1 = 1.0 - gutter_x

        # scene keys
        s_left  = scene_name_for((r, 1))
        s_right = scene_name_for((r, 2))

        # create scenes if missing (plotly can lazily create them)
        if s_left not in fig.layout:
            fig.layout[s_left] = {}
        if s_right not in fig.layout:
            fig.layout[s_right] = {}

        # set domains + prettify axes
        fig.layout[s_left].update(
            domain=dict(x=[x_l0, x_l1], y=[y_bot, y_top]),
            bgcolor="white"
        )
        fig.layout[s_right].update(
            domain=dict(x=[x_r0, x_r1], y=[y_bot, y_top]),
            bgcolor="white"
        )
        for axis in ("xaxis", "yaxis", "zaxis"):
            getattr(fig.layout[s_left], axis).visible = False
            getattr(fig.layout[s_right], axis).visible = False

    # --- Column headers placed above each column (centered over their domains) ---
    left_header_x  = label_frac + ( (0.5 - col_gap/2) - label_frac ) / 2
    right_header_x = (0.5 + col_gap/2) + ( (1.0 - gutter_x) - (0.5 + col_gap/2) ) / 2
    fig.update_layout(annotations=[
        dict(x=left_header_x,  y=1.03, xref="paper", yref="paper", xanchor='center',
            text="Left hemisphere",  showarrow=False, font=dict(size=18)),
        dict(x=right_header_x, y=1.03, xref="paper", yref="paper", xanchor='center',
            text="Right hemisphere", showarrow=False, font=dict(size=18)),
    ])

    # --- Row labels: now place them inside the reserved strip so they NEVER clip ---
    # center each label vertically in its row; put x at middle of the label strip
    label_x = label_frac / 2.0
    row_label_ann = []
    for r, path in enumerate(pt_files, start=1):
        name = os.path.splitext(os.path.basename(path))[0]
        y = 1.0 - (r - 0.5) / n_rows
        row_label_ann.append(dict(
            x=label_x, y=y, xref="paper", yref="paper",
            xanchor='center', yanchor='middle',
            text=name, showarrow=False, font=dict(size=14)
        ))
    fig.update_layout(annotations=(fig.layout.annotations or []) + tuple(row_label_ann))

    # --- Overall layout: responsive height with scrolling, tiny margins ---
    fig.update_layout(
        autosize=True,
        height=max(600, int(320 * n_rows)),   # tall -> browser will add vertical scrollbar
        margin=dict(l=10, r=10, t=90, b=10),
        paper_bgcolor="white", plot_bgcolor="white"
    )



    # Output
    out_path = args.output or os.path.join(os.getcwd(), f"{args.roi_name}_grid.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # Optional: speed up torch load when not using CUDA tensors
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    main()

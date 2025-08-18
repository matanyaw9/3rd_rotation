import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from create_full_brain_map import create_full_brain_map

# Creating the figure with surface subplots for right/left masks + original image

# Load the voxel maps and choose subject

voxel_map_paths = {
    'original': '/home/jonathak/VisualEncoder/Results/transformed_voxels_shared_image_mean/original_voxels.pt',
    'width_20': '/home/jonathak/VisualEncoder/Results/transformed_voxels_shared_image_mean/gaussian_mask_x0.5_y0.5_width_20_image_mean_voxels.pt',
    'width_50': '/home/jonathak/VisualEncoder/Results/transformed_voxels_shared_image_mean/gaussian_mask_x0.5_y0.5_width_50_image_mean_voxels.pt',
    'width_90': '/home/jonathak/VisualEncoder/Results/transformed_voxels_shared_image_mean/gaussian_mask_x0.5_y0.5_width_90_image_mean_voxels.pt',
}

sub = 1
hemispheres = ['lh', 'rh']

# Create the 8 brain map views

views = []

for voxel_map_path in voxel_map_paths.values():
    for hemisphere in hemispheres:
        view = create_full_brain_map(sub, hemisphere, voxel_map_path, transformation_title = None, image_handling = 'mean', engine = 'plotly')
        views.append(view)

# Create the 4x2 subplot grid.
# Each cell is a 3D scene.
fig = make_subplots(
    rows=4, cols=2,
    specs=[
        [{'type': 'scene'}, {'type': 'scene'}],
        [{'type': 'scene'}, {'type': 'scene'}],
        [{'type': 'scene'}, {'type': 'scene'}],
        [{'type': 'scene'}, {'type': 'scene'}]
    ],
    horizontal_spacing=0.0,
    vertical_spacing=0.0
)

# The default scene names in a subplot grid will be "scene", "scene2", etc.
# We map them to (row, col) for clarity:
scene_names = {
    (1, 1): "scene",   (1, 2): "scene2",
    (2, 1): "scene3",  (2, 2): "scene4",
    (3, 1): "scene5",  (3, 2): "scene6",
    (4, 1): "scene7",  (4, 2): "scene8"
}

# -----------------------------------------------------------------------------
# Helper function: add all traces from a given view to the designated subplot.
def add_mesh_trace(mesh, row, col, show_colorbar=False):
    """
    CHANGED: Now we pass the Mesh3d trace directly.
    We also toggle the colorbar for exactly one subplot (or none).
    """
    # Assign the correct scene for this subplot.
    mesh.update(scene=scene_names[(row, col)])
    # Turn off colorbar if desired.
    if not show_colorbar:
        mesh.update(showscale=False)
    fig.add_trace(mesh, row=row, col=col)

# -----------------------------------------------------------------------------
# Add traces to each subplot.
# For the right hemisphere column (col 2), we enable the colorbar only for row 1.

# Row 1 (Original):
add_mesh_trace(views[0], row=1, col=1, show_colorbar=False)
add_mesh_trace(views[1], row=1, col=2, show_colorbar=True)

# Row 2 (Width=20 mask):
add_mesh_trace(views[2], row=2, col=1, show_colorbar=False)
add_mesh_trace(views[3], row=2, col=2, show_colorbar=False)

# Row 3 (Width=50 mask):
add_mesh_trace(views[4], row=3, col=1, show_colorbar=False)
add_mesh_trace(views[5], row=3, col=2, show_colorbar=False)

# Row 4 (Width=90 mask):
add_mesh_trace(views[6], row=4, col=1, show_colorbar=False)
add_mesh_trace(views[7], row=4, col=2, show_colorbar=False)

# -----------------------------------------------------------------------------
# Add annotations for the headers.
# We add two types:
# 1. Column headers at the top: "Left Hemisphere" for left column and "Right Hemisphere" for right column.
# 2. Row labels on the left side: one for each row (Mask 1, Mask 2, Mask 3).
annotations = []

# Column titles:
annotations.append(dict(
    x=0.25, y=1.08, xref="paper", yref="paper",xanchor='center',
    text="Left hemisphere", showarrow=False, font=dict(size=20)
))
annotations.append(dict(
    x=0.75, y=1.08, xref="paper", yref="paper",xanchor='center',
    text="Right hemisphere", showarrow=False, font=dict(size=20)
))

# Row titles (positioned to the left of the left column).
# We choose y positions (in paper coordinates) to roughly center the rows.
annotations.append(dict(
    x=0, y=0.95, xref="paper", yref="paper",xanchor='center',
    text="Original", showarrow=False, font=dict(size=20)
))
annotations.append(dict(
    x=-0, y=0.7, xref="paper", yref="paper",xanchor='center',
    text="Small Gaussian", showarrow=False, font=dict(size=20)
))
annotations.append(dict(
    x=0, y=0.35, xref="paper", yref="paper",xanchor='center',
    text="Medium Gaussian", showarrow=False, font=dict(size=20)
))
annotations.append(dict(
    x=0, y=0.1, xref="paper", yref="paper",xanchor='center',
    text="Large Gaussian", showarrow=False, font=dict(size=20)
))

fig.update_layout(
    annotations=annotations,
    margin=dict(l=80, r=50, t=100, b=50),
    title=dict(
    text="Predicted fMRI for Gaussian masks (subject 1)",
    x=0.5,  # Center the title
    y=0.98,  # Position from bottom (0) to top (1)
    xanchor='center',
    yanchor='top',
    font=dict(size=24)  # Made slightly larger than requested for better hierarchy
    )
)

# Set each scene to use data aspect mode so that the 3D geometry is preserved and disable the axes
for scene in scene_names.values():
    fig.layout[scene].update(aspectmode='data',
                            bgcolor='white')
    
    fig.layout[scene].xaxis.visible = False
    fig.layout[scene].yaxis.visible = False
    fig.layout[scene].zaxis.visible = False

fig.update_layout(
    scene=dict(
        domain=dict(x=[0.0, 0.495], y=[0.75, 1.0])   # row=1, col=1
    ),
    scene2=dict(
        domain=dict(x=[0.505, 1.0], y=[0.75, 1.0])   # row=1, col=2
    ),
    scene3=dict(
        domain=dict(x=[0.0, 0.495], y=[0.5, 0.75])  # row=2, col=1
    ),
    scene4=dict(
        domain=dict(x=[0.505, 1.0], y=[0.5, 0.75])  # row=2, col=2
    ),
    scene5=dict(
        domain=dict(x=[0.0, 0.495], y=[0.25, 0.5])   # row=3, col=1
    ),
    scene6=dict(
        domain=dict(x=[0.505, 1.0], y=[0.25, 0.5])   # row=3, col=2
    ),
    scene7=dict(
        domain=dict(x=[0.0, 0.495], y=[0.0, 0.25])   # row=4, col=1
    ),
    scene8=dict(
        domain=dict(x=[0.505, 1.0], y=[0.0, 0.25])   # row=4, col=2
    )
)

# Save the figure
fig.write_html('/home/jonathak/VisualEncoder/Analysis/Brain_maps/gaussian_masks_surfaces_figure.html')


from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import os

MANUAL_GRID: List[List[Dict[str, Any]]] = [
    [
        {"file_equals": "img_7_stroke_predefined_lh_EBA_start_fMRI.png", "label": "auto"},
        {"file_equals": "img_7_stroke_predefined_rh_EBA_start_fMRI.png", "label": "auto"},
    ],
    [
        {"file_equals": "img_7_stroke_predefined_EBA_start_fMRI.png",    "label": "auto"},
        {"file_equals": "img_7_stroke_predefined_lh_EBA_start_fMRI.png", "label": "auto"},
        {"file_equals": "img_7_stroke_predefined_rh_EBA_start_fMRI.png", "label": "auto"},
    ],
]

# Set your image directory here
IMAGE_DIR = "/home/matanyaw/DIP_decoder/data/matanya_results/results_25_09_15/run_should_work/img_7/roi_EBA"

def load_image(filename: str) -> Image.Image:
    path = os.path.join(IMAGE_DIR, filename)
    return Image.open(path)

def add_label(img: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    text_size = draw.textsize(label, font=font)
    padding = 5
    draw.rectangle([0, 0, text_size[0] + 2 * padding, text_size[1] + 2 * padding], fill=(255,255,255,128))
    draw.text((padding, padding), label, fill="black", font=font)
    return img

def create_montage(grid: List[List[Dict[str, Any]]], output_path: str = "montage.png"):
    # Load images and add labels
    images_grid = []
    max_row_height = []
    max_col_width = []

    # Load and label images
    for row in grid:
        img_row = []
        for cell in row:
            img = load_image(cell["file_equals"])
            img = add_label(img, cell["label"])
            img_row.append(img)
        images_grid.append(img_row)

    # Compute max heights and widths
    num_rows = len(images_grid)
    num_cols = max(len(row) for row in images_grid)
    max_row_height = [max(img.height for img in row) for row in images_grid]
    max_col_width = [0] * num_cols
    for col in range(num_cols):
        max_col_width[col] = max(
            (row[col].width if col < len(row) else 0) for row in images_grid
        )

    # Compute total montage size
    total_width = sum(max_col_width)
    total_height = sum(max_row_height)

    montage = Image.new("RGB", (total_width, total_height), color=(220,220,220))

    # Paste images
    y_offset = 0
    for row_idx, row in enumerate(images_grid):
        x_offset = 0
        for col_idx, img in enumerate(row):
            montage.paste(img, (x_offset, y_offset))
            x_offset += max_col_width[col_idx]
        y_offset += max_row_height[row_idx]

    montage.save(output_path)
    print(f"Montage saved to {output_path}")

if __name__ == "__main__":
    create_montage(MANUAL_GRID)
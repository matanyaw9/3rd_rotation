import os
import math
import argparse
from PIL import Image, ImageDraw, ImageFont

# For Pillow >=10 compatibility
try:
    Resampling = Image.Resampling
except AttributeError:
    Resampling = Image

# Font sizing parameters
MAX_FONT_SIZE = 20
MIN_FONT_SIZE = 10
MAIN_TITLE_EXTRA_SIZE = 6  # extra size for main title
try:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    _ = ImageFont.truetype(FONT_PATH, size=MAX_FONT_SIZE + MAIN_TITLE_EXTRA_SIZE)
    SCALABLE = True
except Exception:
    FONT_PATH = None
    SCALABLE = False

# Fallback default font
DEFAULT_FONT = ImageFont.load_default()
# Dummy draw for measuring
_dummy_img = Image.new('RGB', (1, 1))
_dummy_draw = ImageDraw.Draw(_dummy_img)


def get_image_title(img):
    """
    Extracts a title from image metadata if available.
    Supports PNG 'Title' chunk and JPEG EXIF ImageDescription.
    """
    info = img.info
    title = info.get('Title') or info.get('title') or info.get('Description')
    if title:
        return title
    try:
        exif = img.getexif()
        desc = exif.get(0x010E)
        if desc:
            return desc
    except Exception:
        pass
    return os.path.basename(img.filename)

def delete_montage_files(directory):
    # Delete any image file in output_dir with "montage" in its name
    for f in os.listdir(directory):
        if "montage" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tiff')):
            try:
                os.remove(os.path.join(directory, f))
            except Exception as e:
                print(f"Warning: could not delete {f}: {e}")


def create_montage(input_dir, input_dir2=None, output_path=None, cols=None, thumb_size=None,
                   bg_color=(255, 255, 255), gap=10, main_title=None):
    # Collect image files
    if output_path is None and main_title is not None:
        output_path = os.path.join(input_dir, f'montage_{main_title.replace(" ", "_")}.png')
    elif output_path is None:
        output_path = os.path.join(input_dir, 'montage.png')
        
    output_dir = os.path.dirname(output_path)
    delete_montage_files(output_dir)


    img_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tiff'))]
    if input_dir2:
        files2 = [os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', 'tiff'))]
        img_files = files2 + img_files
    if not img_files:
        raise RuntimeError(f"No images found in {input_dir}")

    # Load images and extract titles
    loaded = []
    for path in img_files:
        img = Image.open(path)
        img.filename = path
        img_title = get_image_title(img)
        img_timestamp = img.info.get('Timestamp', None)
        loaded.append((img, img_title, img_timestamp))

    # Sort loaded by timestamp (None values last)
    loaded.sort(key=lambda x: (x[2] is None, x[2]))
    n = len(loaded)
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Measure main title if provided
    if main_title:
        if SCALABLE:
            font_main = ImageFont.truetype(FONT_PATH, size=MAX_FONT_SIZE + MAIN_TITLE_EXTRA_SIZE)
        else:
            font_main = DEFAULT_FONT
        bbox_main = _dummy_draw.textbbox((0, 0), main_title, font=font_main)
        main_title_h = bbox_main[3] - bbox_main[1]
        reserved_main = gap + main_title_h + gap
    else:
        reserved_main = 0

    # Determine reserved space for each image title
    if SCALABLE:
        reserved_title_space = MAX_FONT_SIZE + 4
    else:
        # approximate
        bbox = _dummy_draw.textbbox((0, 0), "Sample", font=DEFAULT_FONT)
        reserved_title_space = bbox[3] - bbox[1] + 4

    # Determine thumbnail dimensions
    if thumb_size:
        tw, th_img = thumb_size
    else:
        widths, heights = zip(*[(im.size[0], im.size[1]) for im, _, __ in loaded])
        tw = max(widths)
        th_img = max(heights)
    th = reserved_title_space + th_img

    # Calculate montage dimensions
    montage_w = cols * tw + (cols + 1) * gap
    montage_h = reserved_main + rows * th + (rows + 1) * gap
    montage = Image.new('RGB', (montage_w, montage_h), color=bg_color)
    draw = ImageDraw.Draw(montage)

    # Draw main title
    if main_title:
        # center horizontally
        bbox_main = draw.textbbox((0, 0), main_title, font=font_main)
        w_main = bbox_main[2] - bbox_main[0]
        x_main = (montage_w - w_main) // 2
        y_main = gap
        draw.text((x_main, y_main), main_title, fill=(0, 0, 0), font=font_main)

    # Draw each thumbnail and its title
    for idx, (img, img_title, img_timestamp) in enumerate(loaded):
        print(f"Processing image {idx + 1}/{n}: {img_title} ({img.filename})")
        im_thumb = img.copy()
        im_thumb.thumbnail((tw, th_img), resample=Resampling.LANCZOS)

        row, col = divmod(idx, cols)
        x0 = gap + col * (tw + gap)
        y0 = reserved_main + gap + row * (th + gap)

        # Select font for this title
        if SCALABLE:
            for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -1):
                font_candidate = ImageFont.truetype(FONT_PATH, size=size)
                bbox = draw.textbbox((0, 0), img_title, font=font_candidate)
                if bbox[2] - bbox[0] <= tw:
                    font_used = font_candidate
                    break
            else:
                font_used = ImageFont.truetype(FONT_PATH, size=MIN_FONT_SIZE)
        else:
            font_used = DEFAULT_FONT

        # Draw image title
        bbox = draw.textbbox((0, 0), img_title, font=font_used)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        tx = x0 + (tw - text_w) // 2
        ty = y0 + (reserved_title_space - text_h) // 2
        draw.text((tx, ty), img_title, fill=(0, 0, 0), font=font_used)

        # Paste thumbnail
        ix = x0 + (tw - im_thumb.width) // 2
        iy = y0 + reserved_title_space
        montage.paste(im_thumb, (ix, iy))

    montage.save(output_path)
    print(f"Montage saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create an image montage from a directory of images.')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('--input_dir2', default=None, help='Optional second directory for images')
    parser.add_argument('--output_path', default=None, help='File path to save montage (e.g., montage.png)')
    parser.add_argument('--cols', type=int, default=None, help='Number of columns in the grid')
    parser.add_argument('--thumb-width', type=int, default=None, help='Max width of each thumbnail')
    parser.add_argument('--thumb-height', type=int, default=None, help='Max height of each thumbnail')
    parser.add_argument('--gap', type=int, default=10, help='Gap (px) between cells and around edges')
    parser.add_argument('--main-title', dest='main_title', default=None,
                        help='Optional main title text centered above the montage')
    args = parser.parse_args()

    thumb_size = None
    if args.thumb_width and args.thumb_height:
        thumb_size = (args.thumb_width, args.thumb_height)
    create_montage(args.input_dir, input_dir2=args.input_dir2, output_path=args.output_path,
                   cols=args.cols, thumb_size=thumb_size,
                   gap=args.gap, main_title=args.main_title)

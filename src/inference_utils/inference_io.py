# Path validation, ground-truth loading (from raw masks OR YOLO-seg label
# files), mask sanity checks, and a reusable crop helper for the inference
# pipeline. Polygon-validity fixing (contour repair) already happens once at
# dataset-build time in polygon_preprocessing.py, so it isn't repeated here -
# this file only checks the label .txt files themselves are well-formed.

import os
import cv2
import numpy as np


def check_paths(images_path, masks_path=None, labels_path=None,
                 stage1_weights=None, stage2_weights=None):
    """Validates that all provided paths actually exist before doing any work."""
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")

    if masks_path is not None and not os.path.exists(masks_path):
        raise FileNotFoundError(f"Masks path not found: {masks_path}")

    if labels_path is not None and not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels path not found: {labels_path}")

    if stage1_weights is not None and not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"Stage 1 weights not found: {stage1_weights}")

    if stage2_weights is not None and not os.path.exists(stage2_weights):
        raise FileNotFoundError(f"Stage 2 weights not found: {stage2_weights}")


def get_background_class_id(names):
    """Finds the class id whose name is 'background' inside a model.names dict."""
    for cls_id, cls_name in names.items():
        if cls_name.lower() == 'background':
            return cls_id
    raise ValueError("No class named 'background' found in model.names")


def validate_labels_folder(labels_path, image_ids=None, sample_size=None):
    """Structural sanity check on YOLO-seg label .txt files: correct value
    count, complete (x, y) pairs, coordinates within [0, 1]. This is separate
    from the shapely-based contour repair done when the labels were first
    generated - this just catches file-level mistakes before inference runs.

    Returns a list of warning strings (empty if everything looks fine).
    """
    warnings = []
    files = image_ids if image_ids else [f.split('.')[0] for f in os.listdir(labels_path)]
    if sample_size:
        files = files[:sample_size]

    for image_id in files:
        label_file = os.path.join(labels_path, image_id + '.txt')
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            for line_num, line in enumerate(f.readlines()):
                values = line.split()

                if len(values) < 7:  # class_id + at least 3 points (6 coords)
                    warnings.append(f"{image_id} line {line_num}: too few values ({len(values)})")
                    continue

                coords = values[1:]
                if len(coords) % 2 != 0:
                    warnings.append(f"{image_id} line {line_num}: odd number of coordinates")
                    continue

                coord_values = list(map(float, coords))
                if any(c < 0 or c > 1 for c in coord_values):
                    warnings.append(f"{image_id} line {line_num}: coordinates outside [0, 1] range")

    return warnings


def _labels_txt_to_mask(label_path, img_h, img_w):
    """Rasterizes a YOLO-seg label file (class_id x1 y1 x2 y2 ...) into a
    class-id mask of shape (img_h, img_w). Background pixels are -1 so they
    never collide with a real class id (which can legitimately be 0).
    """
    mask = np.full((img_h, img_w), fill_value=-1, dtype=np.int16)

    with open(label_path, 'r') as f:
        for line in f.readlines():
            values = line.split()
            cls_id = int(float(values[0]))
            coords = list(map(float, values[1:]))

            points = np.array(coords).reshape(-1, 2)
            points[:, 0] *= img_w
            points[:, 1] *= img_h
            points = points.astype(np.int32)

            cv2.fillPoly(mask, [points], int(cls_id))

    return mask


def load_ground_truth(image_id, img_h, img_w, masks_path=None, labels_path=None):
    """Loads ground truth from whichever source is available.

    - masks_path: raw ISIC-style binary *_segmentation.png (0/1, no disease
      breakdown - just "is this pixel part of the lesion or not")
    - labels_path: YOLO-seg .txt files (real class ids per polygon) -
      rasterized to a per-pixel class-id mask, background = -1

    Returns (mask, kind) where kind is 'binary', 'multiclass', or None if no
    ground truth was found for this image.
    """
    if masks_path is not None:
        mask_file = os.path.join(masks_path, f"{image_id}_segmentation.png")
        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            return (mask > 127).astype(np.uint8), 'binary'
        return None, None

    if labels_path is not None:
        label_file = os.path.join(labels_path, f"{image_id}.txt")
        if os.path.exists(label_file):
            return _labels_txt_to_mask(label_file, img_h, img_w), 'multiclass'
        return None, None

    return None, None


def check_mask_size(binary_mask, image_id, min_ratio=0.005, max_ratio=0.85):
    """Flags a lesion mask whose area looks unusually small or large relative
    to the whole image. Returns a list of warning strings (possibly empty).
    """
    warnings = []
    img_h, img_w = binary_mask.shape[:2]
    area_ratio = binary_mask.sum() / (img_h * img_w)

    if area_ratio < min_ratio:
        warnings.append(f"{image_id}: lesion area unusually SMALL ({area_ratio:.2%} of image)")
    elif area_ratio > max_ratio:
        warnings.append(f"{image_id}: lesion area unusually LARGE ({area_ratio:.2%} of image)")

    return warnings


def crop_from_mask(image, binary_mask, remove_background=False, padding=0):
    """Crops the image to the bounding box of a binary mask. Optionally
    zeroes out everything outside the mask (background removal). Returns
    None if the mask is empty.
    """
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None

    img_h, img_w = image.shape[:2]
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding, img_w - 1)
    y2 = min(int(ys.max()) + padding, img_h - 1)

    source = image
    if remove_background:
        mask_3ch = np.stack([binary_mask] * 3, axis=-1)
        source = image * mask_3ch

    return source[y1:y2 + 1, x1:x2 + 1]

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


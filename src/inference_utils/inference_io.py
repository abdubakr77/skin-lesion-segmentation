# Path validation, image reading (jpg/png fallback), ground-truth loading
# (auto-detects masks vs YOLO-seg label files from a single path), true-label
# lookup from a dataframe, mask sanity checks, and a reusable crop helper.
#
# Design note on "checking polygon conversion happened correctly": the
# shapely-based contour repair (fixing self-intersecting shapes, recovering
# MultiPolygons, etc.) already happens ONCE, at dataset-build time, when raw
# masks are first converted into polygon label files. By the time inference
# runs, we're reading already-exported label .txt files, not raw contours -
# so what's actually worth re-checking here is the FILE itself: does each
# line have a valid class id, a complete set of (x, y) pairs, and coordinates
# within [0, 1]? That's what validate_labels_folder does.

import os
import re
import cv2
import numpy as np


def check_paths(images_path, ground_truth_path=None):
    """Validates that the given paths exist before doing any work."""
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")

    if ground_truth_path is not None and not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Ground truth path not found: {ground_truth_path}")


def get_background_class_id(names):
    """Finds the class id whose name is 'background' inside a model.names dict."""
    for cls_id, cls_name in names.items():
        if cls_name.lower() == 'background':
            return cls_id
    raise ValueError("No class named 'background' found in model.names")


def find_image_file(images_path, image_id):
    """Returns the full path to an image trying .jpg / .jpeg / .png in turn,
    or None if none of them exist. Used both by read_image() and anywhere
    the pipeline needs to copy the original file (no-detection / low-
    confidence folders) without hardcoding one extension.
    """
    for ext in ('.jpg', '.jpeg', '.png'):
        path = os.path.join(images_path, image_id + ext)
        if os.path.exists(path):
            return path
    return None


def read_image(images_path, image_id):
    """Reads an image trying .jpg / .jpeg / .png in turn, since the dataset
    isn't guaranteed to use one fixed extension. Returns an RGB numpy array.
    Raises FileNotFoundError only if none of the extensions worked.
    """
    path = find_image_file(images_path, image_id)
    if path is not None:
        try:
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    raise FileNotFoundError(
        f"Could not read '{image_id}' as .jpg/.jpeg/.png from {images_path}"
    )


def validate_labels_folder(labels_path, image_ids=None, sample_size=None):
    """Structural sanity check on YOLO-seg label .txt files: correct value
    count, complete (x, y) pairs, coordinates within [0, 1].

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


def load_ground_truth(image_id, img_h, img_w, ground_truth_path):
    """Loads ground truth from ONE path, auto-detecting whether it holds raw
    mask images or YOLO-seg label .txt files - no separate parameter needed
    for each case.

    Detection order: look for a mask file first (any image extension,
    matched by exact prefix so "ISIC_0001" doesn't accidentally match
    "ISIC_00012_x.png"). If nothing matches, fall back to a same-named .txt
    label file. If neither exists, there's simply no ground truth for this
    image - not treated as an error.

    Returns (mask, kind):
        kind is 'binary' (0/1, no class identity - just lesion vs not),
        'multiclass' (-1=background, real class ids elsewhere from the label
        file), or None if nothing was found.
    """
    if ground_truth_path is None:
        return None, None

    mask_matches = [
        f for f in os.listdir(ground_truth_path)
        if (f.startswith(image_id + '_') or f.startswith(image_id + '.'))
        and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if mask_matches:
        mask = cv2.imread(os.path.join(ground_truth_path, mask_matches[0]), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            # ground truth may have different original dimensions than the
            # image actually being processed - resize to match so shapes
            # never mismatch downstream in compute_overlap_metrics. Nearest
            # neighbor keeps the mask strictly binary (no blended edge values).
            if mask.shape[:2] != (img_h, img_w):
                mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            return (mask > 127).astype(np.uint8), 'binary'

    label_file = os.path.join(ground_truth_path, image_id + '.txt')
    if os.path.exists(label_file):
        return _labels_txt_to_mask(label_file, img_h, img_w), 'multiclass'

    return None, None


def get_true_label(df, image_id, image_id_column='image_id', dx_column='dx'):
    """Looks up the ground-truth diagnosis code for an image from a metadata
    dataframe (e.g. the original dx column: 'mel', 'nv', 'akiec', ...).
    Returns None if df wasn't given or the image_id isn't found.
    """
    if df is None:
        return None

    matches = df[df[image_id_column] == image_id]
    if len(matches) == 0:
        return None

    return str(matches.iloc[0][dx_column])


def check_mask_size(binary_mask, image_id, label='mask', min_ratio=0.005, max_ratio=0.85):
    """Flags a mask whose area looks unusually small or large relative to
    the whole image. `label` just tags the warning (e.g. 'true mask' vs
    'predicted mask') so both can be checked and told apart in the log.
    """
    warnings = []
    img_h, img_w = binary_mask.shape[:2]
    area_ratio = binary_mask.sum() / (img_h * img_w)

    if area_ratio < min_ratio:
        warnings.append(f"{image_id}: {label} area unusually SMALL ({area_ratio:.2%} of image)")
    elif area_ratio > max_ratio:
        warnings.append(f"{image_id}: {label} area unusually LARGE ({area_ratio:.2%} of image)")

    return warnings


def crop_from_mask(image, binary_mask, remove_background=False, padding=0):
    """Crops the image to the bounding box of a binary mask. Optionally
    zeroes out everything outside the mask. Returns None if the mask is empty.
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


def compute_overlap_metrics(pred_mask, true_mask):
    """IoU and Dice between two binary masks."""
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()

    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred.sum() + true.sum()) if (pred.sum() + true.sum()) > 0 else 0.0

    return {'iou': float(iou), 'dice': float(dice)}


def normalize_label(name):
    """Strips a leading numeric prefix and lowercases, so '05_NV', 'NV', and
    'nv' all compare equal. Used everywhere a predicted/true label pair needs
    to be checked for a match.
    """
    return re.sub(r'^\d+_', '', str(name)).strip().lower()
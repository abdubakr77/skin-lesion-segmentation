# Two-stage inference pipeline: Stage 1 decides Mel vs Not Mel; anything
# routed as Not Mel goes to Stage 2 to identify the specific disease.
# Handles single image / a list of images / a whole folder, organizes all
# outputs into labeled subfolders, compares against ground truth when
# available, and reports summary statistics at the end.

import os
import shutil
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from src.inference_utils.inference_io import (
    check_paths,
    get_background_class_id,
    validate_labels_folder,
    load_ground_truth,
    check_mask_size,
    crop_from_mask,
    compute_overlap_metrics,
)
from src.inference_utils.inference_visualization import (
    get_class_colors,
    plot_inference_comparison,
)


def _resolve_image_list(images_path, image_names):
    """Figures out which images to process: all of them, one, or a given list."""
    all_ids = [f.split('.')[0] for f in os.listdir(images_path)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if image_names is None:
        return all_ids

    if isinstance(image_names, str):
        image_names = [image_names]

    image_ids = [name.split('.')[0] for name in image_names]
    missing = [n for n in image_ids if n not in all_ids]
    if missing:
        raise FileNotFoundError(f"These image names were not found in images_path: {missing}")

    return image_ids


def _setup_output_dirs(save_dir):
    dirs = {
        'stage1_no_detection': os.path.join(save_dir, 'stage1_model_no_detection'),
        'stage2_no_detection': os.path.join(save_dir, 'stage2_model_no_detection'),
        'mel_viz': os.path.join(save_dir, 'mel_results', 'visualizations'),
        'mel_crop': os.path.join(save_dir, 'mel_results', 'cropped'),
        'disease_viz': os.path.join(save_dir, 'disease_results', 'visualizations'),
        'disease_crop': os.path.join(save_dir, 'disease_results', 'cropped'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

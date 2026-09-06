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


def _run_semantic_model(model, image, background_class_id):
    """Runs one semantic segmentation model on an image.

    Returns (class_masks, names): class_masks is {cls_id: binary_mask} for
    every non-background class found (empty dict if only background), names
    is the model's class-id-to-name dict.
    """
    output = model.predict(image)[0]
    names = output.names

    semantic_mask = output.semantic_mask
    if semantic_mask is None:
        return {}, names

    mask_data = semantic_mask.data.cpu().numpy()
    present_classes = [c for c in np.unique(mask_data) if c != background_class_id]

    class_masks = {int(c): (mask_data == c).astype('uint8') for c in present_classes}
    return class_masks, names


def run_inference_pipeline(stage1_model, stage2_model, images_path,
                            masks_path=None, labels_path=None,
                            image_names=None, save_dir=None, visualize=True,
                            stage1_positive_name='Mel',
                            gt_class_names=None,
                            mask_size_thresholds=(0.005, 0.85)):
    """Runs the full Mel/Not-Mel -> Disease pipeline.

    Args:
        stage1_model: semantic segmentation model with classes Not Mel / Mel / background
        stage2_model: semantic segmentation model with disease classes / background
        images_path: folder of .jpg images to process
        masks_path: optional folder of binary ISIC-style *_segmentation.png ground truth
        labels_path: optional folder of YOLO-seg .txt ground truth (adds per-class GT)
        image_names: None (every image in images_path), a single image id/name,
                     or a list of image ids/names
        save_dir: root output folder. If None, nothing is written to disk.
        visualize: if False, skips all plotting (crops/results are still saved)
        stage1_positive_name: the class name in stage1_model.names meaning "Mel"
        gt_class_names: optional {cls_id: name} mapping to turn label-file class
                        ids into readable names for the classification-accuracy stat
        mask_size_thresholds: (min_ratio, max_ratio) for check_mask_size warnings

    Returns:
        results_df: one row per processed image
        warnings_list: collected warning strings (label file issues, odd mask sizes)
    """
    check_paths(images_path, masks_path, labels_path)
    image_ids = _resolve_image_list(images_path, image_names)

    warnings_list = []
    if labels_path is not None:
        warnings_list += validate_labels_folder(labels_path, image_ids=image_ids)

    dirs = _setup_output_dirs(save_dir) if save_dir else None

    stage1_bg_id = get_background_class_id(stage1_model.names)
    stage2_bg_id = get_background_class_id(stage2_model.names)

    stage1_colors = get_class_colors(stage1_model.names, stage1_bg_id)
    stage2_colors = get_class_colors(stage2_model.names, stage2_bg_id)

    results = []

    for image_id in tqdm(image_ids, desc='Running inference pipeline'):
        image_path = os.path.join(images_path, image_id + '.jpg')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        row = {
            'image_id': image_id, 'stage1_detected': False, 'stage1_prediction': None,
            'stage2_detected': None, 'stage2_prediction': None,
            'true_class_name': None, 'iou': None, 'dice': None,
        }

        gt_mask, gt_kind = load_ground_truth(image_id, img_h, img_w, masks_path, labels_path)
        gt_binary = None
        if gt_mask is not None:
            gt_binary = gt_mask if gt_kind == 'binary' else (gt_mask != -1).astype('uint8')
            warnings_list += check_mask_size(gt_binary, image_id, *mask_size_thresholds)

            if gt_kind == 'multiclass':
                gt_classes, gt_counts = np.unique(gt_mask[gt_mask != -1], return_counts=True)
                if len(gt_classes) > 0:
                    top_gt_cls = int(gt_classes[np.argmax(gt_counts)])
                    row['true_class_name'] = gt_class_names[top_gt_cls] if gt_class_names else str(top_gt_cls)

        # ---- Stage 1 ----
        stage1_masks, stage1_names = _run_semantic_model(stage1_model, image, stage1_bg_id)

        if len(stage1_masks) == 0:
            if save_dir:
                shutil.copy2(image_path, os.path.join(dirs['stage1_no_detection'], image_id + '.jpg'))
            results.append(row)
            continue

        row['stage1_detected'] = True
        top_cls_id = max(stage1_masks, key=lambda c: stage1_masks[c].sum())
        stage1_pred_name = stage1_names[top_cls_id]
        row['stage1_prediction'] = stage1_pred_name

        # ---- Mel branch: final stop ----
        if stage1_pred_name == stage1_positive_name:
            pred_union = np.clip(sum(stage1_masks.values()), 0, 1).astype('uint8')

            metrics = None
            if gt_binary is not None:
                metrics = compute_overlap_metrics(pred_union, gt_binary)
                row['iou'], row['dice'] = metrics['iou'], metrics['dice']

            if save_dir:
                cropped = crop_from_mask(image, pred_union)
                if cropped is not None:
                    cv2.imwrite(os.path.join(dirs['mel_crop'], f'{image_id}.png'),
                                cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            if visualize:
                save_path = os.path.join(dirs['mel_viz'], f'{image_id}.png') if save_dir else None
                plot_inference_comparison(
                    image, stage1_masks, stage1_names, stage1_colors,
                    true_mask=gt_mask, gt_kind=gt_kind, overlap_metrics=metrics,
                    title=f'{image_id} | Stage 1: {stage1_pred_name}', save_path=save_path,
                )

            results.append(row)
            continue

        # ---- Not Mel: pass to Stage 2 ----
        stage2_masks, stage2_names = _run_semantic_model(stage2_model, image, stage2_bg_id)

        if len(stage2_masks) == 0:
            row['stage2_detected'] = False
            if save_dir:
                shutil.copy2(image_path, os.path.join(dirs['stage2_no_detection'], image_id + '.jpg'))
            results.append(row)
            continue

        row['stage2_detected'] = True
        top_cls_id_2 = max(stage2_masks, key=lambda c: stage2_masks[c].sum())
        row['stage2_prediction'] = stage2_names[top_cls_id_2]

        pred_union_2 = np.clip(sum(stage2_masks.values()), 0, 1).astype('uint8')

        metrics = None
        if gt_binary is not None:
            metrics = compute_overlap_metrics(pred_union_2, gt_binary)
            row['iou'], row['dice'] = metrics['iou'], metrics['dice']

        if save_dir:
            cropped = crop_from_mask(image, pred_union_2)
            if cropped is not None:
                cv2.imwrite(os.path.join(dirs['disease_crop'], f'{image_id}.png'),
                            cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        if visualize:
            save_path = os.path.join(dirs['disease_viz'], f'{image_id}.png') if save_dir else None
            plot_inference_comparison(
                image, stage2_masks, stage2_names, stage2_colors,
                true_mask=gt_mask, gt_kind=gt_kind, overlap_metrics=metrics,
                title=f'{image_id} | Stage 2: {row["stage2_prediction"]}', save_path=save_path,
            )

        results.append(row)

    results_df = pd.DataFrame(results)

    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
        if warnings_list:
            with open(os.path.join(save_dir, 'warnings.txt'), 'w') as f:
                f.write('\n'.join(warnings_list))

    return results_df, warnings_list


def compute_pipeline_statistics(results_df):
    """Summarizes detection rates, mask overlap quality, and classification
    accuracy (only computed where ground-truth class ids were available).
    Run this on a test-set results_df to get real evaluation numbers.
    """
    total = len(results_df)
    stage1_detected = results_df['stage1_detected'].sum()

    reached_stage2 = results_df['stage2_detected'].notna().sum()
    stage2_detected = (results_df['stage2_detected'] == True).sum()

    stats = {
        'total_images': total,
        'stage1_detection_rate': stage1_detected / total if total else 0,
        'stage1_no_detection_rate': 1 - (stage1_detected / total) if total else 0,
        'images_reaching_stage2': int(reached_stage2),
        'stage2_detection_rate': (stage2_detected / reached_stage2) if reached_stage2 else None,
        'stage2_no_detection_rate': (1 - stage2_detected / reached_stage2) if reached_stage2 else None,
    }

    overlap_rows = results_df.dropna(subset=['iou', 'dice'])
    if len(overlap_rows) > 0:
        stats['n_images_with_ground_truth'] = len(overlap_rows)
        stats['mean_iou'] = overlap_rows['iou'].mean()
        stats['mean_dice'] = overlap_rows['dice'].mean()

    labeled_rows = results_df.dropna(subset=['true_class_name'])
    if len(labeled_rows) > 0:
        final_pred = labeled_rows['stage2_prediction'].fillna(labeled_rows['stage1_prediction'])
        accuracy = (final_pred.astype(str) == labeled_rows['true_class_name'].astype(str)).mean()
        stats['classification_accuracy'] = accuracy
        stats['confusion_counts'] = pd.crosstab(labeled_rows['true_class_name'], final_pred)

    return stats

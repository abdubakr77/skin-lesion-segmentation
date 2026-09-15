# Two-stage inference pipeline:
#   Stage 1: semantic segmentation model, Mel vs Not Mel (produces a mask).
#   Stage 2: a plain image classifier (e.g. Swin V2 S) over disease classes.
#            It does NOT produce a mask - it classifies the RAW image (no
#            crop, no mask applied). So for anything Stage 1 calls "Not Mel",
#            Stage 1's own mask is kept and simply relabeled with Stage 2's
#            predicted disease name.
#
# Only Stage 1's processed test images are ever read from disk here - Stage
# 2's own per-class image folders are NOT touched. The dataframe (`df`)
# passed in is used purely to look up each image's true diagnosis code (the
# original `dx` column), which supplies ground truth for BOTH stages' title
# correctness checks - it's possible the true dx is 'mel' even though Stage 1
# routed the image to Stage 2 (a false negative); that's shown and scored
# honestly rather than hidden.
#
# Ground truth for the mask panels (IoU/Dice, True Mask) comes from a single
# `ground_truth_path`, auto-detecting whether it holds mask images or YOLO-seg
# label files - see load_ground_truth() for how that's resolved.

import os
import shutil
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image as PILImage
from tqdm import tqdm

from src.inference_utils.inference_io import (
    check_paths,
    get_background_class_id,
    read_image,
    find_image_file,
    validate_labels_folder,
    load_ground_truth,
    get_true_label,
    check_mask_size,
    crop_from_mask,
    compute_overlap_metrics,
    normalize_label,
)
from src.inference_utils.inference_visualization import (
    get_class_colors,
    get_disease_color,
    plot_inference_comparison,
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _resolve_image_list(images_path, image_names):
    """Decides which image(s) to process:
        None            -> one random image
        'all'           -> every image in images_path
        a string        -> that one specific image
        a list          -> exactly those images
    """
    all_ids = sorted({
        os.path.splitext(f)[0] for f in os.listdir(images_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    })

    if image_names is None:
        return [str(np.random.choice(all_ids))]

    if image_names == 'all':
        return all_ids

    if isinstance(image_names, str):
        image_names = [image_names]

    image_ids = [os.path.splitext(name)[0] for name in image_names]
    missing = [n for n in image_ids if n not in all_ids]
    if missing:
        raise FileNotFoundError(f"These image names were not found in images_path: {missing}")

    return image_ids


def _setup_output_dirs(save_dir):
    dirs = {
        'stage1_no_detection': os.path.join(save_dir, 'stage1_model_no_detection'),
        'stage2_low_confidence': os.path.join(save_dir, 'stage2_model_low_confidence'),
        'mel_viz': os.path.join(save_dir, 'mel_results', 'visualizations'),
        'mel_crop': os.path.join(save_dir, 'mel_results', 'cropped'),
        'disease_viz': os.path.join(save_dir, 'disease_results', 'visualizations'),
        'disease_crop': os.path.join(save_dir, 'disease_results', 'cropped'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _validate_df(df, image_id_column, dx_column):
    if df is None:
        return
    missing = [c for c in (image_id_column, dx_column) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required column(s): {missing}")


def _run_semantic_model(model, image, background_class_id, verbose=False):
    """Runs Stage 1's semantic segmentation model on an image.
    Returns (class_masks, names): class_masks is {cls_id: binary_mask} for
    every non-background class found (empty dict if only background).
    """
    output = model.predict(image, verbose=verbose)[0]
    names = output.names

    semantic_mask = output.semantic_mask
    if semantic_mask is None:
        return {}, names

    mask_data = semantic_mask.data.cpu().numpy()
    present_classes = [c for c in np.unique(mask_data) if c != background_class_id]

    class_masks = {int(c): (mask_data == c).astype('uint8') for c in present_classes}
    return class_masks, names


def _run_classifier(model, image, transform, class_names, device):
    """Runs Stage 2's plain image classifier on a single RGB numpy image
    (H, W, 3), full and uncropped. Returns (predicted_class_name, confidence)
    - no mask, since classification doesn't localize anything on its own.
    """
    model.eval()
    pil_image = PILImage.fromarray(image)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    return class_names[int(pred_idx.item())], float(confidence.item())


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------

def run_inference_pipeline(stage1_model, stage2_model, stage2_transform, stage2_class_names,
                            images_path, ground_truth_path=None, df=None,
                            image_id_column='image_id', dx_column='dx',
                            image_names=None, save_dir=None, visualize=True,
                            stage1_positive_name='Mel', device=None,
                            stage2_confidence_threshold=None,
                            mask_size_thresholds=(0.005, 0.85)):
    """Runs Stage 1 (Mel vs Not Mel segmentation); anything Not Mel is handed
    to Stage 2 (a plain classifier) to name the specific disease, reusing
    Stage 1's mask rather than producing a new one.

    Args:
        stage1_model: semantic segmentation model, classes Not Mel / Mel / background
        stage2_model: a plain torch classifier (e.g. Swin V2 S) over disease classes
        stage2_transform: the exact preprocessing transform stage2_model was
                           trained with
        stage2_class_names: list of class names in the SAME order as the
                             classifier's output logits, e.g. an ImageFolder's
                             .classes: ['01_AKIEC', '02_BCC', '03_BKL', '04_DF', '05_NV', '06_VASC']
        images_path: folder of Stage 1's processed TEST images - used for
                     BOTH stages, Stage 2's own per-class folders are never read
        ground_truth_path: optional single folder holding either binary mask
                            images or YOLO-seg .txt label files (auto-detected
                            per image) - used for the True Mask panel and IoU/Dice
        df: optional dataframe with a true diagnosis code per image (the
            original `dx` column: 'mel', 'nv', 'akiec', ...) - this is what
            drives the True/Pred correctness coloring at BOTH stages, since
            it's possible for the true dx to be 'mel' even when Stage 1 sent
            the image to Stage 2 (a false negative) - that case is scored
            honestly as a mismatch, not hidden
        image_id_column, dx_column: column names to use when reading df
        image_names: None (one random image), 'all' (every image in
                     images_path), a single image id/name, or a list of them
        save_dir: root output folder. If None, nothing is written to disk.
        visualize: if False, skips all plotting (crops/results still saved)
        stage1_positive_name: the class name in stage1_model.names meaning "Mel"
        device: torch device for the classifier. Defaults to cuda if available.
        stage2_confidence_threshold: if set, any Stage 2 prediction below this
                                      confidence is routed to a
                                      'stage2_model_low_confidence' folder
                                      instead of being reported as a firm
                                      diagnosis (added since a classifier
                                      always returns *some* class via argmax,
                                      so it can't have a genuine "no
                                      detection" case the way Stage 1 can -
                                      this is the closest honest equivalent)
        mask_size_thresholds: (min_ratio, max_ratio) for check_mask_size warnings

    Returns:
        results_df: one row per processed image
        warnings_list: collected warning strings
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stage2_model = stage2_model.to(device)

    check_paths(images_path, ground_truth_path)
    _validate_df(df, image_id_column, dx_column)

    image_ids = _resolve_image_list(images_path, image_names)

    warnings_list = []
    if ground_truth_path is not None:
        txt_files = [f for f in os.listdir(ground_truth_path) if f.lower().endswith('.txt')]
        if txt_files:
            warnings_list += validate_labels_folder(ground_truth_path, image_ids=image_ids)

    dirs = _setup_output_dirs(save_dir) if save_dir else None

    stage1_bg_id = get_background_class_id(stage1_model.names)
    stage1_colors = get_class_colors(stage1_model.names, stage1_bg_id)

    results = []

    for image_id in tqdm(image_ids, desc='Running inference pipeline'):
        image = read_image(images_path, image_id)
        img_h, img_w = image.shape[:2]

        row = {
            'image_id': image_id,
            'true_dx': get_true_label(df, image_id, image_id_column, dx_column),
            'stage1_detected': False, 'stage1_prediction': None, 'true_class_stage1': None,
            'iou': None, 'dice': None,
            'stage2_prediction': None, 'stage2_confidence': None,
            'final_prediction': None,
        }

        if row['true_dx'] is not None:
            row['true_class_stage1'] = stage1_positive_name if row['true_dx'].lower() == 'mel' else 'Not Mel'

        gt_mask, gt_kind = load_ground_truth(image_id, img_h, img_w, ground_truth_path)
        gt_binary = None
        if gt_mask is not None:
            gt_binary = gt_mask if gt_kind == 'binary' else (gt_mask != -1).astype('uint8')
            warnings_list += check_mask_size(
                gt_binary, image_id, label='true mask',
                min_ratio=mask_size_thresholds[0], max_ratio=mask_size_thresholds[1],
            )

        # ---- Stage 1 ----
        stage1_masks, stage1_names = _run_semantic_model(stage1_model, image, stage1_bg_id, verbose=visualize)

        if len(stage1_masks) == 0:
            if save_dir:
                src_path = find_image_file(images_path, image_id)
                shutil.copy2(src_path, os.path.join(dirs['stage1_no_detection'], os.path.basename(src_path)))
            results.append(row)
            continue

        row['stage1_detected'] = True
        top_cls_id = max(stage1_masks, key=lambda c: stage1_masks[c].sum())
        stage1_pred_name = stage1_names[top_cls_id]
        row['stage1_prediction'] = stage1_pred_name

        pred_union = np.clip(sum(stage1_masks.values()), 0, 1).astype('uint8')
        warnings_list += check_mask_size(
            pred_union, image_id, label='predicted mask',
            min_ratio=mask_size_thresholds[0], max_ratio=mask_size_thresholds[1],
        )

        metrics = None
        if gt_binary is not None:
            metrics = compute_overlap_metrics(pred_union, gt_binary)
            row['iou'], row['dice'] = metrics['iou'], metrics['dice']

        # ---- Mel branch: final stop ----
        if stage1_pred_name == stage1_positive_name:
            row['final_prediction'] = stage1_pred_name

            if save_dir:
                cropped = crop_from_mask(image, pred_union)
                if cropped is not None:
                    cv2.imwrite(os.path.join(dirs['mel_crop'], f'{image_id}.png'),
                                cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

            if visualize:
                save_path = os.path.join(dirs['mel_viz'], f'{image_id}.png') if save_dir else None
                plot_inference_comparison(
                    image, stage1_masks, stage1_names, stage1_colors,
                    true_mask=gt_mask, gt_kind=gt_kind,
                    true_label=row['true_class_stage1'], pred_label=stage1_pred_name,
                    overlap_metrics=metrics,
                    title=f'{image_id} | Stage 1', save_path=save_path,
                )

            results.append(row)
            continue

        # ---- Not Mel: classify the specific disease on the RAW image ----
        disease_name, confidence = _run_classifier(
            stage2_model, image, stage2_transform, stage2_class_names, device
        )
        row['stage2_confidence'] = confidence

        if stage2_confidence_threshold is not None and confidence < stage2_confidence_threshold:
            if save_dir:
                src_path = find_image_file(images_path, image_id)
                shutil.copy2(src_path, os.path.join(dirs['stage2_low_confidence'], os.path.basename(src_path)))
            results.append(row)
            continue

        row['stage2_prediction'] = disease_name
        row['final_prediction'] = disease_name
        clean_name = normalize_label(disease_name).upper()

        # relabel Stage 1's retained mask with the disease name
        display_masks = {0: pred_union}
        display_names = {0: clean_name}
        display_colors = {0: get_disease_color(disease_name)}

        if save_dir:
            cropped = crop_from_mask(image, pred_union)
            if cropped is not None:
                cv2.imwrite(os.path.join(dirs['disease_crop'], f'{image_id}.png'),
                            cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

        if visualize:
            save_path = os.path.join(dirs['disease_viz'], f'{image_id}.png') if save_dir else None
            plot_inference_comparison(
                image, display_masks, display_names, display_colors,
                true_mask=gt_mask, gt_kind=gt_kind,
                true_label=row['true_dx'], pred_label=disease_name,
                overlap_metrics=metrics,
                title=f'{image_id} | Stage 2 (conf {confidence:.2f})', save_path=save_path,
            )

        results.append(row)

    results_df = pd.DataFrame(results)

    if save_dir:
        results_df.to_csv(os.path.join(save_dir, 'results.csv'), index=False)
        if warnings_list:
            with open(os.path.join(save_dir, 'warnings.txt'), 'w') as f:
                f.write('\n'.join(warnings_list))

    return results_df, warnings_list


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def compute_pipeline_statistics(results_df):
    """Summarizes how well the pipeline did on a test set:
        - Stage 1 detection rate (found a lesion at all)
        - Stage 1 Mel/Not-Mel accuracy + confusion counts (needs df-based truth)
        - Mean IoU/Dice against ground-truth masks (needs ground_truth_path)
        - Stage 2 disease classification accuracy + confusion counts (needs df)
        - End-to-end accuracy: does the pipeline's FINAL answer (Mel, or the
          specific disease from Stage 2) match the true dx overall? This
          naturally penalizes false negatives (true dx = 'mel' but Stage 1
          said Not Mel), since Stage 2 can never predict 'mel' itself.
        - A simple distribution of what Stage 2 predicted, for a sanity check
    """
    total = len(results_df)
    stage1_detected = results_df['stage1_detected'].sum()

    stats = {
        'total_images': total,
        'stage1_detection_rate': stage1_detected / total if total else 0,
        'stage1_no_detection_rate': 1 - (stage1_detected / total) if total else 0,
    }

    overlap_rows = results_df.dropna(subset=['iou', 'dice'])
    if len(overlap_rows) > 0:
        stats['n_images_with_ground_truth_mask'] = len(overlap_rows)
        stats['mean_iou'] = overlap_rows['iou'].mean()
        stats['mean_dice'] = overlap_rows['dice'].mean()

    s1_rows = results_df.dropna(subset=['true_class_stage1', 'stage1_prediction'])
    if len(s1_rows) > 0:
        acc = (s1_rows['stage1_prediction'] == s1_rows['true_class_stage1']).mean()
        stats['stage1_classification_accuracy'] = acc
        stats['stage1_confusion_counts'] = pd.crosstab(s1_rows['true_class_stage1'], s1_rows['stage1_prediction'])

    s2_rows = results_df.dropna(subset=['true_dx', 'stage2_prediction'])
    if len(s2_rows) > 0:
        correct = s2_rows.apply(
            lambda r: normalize_label(r['stage2_prediction']) == normalize_label(r['true_dx']), axis=1
        )
        stats['stage2_classification_accuracy'] = correct.mean()
        stats['stage2_confusion_counts'] = pd.crosstab(
            s2_rows['true_dx'].apply(normalize_label), s2_rows['stage2_prediction'].apply(normalize_label)
        )

    e2e_rows = results_df.dropna(subset=['true_dx', 'final_prediction'])
    if len(e2e_rows) > 0:
        e2e_correct = e2e_rows.apply(
            lambda r: normalize_label(r['final_prediction']) == normalize_label(r['true_dx']), axis=1
        )
        stats['end_to_end_accuracy'] = e2e_correct.mean()
        stats['n_images_scored_end_to_end'] = len(e2e_rows)

    disease_counts = results_df['stage2_prediction'].value_counts()
    if len(disease_counts) > 0:
        stats['stage2_prediction_distribution'] = disease_counts.to_dict()

    return stats
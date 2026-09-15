# Visualization for the inference pipeline: predicted mask alone, true mask
# alone, image+predicted overlay, and predicted-vs-true overlay, with
# consistent per-class colors and an IoU/Dice comparison box.
#
# The SAME plotting function is used for both stages - Stage 1 passes its own
# semantic-model class dict/colors; Stage 2 passes a one-entry "display" dict
# (Stage 1's retained mask, relabeled with the classified disease name and a
# color from the fixed disease palette below). Title correctness (green/red)
# is driven by explicit true_label/pred_label strings the pipeline resolves
# beforehand - not by comparing mask class ids - so the same logic works
# whether the truth came from a mask's class id or a metadata dataframe.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from src.inference_utils.inference_io import normalize_label


DISEASE_COLOR_PALETTE = {
    'akiec': (0.121, 0.466, 0.705, 1.0),
    'bcc':   (1.000, 0.498, 0.055, 1.0),
    'bkl':   (0.173, 0.627, 0.173, 1.0),
    'df':    (0.839, 0.153, 0.157, 1.0),
    'nv':    (0.580, 0.404, 0.741, 1.0),
    'vasc':  (0.549, 0.337, 0.294, 1.0),
}


def get_disease_color(name):
    """Consistent color per disease class name, matched after stripping any
    numeric prefix and lowercasing (so '05_NV', 'NV', and 'nv' all resolve to
    the same color). Falls back to a deterministic tab10 color for any name
    outside the known palette, so this never breaks on an unexpected class.
    """
    key = normalize_label(name)
    if key in DISEASE_COLOR_PALETTE:
        return DISEASE_COLOR_PALETTE[key]

    cmap = plt.cm.get_cmap('tab10', 10)
    return cmap(hash(key) % 10)


def get_class_colors(class_names, background_id):
    """Assigns one consistent color per non-background class id (used for
    Stage 1's own Mel / Not Mel classes).
    """
    non_bg = [c for c in class_names if c != background_id]
    cmap = plt.cm.get_cmap('tab10', max(len(non_bg), 1))
    return {cls_id: cmap(i) for i, cls_id in enumerate(non_bg)}


def _colored_overlay(binary_mask, color, alpha):
    overlay = np.zeros((*binary_mask.shape, 4))
    overlay[binary_mask == 1] = (*color[:3], alpha)
    return overlay


def plot_inference_comparison(image, pred_masks, class_names, colors,
                               true_mask=None, gt_kind=None,
                               true_label=None, pred_label=None,
                               overlap_metrics=None, title='', save_path=None):
    """4-panel comparison figure: predicted mask alone, true mask alone,
    image + predicted overlay, and predicted + true overlaid together.

    Args:
        image: RGB image array
        pred_masks: dict {cls_id: binary_mask} of predicted classes present.
                    For Stage 2's disease branch this is a one-entry display
                    dict (e.g. {0: retained_stage1_mask}) built by the
                    pipeline - this function only cares about shapes here.
        class_names: dict {cls_id: name} for the panel-3 legend
        colors: dict {cls_id: color}, matching pred_masks' keys
        true_mask: ground truth mask, or None if unavailable
        gt_kind: 'binary' (0/1, no class info) or 'multiclass' (-1=background,
                 real class ids elsewhere) - tells us how to read true_mask
        true_label / pred_label: plain strings for the title's correctness
                                  check (e.g. 'Mel'/'Not Mel', or a dx code
                                  like 'nv'). Compared after normalize_label()
                                  so prefixes/case never cause a false
                                  mismatch. If either is None, the title is
                                  shown plain with no color coding.
        overlap_metrics: optional dict {'iou': ..., 'dice': ...} shown as a
                         comparison box under the figure
        title: figure title (the true/pred label summary is appended to this)
        save_path: if given, saves the figure to this path
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: predicted mask alone
    axes[0].set_title('Predicted Mask')
    axes[0].imshow(np.zeros_like(image))
    for cls_id, mask in pred_masks.items():
        axes[0].imshow(_colored_overlay(mask, colors[cls_id], alpha=1.0))
    axes[0].axis('off')

    # Panel 2: true mask alone
    axes[1].set_title('True Mask')
    axes[1].imshow(np.zeros_like(image))
    if true_mask is not None:
        if gt_kind == 'binary':
            axes[1].imshow(_colored_overlay(true_mask, (1.0, 1.0, 1.0), alpha=1.0))
        else:
            for cls_id in np.unique(true_mask):
                if cls_id == -1:
                    continue
                color = colors.get(int(cls_id), (1.0, 1.0, 1.0, 1.0))
                axes[1].imshow(_colored_overlay((true_mask == cls_id).astype('uint8'), color, alpha=1.0))
    else:
        axes[1].text(0.5, 0.5, 'No Ground Truth', ha='center', va='center',
                     transform=axes[1].transAxes, color='white', fontsize=12)
    axes[1].axis('off')

    # Panel 3: image + predicted overlay
    axes[2].set_title('Image + Predicted')
    axes[2].imshow(image)
    legend_patches = []
    for cls_id, mask in pred_masks.items():
        axes[2].imshow(_colored_overlay(mask, colors[cls_id], alpha=0.45))
        legend_patches.append(patches.Patch(color=colors[cls_id], label=class_names[cls_id]))
    if legend_patches:
        axes[2].legend(handles=legend_patches, loc='upper right', fontsize=8)
    axes[2].axis('off')

    # Panel 4: predicted vs true, overlaid together
    axes[3].set_title('Predicted vs True')
    axes[3].imshow(image)
    for cls_id, mask in pred_masks.items():
        axes[3].imshow(_colored_overlay(mask, colors[cls_id], alpha=0.4))
    if true_mask is not None:
        if gt_kind == 'binary':
            axes[3].contour(true_mask, colors='white', linewidths=1.5)
        else:
            for cls_id in np.unique(true_mask):
                if cls_id == -1:
                    continue
                axes[3].contour((true_mask == cls_id).astype('uint8'), colors='white', linewidths=1.5)
    axes[3].axis('off')

    # ---- Title: plain, or "True: X | Pred: Y" colored green/red ----
    full_title = title
    suptitle_color = 'black'

    if true_label is not None and pred_label is not None:
        is_correct = normalize_label(true_label) == normalize_label(pred_label)
        full_title = f"{title} | True: {true_label} | Pred: {pred_label}"
        suptitle_color = '#16a34a' if is_correct else '#dc2626'

    fig.suptitle(full_title, color=suptitle_color, fontweight='bold')

    if overlap_metrics:
        text = f"IoU: {overlap_metrics['iou']:.3f}  |  Dice: {overlap_metrics['dice']:.3f}"
        fig.text(0.5, 0.02, text, ha='center', fontsize=12,
                  bbox=dict(facecolor='lightgray', alpha=0.6, boxstyle='round'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close(fig)
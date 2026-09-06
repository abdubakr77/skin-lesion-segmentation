# Visualization for the inference pipeline: predicted mask alone, true mask
# alone, image+predicted overlay, and predicted-vs-true overlay, with
# consistent per-class colors and an IoU/Dice comparison box.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def get_class_colors(class_names, background_id):
    """Assigns one consistent color per non-background class name."""
    non_bg = [c for c in class_names if c != background_id]
    cmap = plt.cm.get_cmap('tab10', max(len(non_bg), 1))
    return {cls_id: cmap(i) for i, cls_id in enumerate(non_bg)}


def _colored_overlay(binary_mask, color, alpha):
    overlay = np.zeros((*binary_mask.shape, 4))
    overlay[binary_mask == 1] = (*color[:3], alpha)
    return overlay


def plot_inference_comparison(image, pred_masks, class_names, colors,
                               true_mask=None, gt_kind=None,
                               overlap_metrics=None, title='', save_path=None):
    """4-panel comparison figure.

    Args:
        image: RGB image array
        pred_masks: dict {cls_id: binary_mask} of predicted classes present
        class_names: model.names dict (for legend labels)
        colors: dict {cls_id: color}, from get_class_colors()
        true_mask: ground truth mask, or None if unavailable
        gt_kind: 'binary' (0/1, no class info) or 'multiclass' (-1=background,
                 real class ids elsewhere) - tells us how to read true_mask
        overlap_metrics: optional dict {'iou': ..., 'dice': ...} shown as a
                         comparison box under the figure
        title: figure title
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

    fig.suptitle(title)

    if overlap_metrics:
        text = f"IoU: {overlap_metrics['iou']:.3f}  |  Dice: {overlap_metrics['dice']:.3f}"
        fig.text(0.5, 0.02, text, ha='center', fontsize=12,
                  bbox=dict(facecolor='lightgray', alpha=0.6, boxstyle='round'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
    plt.close(fig)

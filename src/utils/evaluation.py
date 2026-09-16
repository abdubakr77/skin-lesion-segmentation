# Helpers for loading training results and visualizing evaluation outputs
# across both stages.
#
# Stage 1 (Ultralytics semantic segmentation) writes a real results.csv per
# run, so load_results() and plot_train_val_curves() work with it directly.
#
# Stage 2 (the Swin V2 S classifier) has no results.csv - training only saved
# one combined loss/accuracy curve image, plus a confusion matrix image and
# the test_set_predicted.png sample grid. For Stage 2, show_saved_image() is
# the only function needed: it just displays whichever saved PNG you point
# it at, and doesn't care which stage that PNG came from.

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_results(log_dir):
    """Loads results.csv for a single Stage 1 (Ultralytics) training run.
    Stage 2 has no results.csv - don't call this for Stage 2 runs.
    """
    csv_path = os.path.join(log_dir, 'results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No results.csv found at {csv_path}")
    return pd.read_csv(csv_path)


def plot_train_val_curves(df, metrics, title=''):
    """Plots one or more train/val metric pairs across epochs, for a Stage 1
    results DataFrame (from load_results()).

    Args:
        df: DataFrame from load_results()
        metrics: list of base metric names, e.g. ['dice_loss', 'ce_loss']
                 expects matching 'train/<metric>' and 'val/<metric>' columns
        title: overall figure title
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))

    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_col = f'train/{metric}'
        val_col = f'val/{metric}'

        if train_col in df.columns:
            ax.plot(df['epoch'], df[train_col], label='train')
        if val_col in df.columns:
            ax.plot(df['epoch'], df[val_col], label='val')

        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_metric_comparison_bar(results_dict, metric_col):
    """Bar chart comparing the best value of a metric across multiple Stage 1
    runs (e.g. different training configs), all sharing the same metric
    column such as 'metrics/mIoU'. This only makes sense within Stage 1 -
    Stage 2 has no comparable numeric column to plot alongside it, since it
    isn't a segmentation model and has no results.csv.

    Args:
        results_dict: dict of run_name -> DataFrame (from load_results()),
                       all Stage 1 runs
        metric_col: column name to compare, e.g. 'metrics/mIoU'
    """
    names = list(results_dict.keys())
    best_values = [df[metric_col].max() for df in results_dict.values()]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, best_values)

    for bar, value in zip(bars, best_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.3f}',
                ha='center', va='bottom')

    ax.set_ylabel(metric_col)
    ax.set_title(f'Best {metric_col} Comparison')
    plt.tight_layout()
    plt.show()


def show_saved_image(image_path, title=''):
    """Displays a single saved image file. Works for either stage: Stage 1's
    confusion_matrix.png / results.png, or Stage 2's loss-and-accuracy curve
    image, confusion matrix, and test_set_predicted.png sample grid.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No image found at {image_path}")

    img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
# Helpers for loading training results and visualizing evaluation outputs
# (results.csv, confusion matrices, sample result images) across experiments.

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_results(log_dir):
    """Loads results.csv for a single training run into a DataFrame."""
    csv_path = os.path.join(log_dir, 'results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No results.csv found at {csv_path}")
    return pd.read_csv(csv_path)


def plot_train_val_curves(df, metrics, title=''):
    """Plots one or more train/val metric pairs across epochs.

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
    """Bar chart comparing the best value of a metric across multiple runs.

    Args:
        results_dict: dict of run_name -> DataFrame (from load_results())
        metric_col: column name to compare, e.g. 'metrics/mIoU'
    """
    names = list(results_dict.keys())
    best_values = [df[metric_col].max() for df in results_dict.values()]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, best_values, color=['#4C72B0', '#DD8452'])

    for bar, value in zip(bars, best_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.3f}',
                ha='center', va='bottom')

    ax.set_ylabel(metric_col)
    ax.set_title(f'Best {metric_col} Comparison')
    plt.tight_layout()
    plt.show()


def show_saved_image(image_path, title=''):
    """Displays a single saved image file (e.g. confusion_matrix.png, results.png)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No image found at {image_path}")

    img = mpimg.imread(image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

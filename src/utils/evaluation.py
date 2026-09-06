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


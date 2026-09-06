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



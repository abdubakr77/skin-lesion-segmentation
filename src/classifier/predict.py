import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import RandomSampler

from .transforms import denorm


def _hard_label(label):
    """Collapses a soft/one-hot label to its class index; passes hard labels through."""
    if torch.is_tensor(label) and label.ndim > 0 and label.numel() > 1:
        return int(label.argmax().item())
    return int(label)


def predict_classifier(
    model,
    dataloader,
    device,
    class_names,
    disease_class_idx=None,
    threshold=None,
    show_plot=False,
    num_samples=20,
    save_plot_path=os.getcwd(),
    figsize=(12, 10),
    return_probs=False,
):
    """
    Runs the model over a dataloader and returns true/predicted labels.

    Args:
        model, dataloader, device, class_names: as usual
        disease_class_idx: index of a "critical" class in class_names (e.g. BCC
                            or AKIEC) - required if `threshold` is set.
        threshold: if set, predicts disease_class_idx whenever its probability
                   exceeds this value; otherwise falls back to the argmax over
                   the REMAINING classes. This works for any number of classes
                   (not just binary) - lower than 1/num_classes favors catching
                   more of that class at the cost of more false positives.
        show_plot: if True, saves/shows a grid of sample predictions. Requires
                   `dataloader` to be built with shuffle=False so batch order
                   matches dataset order (checked below).
        return_probs: if True, also returns a list of per-image class-probability
                      lists (all_probs), useful for confidence charts downstream.

    Returns:
        all_labels, all_preds  (and all_probs if return_probs=True)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            hard_labels = labels.argmax(dim=1) if labels.ndim > 1 else labels

            if threshold is not None:
                if disease_class_idx is None:
                    raise ValueError("disease_class_idx is required when threshold is set.")
                disease_prob = probs[:, disease_class_idx]
                # argmax over every class EXCEPT disease_class_idx, so this
                # generalizes past the binary "disease vs not" case
                other_probs = probs.clone()
                other_probs[:, disease_class_idx] = -1.0
                other_preds = other_probs.argmax(dim=1)
                preds = torch.where(
                    disease_prob > threshold,
                    torch.full_like(other_preds, disease_class_idx),
                    other_preds,
                )
            else:
                preds = probs.argmax(dim=1)

            all_labels.extend(hard_labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())
            if return_probs:
                all_probs.extend(probs.detach().cpu().tolist())

    if show_plot:
        if isinstance(getattr(dataloader, "sampler", None), RandomSampler):
            raise ValueError(
                "show_plot requires a non-shuffled dataloader (shuffle=False), "
                "otherwise sample predictions won't line up with the images shown."
            )

        dataset = getattr(dataloader, "dataset", None)
        if dataset is None:
            raise ValueError("dataloader must provide a dataset attribute for plotting")

        if class_names is None:
            class_names = getattr(dataset, "classes", [str(i) for i in sorted(set(all_labels))])

        sample_size = min(num_samples, len(dataset))
        sample_indices = np.random.choice(len(dataset), size=sample_size, replace=False)

        n_cols = 5
        n_rows = int(np.ceil(sample_size / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for ax, idx in zip(axes, sample_indices):
            img, true_label = dataset[idx]
            true_idx = _hard_label(true_label)
            pred_idx = int(all_preds[idx])  # assumes shuffle=False, checked above
            correct = pred_idx == true_idx

            img = denorm(img).permute(1, 2, 0).cpu().numpy()

            ax.imshow(img)
            ax.set_title(
                f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]}",
                color="green" if correct else "red",
                fontsize=9,
            )
            ax.axis("off")

        for ax in axes[sample_size:]:
            ax.axis("off")

        plt.suptitle("Test Set Predictions", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(save_plot_path, "test_set_predicted.png"), dpi=300, bbox_inches="tight")
        plt.show()

    if return_probs:
        return all_labels, all_preds, all_probs
    return all_labels, all_preds
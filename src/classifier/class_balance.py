import os

import torch


def compute_class_weights(samples_per_class, method='inverse'):
    """
    Computes per-class weights from class sample counts, for use with
    nn.CrossEntropyLoss(weight=...), FocalLoss(alpha=...), or LDAMLoss(weight=...).

    Args:
        samples_per_class: list/tensor of sample counts per class, in class-index
                            order (e.g. [NV_count, BKL_count, BCC_count, ...])
        method: 'inverse'   -> classic 1/count, normalized
                'effective' -> Class-Balanced "effective number of samples"
                               weighting (Cui et al., 2019) with beta=0.999.
                               Usually works better than plain inverse-frequency
                               when a couple of classes are extremely rare
                               (like your DF/VASC vs NV imbalance) since raw
                               inverse-frequency can overweight rare classes
                               too aggressively and destabilize training.

    Returns:
        torch.Tensor of shape (num_classes,), weights normalized to sum to
        num_classes so the effective learning rate scale stays comparable
        to unweighted training.
    """
    samples_per_class = torch.as_tensor(samples_per_class, dtype=torch.float32)

    if method == 'inverse':
        weights = 1.0 / samples_per_class
    elif method == 'effective':
        beta = 0.999
        effective_num = 1.0 - torch.pow(torch.tensor(beta), samples_per_class)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unsupported method='{method}', use 'inverse' or 'effective'")

    return weights / weights.sum() * len(samples_per_class)


def suggest_n_copies(
    data_yaml,
    class_names=None,
    dataset_type='segmentation',
    close_ratio_threshold=3.0,
    dampen_power=0.5,
    max_copies=10,
):
    """
    Suggests how many augmented copies to generate per class from raw counts -
    WITHOUT fully equalizing every class to the largest one. Fully equalizing
    via augmentation AND then also applying class weights on top double-
    corrects the same imbalance, which is what you flagged.

    How it decides:
      - Classes reasonably close to the largest class (ratio to the largest
        <= close_ratio_threshold) get fully equalized via augmentation - it's
        cheap and low-risk to close a small gap this way.
      - Classes far from it (ratio > close_ratio_threshold, e.g. DF/VASC) get
        a DAMPENED number of copies instead of the full ratio, capped at
        max_copies, so a class with very few originals doesn't end up as
        dozens of near-duplicate images. Whatever gap is left after this is
        meant to be picked up by loss weights - see `residual_loss_weights`
        below, which computes weights from the counts AFTER this suggested
        augmentation (not the raw counts), so the two don't double-correct
        for the same imbalance.

    Args:
        data_yaml, class_names, dataset_type: same as before
        close_ratio_threshold: classes within this many times of the largest
            class get equalized fully via augmentation (ratio <= threshold)
        dampen_power: for classes beyond the threshold, the extra distance
            past the threshold is raised to this power before being turned
            into copies (0 < dampen_power <= 1). 0.5 = sqrt dampening
            (moderate); lower = gentler augmentation with more of the
            remaining gap left for loss weights to handle; 1.0 = no
            dampening (same as full equalization for every class)
        max_copies: hard cap on n_copies for any single class, regardless of
            how rare it is, to avoid excessive duplication of a tiny class

    Returns:
        counts: {class: raw_count}
        suggestions: {class: n_copies}              -- same shape as before
        projected_counts: {class: raw_count * (n_copies + 1)}
            the counts you'll actually end up with after applying
            `suggestions` - feed this into residual_loss_weights()
    """
    counts = {}

    if dataset_type == 'segmentation':
        labels_path = data_yaml['train'].replace('images', 'labels')
        all_files_no_ext = [f.split('.')[0] for f in os.listdir(data_yaml['train'])]

        for fname in all_files_no_ext:
            label_path = os.path.join(labels_path, fname + '.txt')

            if not os.path.exists(label_path):
                continue

            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls_id = int(float(line.split()[0]))
                    counts[cls_id] = counts.get(cls_id, 0) + 1

    elif dataset_type == 'classifier':
        train_path = data_yaml['train']

        for class_name in os.listdir(train_path):
            class_path = os.path.join(train_path, class_name)

            if not os.path.isdir(class_path):
                continue

            counts[class_name] = sum(
                os.path.isfile(os.path.join(class_path, fname))
                for fname in os.listdir(class_path)
            )

    else:
        raise ValueError(
            f"Invalid dataset_type: {dataset_type}. "
            f"Use 'segmentation' or 'classifier'."
        )

    if not counts:
        print("No classes found.")
        return {}, {}, {}

    max_count = max(counts.values())

    suggestions = {}
    projected_counts = {}
    for cls_id, count in counts.items():
        if not count:
            suggestions[cls_id] = 0
            projected_counts[cls_id] = 0
            continue

        ratio = max_count / count

        if ratio <= close_ratio_threshold:
            # close enough to the majority class - fully equalize, it's cheap
            effective_ratio = ratio
        else:
            # far away - dampen the portion of the ratio beyond the threshold,
            # so rare classes get a meaningful (not absurd) copy count and the
            # remaining gap is left for loss weights to cover instead
            effective_ratio = close_ratio_threshold * (ratio / close_ratio_threshold) ** dampen_power

        n_copies = max(0, round(effective_ratio) - 1)
        n_copies = min(n_copies, max_copies)

        suggestions[cls_id] = n_copies
        projected_counts[cls_id] = int(count * (n_copies + 1))

    if class_names:
        counts = {class_names[k]: v for k, v in counts.items()}
        suggestions = {class_names[k]: v for k, v in suggestions.items()}
        projected_counts = {class_names[k]: v for k, v in projected_counts.items()}

    return counts, suggestions, projected_counts

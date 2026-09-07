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


class ClassDistributionPlanner:
    """
    Turns raw per-class counts into two consistent, ready-to-use outputs:
      1. how many samples each class should be augmented UP TO (or capped
         down to) - for building your offline/online class-based augmentation
      2. loss weights for that same distribution - for FocalLoss,
         ClassBalancedLoss, LDAMLoss, or nn.CrossEntropyLoss(weight=...)

    Why one class instead of eyeballing numbers per class: if you set target
    counts by hand and compute loss weights separately from the raw counts,
    the two can drift out of sync (e.g. you 5x-augment DF and VASC, but the
    weights are still computed from the pre-augmentation counts, so you end
    up double-correcting for an imbalance that no longer exists at that
    magnitude). Building both from the same target distribution keeps them
    honest.

    Args:
        class_counts: dict {class_name: raw_sample_count}, e.g.
            {"NV": 6705, "MEL": 1113, "BKL": 1099, "BCC": 514,
             "AKIEC": 327, "VASC": 142, "DF": 115}

    Usage:
        planner = ClassDistributionPlanner({
            "NV": 6705, "BKL": 1099, "BCC": 514,
            "AKIEC": 327, "VASC": 142, "DF": 115,
        })

        # 1) how many augmented copies to generate per class
        targets = planner.target_counts(strategy='sqrt', min_count=300, max_count=1500)
        # -> {"NV": 1500, "BKL": 1099, "BCC": 514, "AKIEC": 327, "VASC": 300, "DF": 300}
        # use targets[c] - class_counts[c] as how many extra augmented images
        # to generate for class c in your offline augmentation step

        # 2) loss weights that match that same target distribution
        weights = planner.loss_weights(use_target_counts=True, method='effective')
        criterion = FocalLoss(alpha=weights, gamma=2.0)

        # class-index order for wiring into a tensor/list-based API
        print(planner.class_names)   # -> consistent order used everywhere above
    """

    def __init__(self, class_counts: dict):
        if not class_counts:
            raise ValueError("class_counts must be a non-empty dict of {class_name: count}")
        self.class_names = list(class_counts.keys())
        self.counts = torch.tensor(
            [class_counts[c] for c in self.class_names], dtype=torch.float32
        )
        self._targets = None  # cached after target_counts() is called

    def summary(self):
        """Prints counts and the resulting imbalance ratio (max/min)."""
        ratio = (self.counts.max() / self.counts.min()).item()
        print(f"Imbalance ratio (largest/smallest class): {ratio:.1f}x")
        for name, count in zip(self.class_names, self.counts.tolist()):
            print(f"  {name:>8}: {int(count):>6}")

    def target_counts(self, strategy='sqrt', min_count=None, max_count=None):
        """
        Decides how many samples each class SHOULD have after augmentation.

        Args:
            strategy: 'sqrt'     -> target ∝ sqrt(count). Softens the imbalance
                                     without fully flattening it - a reasonable
                                     default: majority classes stay largest,
                                     minority classes get a meaningful boost
                                     without absurd copy counts (e.g. 60x DF).
                       'log'      -> target ∝ log(count + 1). Softer still than
                                     sqrt - use if 'sqrt' still asks for more
                                     augmented copies of DF/VASC than you're
                                     comfortable generating.
                       'balanced' -> every class targets the same count (the
                                     current max, or max_count if given).
                                     Most aggressive - higher overfitting risk
                                     for classes as rare as DF (115 originals)
                                     since most of the "extra" images are
                                     augmented variants of a small base set.
                       'none'     -> keep raw counts as-is (only min/max_count
                                     clipping applied, if given).
            min_count: floor - no class targets fewer than this many samples
            max_count: ceiling - no class targets more than this many samples
                       (also caps how much you oversample/duplicate the
                       majority class, if strategy scales it up)

        Returns:
            dict {class_name: target_count} (rounded to int)
        """
        if strategy == 'sqrt':
            raw = torch.sqrt(self.counts)
        elif strategy == 'log':
            raw = torch.log1p(self.counts)
        elif strategy == 'balanced':
            cap = max_count if max_count is not None else self.counts.max().item()
            raw = torch.full_like(self.counts, cap)
        elif strategy == 'none':
            raw = self.counts.clone()
        else:
            raise ValueError(
                f"Unsupported strategy='{strategy}', use 'sqrt', 'log', 'balanced', or 'none'"
            )

        if strategy in {'sqrt', 'log'}:
            # rescale so the largest class keeps (approximately) its original
            # count, and everything else scales up relative to it
            raw = raw / raw.max() * self.counts.max()

        if min_count is not None:
            raw = torch.clamp(raw, min=min_count)
        if max_count is not None:
            raw = torch.clamp(raw, max=max_count)

        targets = {name: int(round(v)) for name, v in zip(self.class_names, raw.tolist())}
        self._targets = targets
        return targets

    def loss_weights(self, use_target_counts=False, method='effective'):
        """
        Loss weights matching either the raw counts or the planned target
        counts (call target_counts() first if use_target_counts=True).

        Args:
            use_target_counts: if True, computes weights from the distribution
                you planned with target_counts() instead of the raw counts -
                use this when your augmentation will actually reach those
                target counts, so the loss doesn't double-correct on top of
                augmentation that already balanced things.
            method: 'inverse' or 'effective' - see compute_class_weights()

        Returns:
            torch.Tensor of shape (num_classes,), in self.class_names order
        """
        if use_target_counts:
            if self._targets is None:
                raise ValueError("Call target_counts() before loss_weights(use_target_counts=True).")
            counts = torch.tensor([self._targets[c] for c in self.class_names], dtype=torch.float32)
        else:
            counts = self.counts

        return compute_class_weights(counts, method=method)


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


def residual_loss_weights(projected_counts: dict, method='effective'):
    """
    Loss weights computed from POST-augmentation projected counts (the third
    value returned by suggest_n_copies), so the weighting only corrects for
    whatever imbalance the suggested augmentation didn't already fix - instead
    of double-correcting for the full raw imbalance on top of augmentation.

    Args:
        projected_counts: dict {class_name_or_id: projected_count}, i.e. the
            `projected_counts` returned by suggest_n_copies()
        method: 'inverse' or 'effective' - see compute_class_weights()

    Returns:
        dict {class_name_or_id: weight}, same key order as given. Convert to
        a tensor in your class-index order before passing to FocalLoss(alpha=...)
        or nn.CrossEntropyLoss(weight=...).
    """
    keys = list(projected_counts.keys())
    values = [projected_counts[k] for k in keys]
    weights = compute_class_weights(values, method=method)
    return dict(zip(keys, weights.tolist()))
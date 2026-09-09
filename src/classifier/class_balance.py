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
      2. loss weights for the resulting class distribution - for FocalLoss,
         ClassBalancedLoss, LDAMLoss, or nn.CrossEntropyLoss(weight=...)

    Why one class instead of eyeballing numbers per class: if you set target
    counts by hand and compute loss weights separately from the raw counts,
    the two can drift out of sync (e.g. you 5x-augment DF and VASC, but the
    weights are still computed from the pre-augmentation counts, so you end
    up double-correcting for an imbalance that no longer exists at that
    magnitude). Building the augmentation plan first and computing loss
    weights from its projected counts keeps them consistent.

    Args:
        class_counts: dict {class_name: raw_sample_count}, e.g.
            {"NV": 6705, "MEL": 1113, "BKL": 1099, "BCC": 514,
             "AKIEC": 327, "VASC": 142, "DF": 115}

    Usage:
        # Option A: you already have counts
        planner = ClassDistributionPlanner({
            "NV": 6705, "BKL": 1099, "BCC": 514,
            "AKIEC": 327, "VASC": 142, "DF": 115,
        })

        # Option B: build directly from a YOLO/classifier data_yaml
        planner = ClassDistributionPlanner.from_data_yaml(
            data_yaml, class_names=names, dataset_type='segmentation'
        )

        # 1) Build the augmentation plan
        plan = planner.augmentation_plan(
            strategy='sqrt', min_count=300, max_count=1500
        )

        # 2) Use the number of copies for augmentation
        n_copies = plan['n_copies']

        # 3) Compute loss weights from the projected post-augmentation counts
        weights = planner.loss_weights(
            counts=plan['projected_counts'],
            method='effective'
        )

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

    @classmethod
    def from_data_yaml(cls, data_yaml, class_names=None, dataset_type='segmentation'):
        """
        Builds a ClassDistributionPlanner directly from a data_yaml dict,
        counting raw samples per class instead of requiring you to pass
        class_counts yourself.

        Args:
            data_yaml: dict with at least a 'train' key pointing to the
                images dir (segmentation) or the class-subfolder root
                (classifier)
            class_names: optional list/dict mapping class id -> class name.
                For dataset_type='segmentation', label files use integer
                class ids, so pass this to get readable names instead of
                raw ids as keys. For dataset_type='classifier', folder
                names are already used as-is, so this is normally not
                needed.
            dataset_type: 'segmentation' (reads YOLO-style .txt labels) or
                'classifier' (reads one subfolder per class)

        Returns:
            ClassDistributionPlanner instance built from the counted classes
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
            raise ValueError("No classes found in data_yaml['train'].")
        if class_names:
            counts = {class_names[k]: v for k, v in counts.items()}
        return cls(counts)

    def summary(self):
        """Prints counts and the resulting imbalance ratio (max/min)."""
        ratio = (self.counts.max() / self.counts.min()).item()
        print(f"Imbalance ratio (largest/smallest class): {ratio:.1f}x")
        for name, count in zip(self.class_names, self.counts.tolist()):
            print(f"  {name:>8}: {int(count):>6}")

    def augmentation_plan(self, strategy='sqrt', min_count=None, max_count=None):
        """
        Builds an augmentation plan from the current class distribution.

        The target count for each class is calculated internally using the
        selected strategy, then converted into a whole-number number of
        augmented copies per existing image. Because the number of copies
        must be an integer, projected_counts may differ from the internal
        target count. Use projected_counts when computing loss weights after
        augmentation.

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
            dict with:
              'current_counts':   {class_name: current raw count}
              'n_copies':         {class_name: how many extra copies per
                                   existing image to generate}
              'projected_counts': {class_name: current_count * (n_copies + 1)}
                                   (the count you'll actually end up with after
                                   augmentation)
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
            # Rescale so the largest class keeps approximately its original
            # count, and everything else scales up relative to it.
            raw = raw / raw.max() * self.counts.max()

        if min_count is not None:
            raw = torch.clamp(raw, min=min_count)
        if max_count is not None:
            raw = torch.clamp(raw, max=max_count)

        targets = {
            name: int(round(v))
            for name, v in zip(self.class_names, raw.tolist())
        }

        current_counts = {}
        n_copies = {}
        projected_counts = {}

        for name, count in zip(self.class_names, self.counts.tolist()):
            count = int(count)
            target = targets[name]
            current_counts[name] = count

            if count <= 0:
                n_copies[name] = 0
                projected_counts[name] = 0
                continue

            ratio = target / count
            copies = max(0, round(ratio) - 1)
            n_copies[name] = copies
            projected_counts[name] = count * (copies + 1)

        return {
            'current_counts': current_counts,
            'n_copies': n_copies,
            'projected_counts': projected_counts,
        }

    def loss_weights(self, counts=None, method='effective'):
        """
        Computes loss weights for a given class distribution.

        By default, weights are computed from the raw class counts. If
        augmentation has been planned, pass the plan's projected_counts
        so the weights reflect the distribution that will actually exist
        after augmentation.

        Args:
            counts: optional dict {class_name: count}. If None, the raw
                class counts are used. For post-augmentation weighting,
                pass augmentation_plan()['projected_counts'].
            method: 'inverse' or 'effective' - see compute_class_weights()

        Returns:
            torch.Tensor of shape (num_classes,), in self.class_names order
        """
        if counts is None:
            counts = self.counts
        else:
            counts = torch.tensor(
                [counts[c] for c in self.class_names], dtype=torch.float32
            )

        return compute_class_weights(counts, method=method)
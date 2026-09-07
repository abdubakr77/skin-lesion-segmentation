import torch
from torchvision.transforms import v2

# Shared so get_transforms() and denorm() can never drift out of sync
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size, apply_on_train=False):
    """
    Build a transform pipeline for training or validation/test.

    Args:
        image_size: target square image size in pixels
        apply_on_train: if True, adds light augmentation before normalization.
                         Keep this False (default) if you're already doing
                         class-based augmentation offline for the minority
                         classes (DF/VASC) - you don't want to double-augment
                         those crops.

    Returns:
        v2.Compose transform pipeline
    """
    base = [
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]

    augmentation = [
        v2.RandomAutocontrast(p=0.5),
        v2.RandomEqualize(p=0.5),
    ]

    tail = [v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

    if apply_on_train:
        return v2.Compose(base + augmentation + tail)
    return v2.Compose(base + tail)


def denorm(imgs):
    """
    Reverse ImageNet normalization so images can be displayed correctly.

    Args:
        imgs: normalized tensor of shape (B, C, H, W) or (C, H, W)

    Returns:
        tensor in [0, 1] range, same shape as input
    """
    mean = torch.tensor(IMAGENET_MEAN, device=imgs.device, dtype=imgs.dtype)
    std = torch.tensor(IMAGENET_STD, device=imgs.device, dtype=imgs.dtype)

    if imgs.dim() == 3:
        mean, std = mean.view(3, 1, 1), std.view(3, 1, 1)
    elif imgs.dim() == 4:
        mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"denorm expects 3D or 4D tensor, got shape {tuple(imgs.shape)}")

    return (imgs * std + mean).clamp(0, 1)
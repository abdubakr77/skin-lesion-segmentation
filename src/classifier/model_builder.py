import torch.nn as nn

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _find_classification_head(model: nn.Module):
    """
    Generically locates the model's final classification layer (nn.Linear),
    no matter what it's called: `head` (Swin), `fc` (ResNet/RegNet),
    `classifier` (EfficientNet/ConvNeXt/DenseNet/VGG), `heads.head` (ViT), etc.
    Works even if the head is nested inside an nn.Sequential (e.g. classifier[1]).

    Why this works: in every torchvision classification model, the final
    classifier layer is always the LAST module registered in __init__.
    named_modules() walks the model in registration order, so the last
    nn.Linear we see while walking the whole model is guaranteed to be it.

    Returns:
        (dotted_path, module, in_features)
        dotted_path example: "head", "fc", "classifier.1", "heads.head"
    """
    last_name, last_module = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last_module = name, module

    if last_module is None:
        raise ValueError(
            "Couldn't find any nn.Linear layer in this model to use as the "
            "classification head. This architecture may need manual handling."
        )

    return last_name, last_module, last_module.in_features


def _get_module_by_path(model: nn.Module, dotted_path: str):
    """Fetches a submodule anywhere in the model given its dotted path."""
    obj = model
    for p in dotted_path.split("."):
        obj = getattr(obj, p)
    return obj


def _set_module_by_path(model: nn.Module, dotted_path: str, new_module: nn.Module):
    """Replaces a submodule anywhere in the model given its dotted path."""
    parts = dotted_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def show_model_layers(model: nn.Module, max_depth: int = 1):
    """
    Prints the model's submodule names so you know what strings to pass into
    `unfreeze_layers`. Works generically for ANY model you pass in (Swin,
    ResNet, EfficientNet, ConvNeXt, ViT, ...) - not hardcoded to one architecture.

    Args:
        max_depth: how many levels deep to expand nested blocks
                   (1 = top-level names only, 2 = one level inside each block, ...)

    Example:
        model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        show_model_layers(model)
        # features.0, features.1, ..., features.7, norm, permute, avgpool, flatten, head

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        show_model_layers(model)
        # conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    """
    def _walk(module, prefix="", depth=0):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print(full_name)
            if depth < max_depth - 1:
                _walk(child, full_name, depth + 1)

    print(f"Layer names for {model.__class__.__name__}:")
    _walk(model)

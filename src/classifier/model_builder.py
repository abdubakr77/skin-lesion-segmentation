import torch.nn as nn

HEAD_KEYWORDS = ('head', 'heads', 'fc', 'classifier')

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


def show_model_layers(model: nn.Module, max_depth: int = 1, recommend_last_n: int = 2):
    """
    Prints the model's submodule names, marking which ones are recommended
    fine-tuning candidates based on generic heuristics (works for any
    architecture: Swin, ResNet, EfficientNet, ConvNeXt, ViT, ...):

    - The classification head (fc / head / classifier / heads) is always
      recommended, since it's task-specific and untrained for your data.
    - The last `recommend_last_n` parameterized blocks before the head are
      recommended, since late layers hold the most task-specific features.
    - Everything else with trainable parameters is marked as usually kept
      frozen (early, general-purpose features).
    - Layers with zero trainable parameters (activations, pooling, flatten,
      permute, dropout) are marked as having nothing to unfreeze.

    Args:
        max_depth: how many levels deep to expand nested blocks
                   (1 = top-level names only, 2 = one level inside each block, ...)
                   Increase this if a recommended block still looks too broad.
        recommend_last_n: how many of the last parameterized backbone blocks
                           to flag as recommended, besides the head

    Returns:
        recommended: list of layer name strings, ready to pass into
                     `unfreeze_layers`
    """
    entries = []  # dicts: name, has_params, is_leaf

    def _walk(module, prefix="", depth=0):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            has_children = any(True for _ in child.named_children())
            will_expand = has_children and depth < max_depth - 1
            has_params = any(True for _ in child.parameters(recurse=True))

            entries.append({'name': full_name, 'has_params': has_params, 'is_leaf': not will_expand})

            if will_expand:
                _walk(child, full_name, depth + 1)

    _walk(model)

    def _is_head(name):
        return name.split('.')[-1].lower() in HEAD_KEYWORDS

    leaves = [e for e in entries if e['is_leaf']]

    head_names = [e['name'] for e in leaves if e['has_params'] and _is_head(e['name'])]
    backbone_candidates = [e['name'] for e in leaves if e['has_params'] and e['name'] not in head_names]
    recommended_backbone = backbone_candidates[-recommend_last_n:] if recommend_last_n > 0 else []

    recommended = recommended_backbone + head_names

    print(f"Layer names for {model.__class__.__name__}:")
    print("(* = recommended to unfreeze)\n")

    for e in entries:
        if not e['is_leaf']:
            tag = "(expanded below)"
        elif not e['has_params']:
            tag = "(no trainable parameters)"
        elif e['name'] in head_names:
            tag = "* recommended (classifier head)"
        elif e['name'] in recommended_backbone:
            tag = "* recommended (late backbone block)"
        else:
            tag = "usually kept frozen"

        print(f"{e['name']:<25}{tag}")

    print(f"\nSuggested: unfreeze_layers(model, {recommended})")

    return recommended


# ----------------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------------

def build_model(
    model,
    num_classes,
    lr=1e-3,
    weight_decay=0.0001,
    epochs=10,
    freeze_backbone=True,
    unfreeze_layers=None,
    optimizer_type='adamw',
    scheduler_type='cosine',
    dropout=0.0,
    steps_per_epoch=None,
):
    """
    Generic classifier builder: works with ANY torchvision classification model
    and its matching Weights enum - not just Swin.

    Args:
        model_fn: the torchvision model constructor, e.g. swin_v2_t, resnet50, efficientnet_b0
        weights:  the matching Weights enum value, e.g. Swin_V2_T_Weights.DEFAULT
        steps_per_epoch: Number of steps per epoch. Only used when scheduler_type='onecycle'

    It automatically:
      - loads the model with pretrained weights
      - finds the final classification layer regardless of its name/location
      - replaces it with a new Linear (optionally preceded by Dropout)
      - freezes the backbone and keeps only the new head trainable (if freeze_backbone=True)
      - unfreezes any extra layers you name in `unfreeze_layers`

    To find out what layer names to use in `unfreeze_layers` for whichever model
    you pick, call `show_model_layers(model)` first - it's not tied to Swin.

    unfreeze_layers example: ['features.6', 'features.7', 'norm'] unfreezes any
    parameter whose name contains one of these substrings (works the same way
    regardless of which model_fn you passed in).

    Usage:
        from torchvision.models import swin_v2_t, Swin_V2_T_Weights
        model, optimizer, scheduler = build_model(swin_v2_t, Swin_V2_T_Weights.DEFAULT, num_classes=5)

        from torchvision.models import resnet50, ResNet50_Weights
        model, optimizer, scheduler = build_model(resnet50, ResNet50_Weights.DEFAULT, num_classes=5)

        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model, optimizer, scheduler = build_model(efficientnet_b0, EfficientNet_B0_Weights.DEFAULT, num_classes=5)
    """
    from torch.optim import AdamW, lr_scheduler, SGD

    print(f"Model Loaded: {model.__class__.__name__} (num_classes={num_classes})")

    # ---- generically find & replace the classification head ----
    head_path, _, in_features = _find_classification_head(model)

    if dropout > 0:
        new_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        new_head = nn.Linear(in_features, num_classes)

    _set_module_by_path(model, head_path, new_head)
    print(f"Head Replaced: '{head_path}' -> Linear(in_features={in_features}, out_features={num_classes})")

    # ---- freezing ----
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # keep the newly plugged-in head trainable
        for param in _get_module_by_path(model, head_path).parameters():
            param.requires_grad = True
        print("Backbone Frozen: True | Head Trainable: True")
    else:
        print("Backbone Frozen: False | All parameters trainable")

    if unfreeze_layers:
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
        print(f"Selective Unfreeze Enabled: {unfreeze_layers}")

    # ---- optimizer ----
    opt_type = optimizer_type.strip().lower()
    if opt_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"Optimizer Loaded: AdamW (lr={lr}, weight_decay={weight_decay})")
    elif opt_type in {'sgd', 'stochastic_gradient_descent'}:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        print(f"Optimizer Loaded: SGD (lr={lr}, momentum=0.9, weight_decay={weight_decay})")
    else:
        raise ValueError(f"Unsupported optimizer_type='{optimizer_type}'")

    # ---- scheduler ----
    sched_type = scheduler_type.strip().lower()
    if sched_type in {'cosine', 'cosineannealing'}:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        print(f"Scheduler Loaded: CosineAnnealingLR (T_max={epochs}, eta_min=1e-5)")
    elif sched_type in {'step', 'steplr'}:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
        print(f"Scheduler Loaded: StepLR (step_size={max(1, epochs // 3)}, gamma=0.1)")
    elif sched_type in {'reduce_on_plateau', 'reducelronplateau'}:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        print("Scheduler Loaded: ReduceLROnPlateau (mode='min', factor=0.5, patience=1)")
    elif sched_type in {'one_cycle', 'onecycle'}:
        if steps_per_epoch is None:
            steps_per_epoch = 1
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
        print(f"Scheduler Loaded: OneCycleLR (max_lr={lr}, steps_per_epoch={steps_per_epoch}, epochs={epochs})")
    else:
        scheduler = None
        print(f"Scheduler Loaded: None (scheduler_type='{scheduler_type}')")

    print("Model Build Complete")
    return model, optimizer, scheduler
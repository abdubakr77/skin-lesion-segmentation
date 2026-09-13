import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017). Down-weights easy, already-well-
    classified samples so training focuses on hard/rare examples - a solid
    default for imbalanced classification.

    Also handles SOFT labels (one-hot / mixed), so it works fine if you plug
    it in as the `criterion` alongside CutMix or MixUp.

    Args:
        alpha: per-class weighting, or None. Pass a per-class tensor from
               compute_class_weights() (see class_balance.py) to combine
               focusing + reweighting.
        gamma: focusing parameter. 0 reduces to (weighted) cross-entropy.
               Typical range: 1-3, 2 is a common default.
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: optional, only applied on the hard-label path
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if targets.ndim > 1:
            # soft/one-hot labels (e.g. from CutMix/MixUp).
            # probs_t here is an approximation of "confidence in the true
            # mixture", not a single class probability - standard practice
            # when combining focal loss with mixed labels.
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(targets * log_probs).sum(dim=1)
            probs_t = torch.exp(-ce_loss)
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, reduction='none', label_smoothing=self.label_smoothing
            )
            probs_t = torch.exp(-ce_loss)

        loss = ((1 - probs_t) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = (targets * alpha).sum(dim=1) if targets.ndim > 1 else alpha[targets]
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
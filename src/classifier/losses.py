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

class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (Cao et al., NeurIPS 2019).
    Instead of reweighting the loss, it enforces a LARGER decision margin for
    rare classes so the model is pushed to separate them more confidently.
    A good complement to reweighting for long-tailed data - some papers pair
    it with class weights applied only in later epochs ("deferred
    re-weighting"), but it works fine as a standalone loss too.

    Note: margins are defined per hard class label, so if you use this with
    CutMix/MixUp, soft targets are collapsed to their argmax first (a minor
    approximation - the two techniques don't combine perfectly).

    Args:
        samples_per_class: sample counts per class, in class-index order
        max_m: maximum margin (typical: 0.5)
        weight: optional per-class weight tensor (e.g. from compute_class_weights)
        s: logit scaling factor (typical: 30)
    """

    def __init__(self, samples_per_class, max_m=0.5, weight=None, s=30):
        super().__init__()
        samples_per_class = torch.as_tensor(samples_per_class, dtype=torch.float32)
        m_list = 1.0 / torch.sqrt(torch.sqrt(samples_per_class))
        self.m_list = m_list * (max_m / m_list.max())
        self.weight = weight
        self.s = s

    def forward(self, inputs, targets):
        if targets.ndim > 1:
            targets = targets.argmax(dim=1)

        m_list = self.m_list.to(inputs.device)
        index = torch.zeros_like(inputs, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)

        batch_m = m_list[targets].view(-1, 1)
        x_m = inputs - batch_m
        logits = torch.where(index, x_m, inputs) * self.s

        weight = self.weight.to(inputs.device) if self.weight is not None else None
        return F.cross_entropy(logits, targets, weight=weight)
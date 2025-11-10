import torch
import torch.nn.functional as F

def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor, alpha: float=0.25, gamma: float=2.0, reduction: str='mean') -> torch.Tensor:
    """Multi-class sigmoid focal loss.
    inputs: [Q, C] logits
    targets: [Q, C] binary one-hot targets
    """
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = targets * p + (1 - targets) * (1 - p)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        loss = alpha_t * loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
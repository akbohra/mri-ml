import torch
import torch.nn.functional as F

from core.base import BaseTask


class BinaryNCADTask(BaseTask):
    """
    Binary classification task:
    - NC (0)
    - AD (1)
    """

    def compute_loss(self, predictions, targets):
        """
        raw model outputs as predictions,
        ground truth labels 0's and 1's as targets,
        using cross_entropy here for now as it is stable
        and it can work for binary and multiclass
        """
        return F.cross_entropy(predictions, targets)

    def compute_metrics(self, predictions, targets):
        """
        Returns simple accuracy for now.
        can return sensitivity , specificity, and AUC, etc.
        """
        predicted_labels = torch.argmax(predictions, dim=1)
        correct = (predicted_labels == targets).sum().item()
        total = targets.numel()

        accuracy = correct / total
        return {"accuracy": accuracy}

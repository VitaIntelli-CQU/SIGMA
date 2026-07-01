import random

import numpy as np
import torch
import torch.nn.functional as F


def cross_modal_contrastive_loss(projected_modalities, temperature=0.1):
    """
    Bidirectional cross-modal InfoNCE loss for aligning paired embeddings.
    """
    if len(projected_modalities) < 2:
        return torch.tensor(0.0, device=projected_modalities[0].device)

    num_modalities = len(projected_modalities)
    num_samples = projected_modalities[0].shape[0]

    normalized = [F.normalize(proj, dim=1) for proj in projected_modalities]

    total_loss = torch.tensor(0.0, device=projected_modalities[0].device)
    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            sim_ij = torch.mm(normalized[i], normalized[j].T) / temperature
            sim_ji = torch.mm(normalized[j], normalized[i].T) / temperature

            labels = torch.arange(num_samples, device=sim_ij.device)
            loss_ij = F.cross_entropy(sim_ij, labels)
            loss_ji = F.cross_entropy(sim_ji, labels)
            total_loss += (loss_ij + loss_ji) * 0.5

    num_pairs = num_modalities * (num_modalities - 1) // 2
    return total_loss / num_pairs


def laplacian_regularization(x, edge_index):
    """
    Compute Laplacian regularization loss.
    """
    row, col = edge_index
    diff = x[row] - x[col]
    loss = (diff ** 2).sum(dim=1).mean()
    return loss


def modality_entropy_regularization(modality_weights, eps=1e-12):
    """
    Encourage non-collapsed modality weights.
    """
    entropy = -(modality_weights * torch.log(modality_weights + eps)).sum(dim=1).mean()
    return -entropy


def set_training_seed(seed):
    """
    Reset Python, NumPy, and PyTorch RNG states before each training run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_backend(strict_reproducibility=False):
    """
    Configure deterministic backend flags.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass

    if strict_reproducibility:
        return torch.device("cpu")
    return None


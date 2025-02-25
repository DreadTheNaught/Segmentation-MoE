import torch.nn as nn
import torch
import torch.nn.functional as F

# def dice_loss(true, logits, eps=1e-7):
#     """Computes the Sørensen–Dice loss.

#     Note that PyTorch optimizers minimize a loss. In this
#     case, we would like to maximize the dice loss so we
#     return the negated dice loss.

#     Args:
#         true: a tensor of shape [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         eps: added to the denominator for numerical stability.

#     Returns:
#         dice_loss: the Sørensen–Dice loss.
#     """
#     num_classes = logits.shape[1]
#     if num_classes == 1:
#         # print(true.squeeze(1).dtype)
#         true = true.long()
#         true_1_hot = torch.eye(num_classes + 1, device='cuda')[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true = true.long()
#         true_1_hot = torch.eye(num_classes, device='cuda')[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     cardinality = torch.sum(probas + true_1_hot, dims)
#     dice_loss = (2. * intersection / (cardinality + eps)).mean()
#     return dice_loss


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss."""
    num_classes = logits.shape[1]

    # Ensure true is long for indexing
    true = true.long()

    if num_classes == 1:
        # One-hot encoding for binary segmentation
        true_1_hot = F.one_hot(true.squeeze(
            1), num_classes=2).permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.detach()  # Ensure it's not part of the computation graph

        # Compute probabilities
        pos_prob = torch.sigmoid(logits)  # Foreground probability
        neg_prob = 1 - pos_prob  # Background probability
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        # Multi-class case: One-hot encoding
        true_1_hot = F.one_hot(true.squeeze(
            1), num_classes=num_classes).permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.detach()  # Prevent gradients

        probas = F.softmax(logits, dim=1)

    # Compute intersection and union
    # Sum over spatial dimensions
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dim=dims, keepdim=True)
    cardinality = torch.sum(probas + true_1_hot, dim=dims, keepdim=True)

    # Compute Dice loss
    dice_score = (2. * intersection + eps) / (cardinality + eps)
    dice_loss = 1 - dice_score.mean()

    return dice_loss


# def helper_dice(true,logits,temperature):
#     batch_size = true.shape[0]
#     dice = 0
#     for i in range(batch_size):
#         dice += torch.abs(dice_loss(torch.unsqueeze(true[i,:,:,:],0),torch.unsqueeze(logits[i,:,:,:],0)) - temperature)
#     return dice/batch_size


# def helper_dice_similarity(map1, map2):
#     """
#     Computes a similarity loss between two segmentation maps.

#     Args:
#         map1: first segmentation map of shape [B, C, H, W].
#         map2: second segmentation map of shape [B, C, H, W].

#     Returns:
#         similarity_loss: scalar similarity loss.
#     """
#     batch_size = map1.shape[0]
#     loss = 0
#     for i in range(batch_size):
#         loss += dice_loss(
#             torch.unsqueeze(map1[i], 0),
#             torch.unsqueeze(map2[i], 0)
#         )
#     # Higher similarity reduces the loss.
#     return 1 - loss / batch_size


def helper_dice_similarity(map1, map2):
    """
    Computes a similarity loss between two segmentation maps.

    Args:
        map1: first segmentation map of shape [B, C, H, W].
        map2: second segmentation map of shape [B, C, H, W].

    Returns:
        similarity_loss: scalar similarity loss.
    """
    return 1 - dice_loss(map1, map2)  # Directly return Dice similarity loss



def IntersectionOverUnion(seg1, seg2):

    intersection = torch.min(seg1, seg2)
    union = torch.max(seg1, seg2)

    IoU = intersection.sum() / union.sum()
    return IoU


# def MutuallyExclusiveLoss(partial_segmentation_maps):
#     loss = 0
#     N = len(partial_segmentation_maps) * len(partial_segmentation_maps[0])
#     for i in range(len(partial_segmentation_maps)):
#         for j in range(len(partial_segmentation_maps[0])):
#             for k in range(len(partial_segmentation_maps)):
#                 for l in range(len(partial_segmentation_maps[0])):
#                     if (i, j) != (k, l):
#                         loss += helper_dice_similarity(
#                             partial_segmentation_maps[i][j], partial_segmentation_maps[k][l])

#     loss = loss / (2 * N * (N - 1))
#     return loss


def MutuallyExclusiveLoss(partial_segmentation_maps):
    """
    Computes mutual exclusivity loss for partial segmentation maps.

    Args:
        partial_segmentation_maps: Tensor of shape [num_experts, B, 1, H, W]

    Returns:
        loss: scalar loss value
    """
    num_experts = partial_segmentation_maps.shape[0]
    # Ensure gradient tracking
    loss = torch.tensor(0., device=partial_segmentation_maps.device)
    # loss = partial_segmentation_maps.new_zeros(1)
    N = num_experts

    for i in range(num_experts):
        for j in range(i + 1, num_experts):  # Avoid redundant comparisons
            loss += dice_loss(
                partial_segmentation_maps[i], partial_segmentation_maps[j])

    loss = loss / ((N * (N - 1)) / 2)  # Normalize
    return loss



def CombinedLoss(pred, psm, target):
    BCE = nn.BCEWithLogitsLoss()
    return BCE(pred, target) + MutuallyExclusiveLoss(psm)
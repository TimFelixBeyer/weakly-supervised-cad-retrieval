import torch
import torch.nn.functional as F


def iou(pred, target, mask=1):
    """Compute the (optionally masked) IoU metric between two occupancy grids.

    Parameters
    ----------
    pred : torch.tensor, shape=(B, 1, D, D, D) or (B, 1, 1, D, D, D)
        Predicted occupancy voxel grid.
    target : torch.tensor, shape=(B, 1, D, D, D) or (1, C, 1, 1, D, D, D)
        Target occupancy voxel grid.
    mask : torch.tensor, shape=(B, 1, D, D, D)
        Mask that is 1 where the IoU should be considered, and 0 otherwise.

    Returns
    -------
    iou : torch.tensor, shape=(B,) or (B, C)
        Per-element IoU.
    """
    intersection = (torch.logical_and(pred, target) * mask).sum((-1, -2, -3, -4))
    union = (torch.logical_or(pred, target) * mask).sum((-1, -2, -3, -4))
    return intersection / (union + 1e-10)


def l1(pred, target, mask=1):
    l1_dist = (torch.abs(pred - target) * mask).sum((-1, -2, -3, -4))
    return l1_dist


def perceptual_similarity(
    query_maps=None,
    target_maps=None,
    model=None,
    query_embeddings=None,
    target_embeddings=None,
    norm='euclidean'
):
    assert query_maps is not None or query_embeddings is not None
    assert target_maps is not None or target_embeddings is not None
    if query_embeddings is None:
        query_embeddings = model(query_maps.reshape(-1, *query_maps.shape[2:])).reshape(
            query_maps.shape[0], query_maps.shape[1], -1
        ).transpose(0, 1)

    if target_embeddings is None:
        target_embeddings = model(target_maps.reshape(-1, *target_maps.shape[2:])).reshape(
            target_maps.shape[0], target_maps.shape[1], -1
        ).transpose(0, 1)

    if norm == "cosine":
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        target_embeddings = F.normalize(target_embeddings, dim=-1)

    weights = 1 - torch.mean((query_maps == 0.5).float(), dim=(-3, -2, -1)).transpose(0, 1)
    weights = weights / weights.sum(0, keepdim=True)[0]
    dists = torch.einsum("vqd,vtd->vqt", query_embeddings, target_embeddings) # [N_views, 1, 3049]
    dists = torch.einsum("vq,vqt->qt", weights, dists)
    return dists  # torch.cdist(query_embeddings, target_embeddings)

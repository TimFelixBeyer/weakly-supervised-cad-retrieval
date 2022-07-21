"""
Create caches for the filledIoU similarity metric (rescaling each shapenet model to fit each other).
WARNING: This takes very long.
"""
import os
import numpy as np
import torch
from instance_retrieval.baselines import EvaluationDataset
from instance_retrieval.geometry import rescale_voxel_grids
from instance_retrieval.network.metrics import iou
from tqdm import tqdm


def batched_iou(pred, target, mask, batch_size=1):
    ious = []
    target = target.to("cuda")
    mask = mask.to("cuda")
    for i in range(0, pred.shape[0], batch_size):
        ious.append(iou(pred[i : i + batch_size].to("cuda"), target, mask).cpu())
    return torch.cat(ious)


def compute_filledIoU_matrices():
    shape_dset = EvaluationDataset("./assets/shapenet_paths.npy")

    print("Loading Shape Voxels (All)")
    shape_voxels = torch.stack([shape["filled"] for shape in tqdm(shape_dset)])
    shape_bboxes = np.stack([shape["bbox"] for shape in tqdm(shape_dset)])

    # IoU is a similarity metric, 1 - IoU a distance metric (lower is more similar)
    distance_matrix = np.load("./assets/fillediou_distance_cache.npy")  # [3049, 3049]

    for i in tqdm(range(shape_bboxes.shape[0])):
        shape_voxels_ = torch.as_tensor(
            rescale_voxel_grids(
                shape_voxels[i:],
                [np.clip(shape_bboxes[i] / s, 0.05, 10) for s in shape_bboxes[i:]],
            )
        )
        iou = batched_iou(
            pred=shape_voxels_.unsqueeze(1) > 0.1,
            target=torch.as_tensor(shape_voxels[i : i + 1]).unsqueeze(0),
            mask=torch.tensor(1),
        )
        distance_matrix[i, i:] = iou.numpy().T
    return distance_matrix


if __name__ == "__main__":
    if not os.path.exists("./assets/fillediou_distance_cache.npy"):
        print("Computing filledIoU distance cache.")
        distance_matrix = compute_filledIoU_matrices()
        np.save("./assets/fillediou_distance_cache.npy", distance_matrix)
    else:
        print("filledIoU distance cache already exists.")



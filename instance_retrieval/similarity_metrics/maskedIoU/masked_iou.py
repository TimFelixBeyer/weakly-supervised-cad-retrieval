"""
Evaluate the maskedIoU similarity metric.
"""
import numpy as np
import torch
from tqdm import tqdm
from instance_retrieval.baselines import EvaluationDataset, evaluate_distance_matrix
from instance_retrieval.network.metrics import iou


def batched_iou(pred, target, mask, batch_size=1):
    ious = []
    target = target.to("cuda")
    mask = mask.to("cuda")
    for i in tqdm(range(0, pred.shape[0], batch_size)):
        ious.append(iou(pred[i : i + batch_size].to("cuda"), target, mask).cpu())
    return torch.cat(ious)


def compute_mIoU_matrices():
    scan_dset = EvaluationDataset("./assets/scannet_paths.npy")
    shape_dset_all = EvaluationDataset("./assets/shapenet_paths_all.npy")

    print("Loading Scan Voxels + SDFS")
    scan_voxels = torch.stack([scan["voxels"] for scan in tqdm(scan_dset)])
    scan_sdfs = torch.stack([scan["sdf"] for scan in tqdm(scan_dset)])
    print("Loading Shape Voxels (All)")
    shape_voxels_all = torch.stack([shape["voxels"] for shape in tqdm(shape_dset_all)])

    # IoU is a similarity metric, 1 - IoU a distance metric (lower is more similar).
    # fmt: off
    dist_mat_all = (
        1 - batched_iou(
            pred=shape_voxels_all.unsqueeze(1),
            target=scan_voxels.unsqueeze(0),
            mask=scan_sdfs > 0,
        ).numpy().T
    )
    # fmt: on
    return dist_mat_all


if __name__ == "__main__":
    dist_mat_all = compute_mIoU_matrices()
    np.save("./assets/miou_distance_cache_all.npy", dist_mat_all)
    evaluate_distance_matrix(dist_mat_all)

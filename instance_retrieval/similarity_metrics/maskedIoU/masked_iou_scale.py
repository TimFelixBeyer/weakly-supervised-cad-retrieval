"""
Evaluate the maskedIoU similarity metric while rescaling each shapnet model to
fit the scan.
"""
import numpy as np
import torch
from tqdm import tqdm
from instance_retrieval.baselines import EvaluationDataset, evaluate_distance_matrix
from instance_retrieval.geometry import rescale_voxel_grids
from instance_retrieval.network.metrics import iou

device = "cuda" if torch.cuda.is_available() else "cpu"


def cos_sim(a):
    return sum(a) / np.linalg.norm(a) / (3 ** 0.5)


def get_squish(scan_box_norm, scale):
    if cos_sim(scan_box_norm / scale) < 0.99:
        return np.nan * np.ones(3)
    return scan_box_norm / scale


def batched_iou(pred, target, mask, batch_size=1):
    ious = []
    target = target.to("cuda")
    mask = mask.to("cuda")
    for i in range(0, pred.shape[0], batch_size):
        ious.append(iou(pred[i : i + batch_size].to("cuda"), target, mask).cpu())
    return torch.cat(ious)


def compute_mIoU():
    scan_dset = EvaluationDataset("./assets/scannet_paths.npy")
    shape_dset_all = EvaluationDataset("./assets/shapenet_paths_all.npy")
    print("Loading Scan Voxels + SDFS")
    scans = [scan for scan in tqdm(scan_dset)]
    scan_voxels = torch.stack([scan["voxels"] for scan in tqdm(scans)])
    scan_sdfs = torch.stack([scan["sdf"] for scan in tqdm(scans)])
    scan_bboxes = np.stack([scan["bbox"] for scan in tqdm(scans)])
    del scans

    print("Loading Shape Voxels (All)")
    shape_voxels_all = torch.stack([shape["voxels"] for shape in tqdm(shape_dset_all)])
    shape_bboxes_all = np.stack([shape["bbox"] for shape in tqdm(shape_dset_all)])

    # IoU is a similarity metric, 1 - IoU a distance metric (lower is more similar)
    dist_mat_all = np.zeros((14150, 10851))

    for i in tqdm(range(scan_voxels.shape[0])):
        shape_voxels_all_scaled = torch.as_tensor(
            rescale_voxel_grids(
                shape_voxels_all,
                [get_squish(scan_bboxes[i], s) for s in shape_bboxes_all],
            )
        )
        iou = batched_iou(
            pred=shape_voxels_all_scaled.unsqueeze(1) > 0.1,
            target=scan_voxels[i : i + 1].unsqueeze(0),
            mask=scan_sdfs[i : i + 1] > 0,
        )
        dist_mat_all[i] = (1 - iou).numpy().T
    return dist_mat_all


if __name__ == "__main__":
    dist_mat_all = compute_mIoU()
    np.save("./assets/miou_distance_cache_all.npy", dist_mat_all)
    evaluate_distance_matrix(dist_mat_all)

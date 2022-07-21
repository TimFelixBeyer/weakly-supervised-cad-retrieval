from functools import lru_cache

import numpy as np
import torch
from instance_retrieval.io import get_scan_boxes, load_json, load_npy_to_torch
from scipy.spatial.distance import cdist
from tqdm import tqdm


def split_matrix(dist_mat_all, shape_paths_ranked, shape_paths, shape_paths_all):
    shape_idx_ranked = [shape_paths_all.index(path) for path in shape_paths_ranked]
    shape_idx = [shape_paths_all.index(path) for path in shape_paths]
    dist_mat_ranked = dist_mat_all[:, shape_idx_ranked]
    dist_mat = dist_mat_all[:, shape_idx]
    return dist_mat_ranked, dist_mat, dist_mat_all


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        scannet_path=load_json("data/datapaths.json")["scannet_instances"],
        shapenet_path=load_json("data/datapaths.json")["shapenet_instances"],
    ):
        self.paths = np.load(paths)
        if self.paths[0].startswith("scene"):
            if scannet_path != load_json("data/datapaths.json")["scannet_instances"]:
                cv_matches = load_json("assets/3dmatches_all_canonical_voting.json")
                self.paths = [p for p in self.paths if p in cv_matches]
            self.prefix = scannet_path
            self.type = "scan"
            self.bboxes = get_scan_boxes("assets/scan2cad/full_annotations.json")
        else:
            self.prefix = shapenet_path
            self.type = "cad"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        id = self.paths[index]
        voxel_path = f"{self.prefix}voxels/{id}.npy"
        vertices_path = f"{self.prefix}vertices/{id}.npy"
        faces_path = f"{self.prefix}faces/{id}.npy"
        # Add channel dim
        voxels = load_npy_to_torch(voxel_path).float().unsqueeze(0)
        vertices = load_npy_to_torch(vertices_path).float()
        faces = load_npy_to_torch(faces_path).float()
        bbox = np.maximum(vertices.max(0).values, -vertices.min(0).values)
        bbox /= bbox.max()

        sample = {
            "voxels": voxels,
            "vertices": vertices,
            "faces": faces,
            "bbox": bbox,
            "type": self.type,
            "id": id,
        }
        if self.type == "scan":
            sdf_path = f"{self.prefix}sdf/{id}.npy"
            sample["sdf"] = load_npy_to_torch(sdf_path).float().unsqueeze(0)
            sample["bbox"] = self.bboxes[id]
        return sample


def evaluate_method(method, distance="euclidean"):
    """Evaluates a given method using the metrics from the Joint Embedding paper,
    and against the new dataset using the new metrics we proposed.

    Parameters
    ----------
    method : function(Dict[str|torch.tensor]) -> np.ndarray
        Takes a sample from EvaluationDataset and returns a feature vector as np.array.
    """
    with torch.no_grad():
        print("Computing Scan Embeddings")
        scan_dset = EvaluationDataset("./assets/scannet_paths.npy")
        scan_embeddings = [method(scan) for scan in tqdm(scan_dset)]
        print("Computing Shape Embeddings")
        shape_dset_all = EvaluationDataset("./assets/shapenet_paths_all.npy")
        shape_embeddings_all = [method(shape) for shape in tqdm(shape_dset_all)]

    scan_embeddings = np.concatenate(scan_embeddings)
    shape_embeddings_all = np.concatenate(shape_embeddings_all)

    dist_mat_all = cdist(scan_embeddings, shape_embeddings_all, distance)
    evaluate_distance_matrix(dist_mat_all)


def evaluate_distance_matrix(dist_mat_all):
    """Takes three distance matrices, one for each way of evaluating the model.

    Parameters
    ----------
    dist_mat_all : np.ndarray, shape=(n_scans, 10851)
        The distance matrix pairing every scan with every cad model used in the
        Scan2CAD or scan-cad-similarity dataset.
    """
    datapaths = load_json("./data/datapaths.json")
    scan_paths = np.load("./assets/scannet_paths.npy").tolist()
    shape_paths_ranked = np.load("./assets/shapenet_paths_ranked.npy").tolist()
    shape_paths = np.load("./assets/shapenet_paths.npy").tolist()
    shape_paths_all = np.load("./assets/shapenet_paths_all.npy").tolist()
    dist_mat_ranked, dist_mat, dist_mat_all = split_matrix(
        dist_mat_all, shape_paths_ranked, shape_paths, shape_paths_all
    )
    print("Eval on SEEN classes (as in Joint Embedding)")
    print_metrics_joint_embedding(
        dist_mat_ranked, scan_paths, shape_paths_ranked, datapaths
    )

    print("Eval on UNSEEN classes (Scan2CAD catalog)")
    print_metrics(dist_mat, scan_paths, shape_paths, datapaths)

    # print("Eval on UNSEEN classes (full CAD catalog)")
    # print_metrics(dist_mat_all, scan_paths, shape_paths_all, datapaths)


def idx_to_name(idx, paths):
    return paths[idx]


@lru_cache(None)
def cached_load(path):
    return load_npy_to_torch(path).to("cuda")


def iou_from_idx(gt_idx, pred_idx, shape_paths, datapaths):
    """Given the indices of a ground truth shape

    Parameters
    ----------
    gt_idx : List[Int]
        [description]
    pred_idx : List[Int]
        [description]
    shape_paths : List[String]
        List of all the shapenet models.

    Returns
    -------
    iou : float
        The mean IoU of the gt shape with the pred shapes
    """
    prefix = f"{datapaths['shapenet_instances']}filled/"
    gt_shape = cached_load(f"{prefix}{shape_paths[gt_idx]}.npy").unsqueeze(0)
    shapeB = torch.stack(
        [cached_load(f"{prefix}{shape_paths[i]}.npy") for i in pred_idx]
    )
    intersection = torch.logical_and(gt_shape, shapeB).sum((1, 2, 3))
    union = torch.logical_or(gt_shape, shapeB).sum((1, 2, 3))
    return torch.mean(intersection / union).cpu().numpy()


def print_dict(dict):
    for k, v in dict.items():
        print(f"{k}: {v:.5f}, ", end="")
    print("")


def print_avg_metrics(metrics):
    avg_metrics = {}
    for key, value in metrics.items():
        if not key.startswith("_"):
            avg_metrics[key] = np.mean(value)
    print_dict(avg_metrics)


def print_cat_metrics(metrics):
    cats = np.unique(metrics["_cats"])
    cat_idx = np.array(metrics["_cats"])
    class_metrics = {}
    synset = load_json("./data/taxonomy.json")
    for metric, value in metrics.items():
        if not metric.startswith("_"):
            for cat in cats:
                idx = cat == cat_idx
                cat_ = synset[str(cat)]
                class_metrics.setdefault(cat_, {})
                class_metrics[cat_][metric] = np.mean(np.array(value)[idx])
    print(class_metrics)


def print_metrics(distance_matrix, scan_paths, shape_paths, datapaths):
    assert distance_matrix.shape[0] == len(scan_paths)
    assert distance_matrix.shape[1] == len(shape_paths)

    all_matches = load_json("./assets/3dmatches_all.json")
    train_matches = load_json("./assets/3dmatches_train.json")
    val_matches = load_json("./assets/3dmatches_val.json")
    test_matches = load_json("./assets/3dmatches_test.json")

    train_metrics = {
        "top1": [],
        "top5": [],
        "cat": [],
        "IoU": [],
        "IoU (Mean Top 5)": [],
        "mrr": [],
        "_cats": [],
    }
    val_metrics = {
        "top1": [],
        "top5": [],
        "cat": [],
        "IoU": [],
        "IoU (Mean Top 5)": [],
        "mrr": [],
        "_cats": [],
    }
    test_metrics = {
        "top1": [],
        "top5": [],
        "cat": [],
        "IoU": [],
        "IoU (Mean Top 5)": [],
        "mrr": [],
        "_cats": [],
    }
    shape_paths_idx = {k: v for v, k in enumerate(shape_paths)}
    sorted_preds = np.argsort(distance_matrix, 1)
    # correct_preds = {}
    for i, scan in enumerate(scan_paths):
        sorted_pred = sorted_preds[i]
        gt_idx = shape_paths_idx[all_matches[scan]]
        rank = 1 + np.where(sorted_pred == gt_idx)[0]
        gt_cat = all_matches[scan].split("/")[0]
        pred_cat = shape_paths[sorted_pred[0]].split("/")[0]

        if scan in train_matches:
            metrics = train_metrics
        elif scan in val_matches:
            metrics = val_metrics
        elif scan in test_matches:
            metrics = test_metrics
        else:
            continue

        metrics["top1"].append(rank == 1)
        metrics["top5"].append(rank <= 5)
        metrics["cat"].append(gt_cat == pred_cat)
        metrics["mrr"].append(1 / rank)
        metrics["IoU"].append(
            iou_from_idx(gt_idx, sorted_pred[:1], shape_paths, datapaths)
        )
        metrics["IoU (Mean Top 5)"].append(
            iou_from_idx(gt_idx, sorted_pred[:5], shape_paths, datapaths)
        )
        metrics["_cats"].append(gt_cat)

    print("Train")
    print_avg_metrics(train_metrics)
    print_cat_metrics(train_metrics)
    print("Val")
    print_avg_metrics(val_metrics)
    print_cat_metrics(val_metrics)
    print("Test")
    print_avg_metrics(test_metrics)
    print_cat_metrics(test_metrics)


def print_metrics_joint_embedding(distance_matrix, scan_paths, shape_paths, datapaths):
    assert distance_matrix.shape[0] == len(scan_paths)
    assert distance_matrix.shape[1] == len(shape_paths)

    ranked_samples = load_json(
        datapaths["scan_cad_similarity"]
        + "scan-cad_similarity_public_augmented-100_v0.json"
    )
    splits = load_json(
        datapaths["scan_cad_similarity"] + "scan2cad_objects_split.json"
    )["scan2cad_objects"]

    train_metrics = {
        "top1": [],
        "cat": [],
        "IoU": [],
        "IoU (Mean Top 5)": [],
        "rq": [],
        # "mrr": [],
        "_cats": [],
    }
    val_metrics = {
        "top1": [],
        "cat": [],
        "IoU": [],
        "IoU (Mean Top 5)": [],
        "rq": [],
        # "mrr": [],
        "_cats": [],
    }
    test_metrics = {
        "top1": [],
        "cat": [],
        "IoU": [],
        "IoU (Mean Top 5)": [],
        "rq": [],
        # "mrr": [],
        "_cats": [],
    }
    shape_indexes = {p: i for i, p in enumerate(shape_paths)}
    scan_indexes = {p: i for i, p in enumerate(scan_paths)}

    def id_from_name(name):
        return "_".join(name.split("/")[2].split("_")[:4]).replace("__", "/")

    def ranking_indices(sample):
        shape_indices = []
        for cad in sample["ranked"]:
            shape_indices.append(shape_indexes[cad["name"][5:]])
        for cad in sample["pool"]:
            shape_indices.append(shape_indexes[cad["name"][5:]])
        return shape_indices

    for sample in ranked_samples["samples"]:
        try:
            scan_idx = scan_indexes[id_from_name(sample["reference"]["name"])]
        except KeyError as e:
            # print(e)
            continue
        gt_category = sample["reference"]["name"].split("_")[4]

        shape_indices = ranking_indices(sample)
        # Select only the 106 ranked + augmented shapes from the distance matrix
        dists = distance_matrix[scan_idx][shape_indices]
        sorted_preds = np.array(shape_indices)[np.argsort(dists)]
        top1_pred = idx_to_name(sorted_preds[0], shape_paths)
        top1_pred_category = top1_pred.split("/")[0]
        ranking_quality = np.mean(
            [
                shape_indexes[s["name"][5:]] == sorted_preds[i]
                for i, s in enumerate(sample["ranked"])
            ]
        )

        split = splits[sample["reference"]["name"][6:]]
        if split == "train":
            metrics = train_metrics
        elif split == "validation":
            metrics = val_metrics
        elif split == "test":
            metrics = test_metrics
        else:
            continue

        metrics["top1"].append(
            top1_pred in [cad["name"][5:] for cad in sample["ranked"]]
        )

        metrics["cat"].append(gt_category == top1_pred_category)
        metrics["IoU"].append(
            iou_from_idx(shape_indices[0], sorted_preds[:1], shape_paths, datapaths)
        )
        metrics["IoU (Mean Top 5)"].append(
            iou_from_idx(shape_indices[0], sorted_preds[:5], shape_paths, datapaths)
        )
        metrics["rq"].append(ranking_quality)
        metrics["_cats"].append(gt_category)

    print("Train")
    print_avg_metrics(train_metrics)
    print_cat_metrics(train_metrics)
    print("Val")
    print_avg_metrics(val_metrics)
    print_cat_metrics(val_metrics)
    print("Test")
    print_avg_metrics(test_metrics)
    print_cat_metrics(test_metrics)

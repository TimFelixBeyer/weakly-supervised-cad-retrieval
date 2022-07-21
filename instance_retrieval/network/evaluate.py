"""Functions to evaluate a model and log results to tensorboard."""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance as chamfer_distance_p3d
from torchvision.io import read_image
from torchvision.transforms import Resize

from instance_retrieval.io import (
    dump_json,
    load_json,
    load_npy_to_torch,
    load_paths,
    load_shapenet_voxels,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


CHAMFER_DISTANCE_MATRIX = None
SHAPENET_LABELS = None
SHAPENET_VOXELS = None
SHAPENET_VOXELS_FILLED = None


def topk_acc(values, targets, k=1):
    return torch.any(torch.eq(values[:, :k], targets.unsqueeze(1)), dim=1)


def cat_acc(preds, targets):
    preds = np.array([SHAPENET_LABELS[v].split("/")[0] for v in preds[:, 0]])
    tgts = np.array([SHAPENET_LABELS[t].split("/")[0] for t in targets])
    return torch.as_tensor((preds == tgts))


def recipr_rank(preds, targets):
    return 1 / (1 + torch.eq(preds, targets.unsqueeze(1)).nonzero()[:, 1])


def iou(preds, targets, i=0):
    retrieved_models = SHAPENET_VOXELS_FILLED[preds[:, i]]
    correct_models = SHAPENET_VOXELS_FILLED[targets]
    intersection = torch.logical_and(correct_models, retrieved_models).sum((1, 2, 3, 4))
    union = torch.logical_or(correct_models, retrieved_models).sum((1, 2, 3, 4))
    return intersection / union


def topk_iou(preds, targets, k=1):
    return torch.mean(torch.stack([iou(preds, targets, i) for i in range(k)]), dim=0)


def chamfer_distance(preds, targets, datapaths):
    """Compute the chamfer distance between a set of predictions and targets.
    This is generally slow, therefore we precompute a cache on the first run.

    Parameters
    ----------
    preds : torch.Tensor
        Indices of the predicted shapenet retrievals
    targets : torch.Tensor
        Indices of the correct shapenet matches.
    datapaths : Dict
        Paths to the datasets.

    Returns
    -------
    torch.Tensor
        Chamfer distances between predictions and targets.
    """
    if not os.path.exists(datapaths["shapenet_instances"] + "vertices/cache.npy"):
        _build_chamfer_cache(datapaths)
    idx_table = load_json(datapaths["shapenet_instances"] + "vertices/idx_table.json")

    val_idx = [idx_table[SHAPENET_LABELS[v] + ".npy"] for v in preds[:, 0]]
    tgt_idx = [idx_table[SHAPENET_LABELS[t] + ".npy"] for t in targets]

    global CHAMFER_DISTANCE_MATRIX
    if CHAMFER_DISTANCE_MATRIX is None:
        CHAMFER_DISTANCE_MATRIX = load_npy_to_torch(
            datapaths["shapenet_instances"] + "vertices/cache.npy"
        ).to(device)
    return CHAMFER_DISTANCE_MATRIX[val_idx, tgt_idx]


def _build_chamfer_cache(datapaths):
    """Build a matrix with pairwise chamfer distances for all meshes in the target set.
    Also saves a JSON dict that maps cat/id to indices for the distance matrix.

    Args:
        datapaths (str): Path to where the meshes are stored.
    """
    shapenet_path = datapaths["shapenet_instances"] + "vertices/"
    instances = load_paths(datapaths["shapenet_instances"])
    idx_table = {}
    vertices_all = []
    lengths = []
    for cat, instance in instances:
        idx_table[cat + "/" + instance] = len(vertices_all)
        vertices = load_npy_to_torch(
            f"{shapenet_path}{cat}/{instance}.npy"
        )
        vertices_all.append(vertices.float().to(device))
        lengths.append(vertices.shape[0])

    dump_json(shapenet_path + "idx_table.json", idx_table)
    N = len(vertices_all)
    distances = np.zeros((N, N))
    print(f"Building chamfer dist cache for {N} meshes from ShapeNet, get a coffee!")
    batch = 256
    for i in range(N):
        for j in range(i + 1, N, batch):
            vj = torch.nn.utils.rnn.pad_sequence(
                vertices_all[j : j + batch], batch_first=True
            )
            vi = torch.as_tensor(vertices_all[i]).expand(vj.shape[0], -1, -1)
            lj = torch.as_tensor(lengths[j : j + batch])
            li = torch.as_tensor(lengths[i : i + 1]).expand(vj.shape[0])
            distances[i, j : j + batch] = chamfer_distance_p3d(
                vi, vj, li.to(device), lj.to(device), batch_reduction=None,
            )[0].cpu()
        print(f"{i}/{N}")
    distances = distances + distances.T
    np.save(shapenet_path + "cache.npy", distances)


def sample_query_images(preds, targets, thumbs, datapaths, n=5, k=5, res=128):
    """Creates sample images for n queries.
    Layout:
        Q  Q  Q  Q   (Images of the query scene, n=4)
        GT GT GT GT  (Renders of the ground-truth ShapeNet model)
        P1 P1 P1 P1  (Top 1 predictions)
        P2 P2 P2 P2  (Top 2 predictions)
        ...
        PK PK PK PK  (Top k-th predictions)

    Args:
        preds ([type]): [description]
        targets ([type]): [description]
        thumbs ([type]): [description]
        datapaths (String): [description]
        n (int, optional): [description]. Defaults to 5.
        k (int, optional): [description]. Defaults to 5.

    Returns:
        imgs (torch.Tensor): [description]
    """
    N = min(n, preds.shape[0])
    resize = Resize(res)
    shapenet_path = datapaths["shapenet_instances"] + "thumbs/"

    imgs = []
    for v, t in zip(preds[:N, :k], targets[:N]):
        img = [resize(read_image(f"{shapenet_path}{SHAPENET_LABELS[t]}.png"))]
        for v_ in v:
            i = resize(read_image(f"{shapenet_path}{SHAPENET_LABELS[v_]}.png"))
            if v_ != t:  # Color bad predictions in red, others remain green
                i = i[[1, 0, 2]]
            img.append(i)
        imgs.append(torch.cat(img, 1))
    imgs = torch.cat(imgs, 2)
    # Add normalized query image to top
    thumbs = 255 * torch.cat([resize(t[:3] / t[:3].max()) for t in thumbs[:N]], 2)
    thumbs = thumbs.to(torch.uint8)
    imgs = torch.cat([thumbs, imgs], 1)
    return imgs


def sample_query_ious(preds, targets, n=50):
    def _iou(v, i):
        intersection = torch.logical_and(v[i : i + 1], v).sum((1, 2, 3, 4))
        union = torch.logical_or(v[i : i + 1], v).sum((1, 2, 3, 4))
        return intersection / union

    def _topk_iou(v, i, k):
        iou_scores = _iou(v, i)
        idx = iou_scores.argsort(descending=True)[:k]
        return iou_scores[idx], idx

    correct_models = SHAPENET_VOXELS_FILLED[targets]
    ious = []
    for i in range(n):
        retrieved_models_i = SHAPENET_VOXELS_FILLED[preds[:, i : i + 1]][:, 0]
        intersection = torch.logical_and(correct_models, retrieved_models_i).sum(
            (1, 2, 3, 4)
        )
        union = torch.logical_or(correct_models, retrieved_models_i).sum((1, 2, 3, 4))
        ious.append(intersection / union)
    ious = torch.stack(ious).mean(1)  # [N, B]
    oracle_ious = []
    for i in range(targets.shape[0]):
        oracle_ious.append(_topk_iou(SHAPENET_VOXELS_FILLED, targets[i], k=n)[0])
    oracle_ious = torch.stack(oracle_ious).mean(0)

    plt.close()
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(ious.cpu().numpy(), "o")
    ax.plot(oracle_ious.cpu().numpy(), "o")
    ax.set_ylim(bottom=0)
    return fig


def evaluate_retrieval(
    loader, writer, loss_fn, cad_encoder, scan_encoder, step=0, store_embeddings=False
):
    """Evaluate a retrieval method on a dataset and report/save various metrics.

    Parameters
    ----------
    loader : torch.DataLoader
        Dataloader that iterates over the evaluation dataset.
    writer : tensorboardX.SummaryWriter
        The writer to be used for writing the metrics.
    loss_fn : Function
        Function to be called like loss_fn(cad_embeddings, scan_embeddings, batch).
        Used for reporting the loss over the dataset.
    cad_encoder : torch.nn.Module
        Model that maps a 32^3 voxel grid of a shapenet model to an embedding.
    scan_encoder : torch.nn.Module
        Model that maps a 32^3 voxel grid of a scan to an embedding.
    step : int, optional
        Current step for saving to TensorBoard, by default 0
    store_embeddings : bool, optional
        Whether to store the embeddings e.g. for a T-SNE visualization,
        by default False, as this can use a lot of disk space.

    Returns
    -------
    avg_metrics : Dict
        Contains metrics averaged over the dataset.
    metrics : Dict
        Contains metrics for each instance in the dataset.
    """
    with torch.no_grad():
        # First load, cache and encode all of shapenet.
        datapaths = load_json("./data/datapaths.json")
        global SHAPENET_VOXELS, SHAPENET_VOXELS_FILLED, SHAPENET_LABELS
        if SHAPENET_VOXELS is None:
            shapenet_path = datapaths["shapenet_instances"]
            (
                SHAPENET_VOXELS,
                SHAPENET_VOXELS_FILLED,
                SHAPENET_LABELS,
            ) = load_shapenet_voxels(shapenet_path)
        SHAPENET_VOXELS_FILLED.to("cpu")  # Fix GPU OOM
        keys = torch.cat(
            [
                cad_encoder(SHAPENET_VOXELS[i : i + 256])
                for i in range(0, SHAPENET_VOXELS.shape[0], 256)
            ]
        )
        SHAPENET_VOXELS_FILLED.to(device)
        metrics = {"loss": [], "top1": [], "top5": [], "cat": [], "IoU": [], "IoU (Mean Top 5)": [], "chamfer": [], "mrr": []}
        tsne_data = {"embeddings": [], "label_img": [], "metadata": []}

        for i, batch in enumerate(loader):
            cad_voxel = batch["shapenet_voxel"].to(device)  # [N, 1, 32, 32, 32])
            scan_voxel = batch["scannet_voxel"].to(device)

            # Compute loss
            cad_embeddings = cad_encoder(cad_voxel)  # Only needed for loss computation
            scan_embeddings = scan_encoder(scan_voxel)

            loss = loss_fn(cad_embeddings, scan_embeddings, batch)

            targets = [
                SHAPENET_LABELS.index(a + "/" + b)
                for a, b in zip(batch["catid_cad"], batch["cad_id"])
            ]
            targets = torch.as_tensor(targets).to(device)
            # Compare embeddings to shapenet
            distances = 1 - torch.mm(scan_embeddings, keys.T)
            # Uncomment to remove GT object from the candidate set
            # distances[torch.arange(scan_embeddings.shape[0]), targets] = 10000
            preds = torch.argsort(distances, dim=1)  # [N, K]

            metrics["loss"].extend([loss.item()] * scan_embeddings.shape[0])
            metrics["top1"].extend(topk_acc(preds, targets, k=1).tolist())
            metrics["top5"].extend(topk_acc(preds, targets, k=5).tolist())
            metrics["cat"].extend(cat_acc(preds, targets).tolist())
            metrics["IoU"].extend(iou(preds, targets).tolist())
            metrics["IoU (Mean Top 5)"].extend(topk_iou(preds, targets, k=5).tolist())
            metrics["chamfer"].extend(chamfer_distance(preds, targets, datapaths).tolist())
            metrics["mrr"].extend(recipr_rank(preds, targets).tolist())
            # Collect embeddings & thumbnails for viz if imgs are provided in the loader
            tsne_data["metadata"].extend(batch["catid_cad"])
            if "scannet_thumbnail" in batch:
                tsne_data["embeddings"].extend(scan_embeddings)
                tsne_data["label_img"].extend(batch["scannet_thumbnail"])
                if i == 0:
                    imgs = sample_query_images(
                        preds, targets, tsne_data["label_img"], datapaths, n=10, k=5
                    )
                    writer.add_image("Sample Prediction", imgs, global_step=step)
                    fig = sample_query_ious(preds, targets, n=50)
                    writer.add_figure("IoU figure", fig, global_step=step)

        avg_metrics = {}
        for key, value in metrics.items():
            avg_metrics[key] = np.mean(value)
            writer.add_scalar(key, np.mean(value), global_step=step)

        # Save category-level metrics
        cats = np.unique(tsne_data["metadata"])  # get the categories
        cat_idx = np.array(tsne_data["metadata"])
        class_metrics = {}
        synset = load_json("./data/taxonomy.json")
        for metric, value in metrics.items():
            for cat in cats:
                idx = cat == cat_idx
                cat_ = synset[str(cat)]
                class_metrics.setdefault(cat_, {})
                class_metrics[cat_][metric] = np.mean(np.array(value)[idx])
                writer.add_scalar(
                    f"class_metrics/{metric}/{cat_}",
                    class_metrics[cat_][metric],
                    global_step=step,
                )
            writer.add_scalar(
                f"class_metrics/{metric}/mean",
                np.mean([c[metric] for c in class_metrics.values()]),
                global_step=step,
            )

        if store_embeddings and tsne_data["embeddings"]:
            mat = torch.stack(tsne_data["embeddings"])
            label_img = torch.stack(tsne_data["label_img"])
            metadata = tsne_data["metadata"]
            writer.add_embedding(
                mat, label_img=label_img, metadata=metadata, global_step=step
            )
    return avg_metrics, metrics

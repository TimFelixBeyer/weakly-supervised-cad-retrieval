import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from instance_retrieval.baselines import (
    EvaluationDataset,
    evaluate_distance_matrix,
)
from instance_retrieval.render import render_depthmap, render_normalmap
from torchvision import models
from tqdm import tqdm


def batched_distance(shape_embeddings, scan_embeddings, scan_weights, batch_size=16):
    distance_matrix = torch.zeros((scan_embeddings.shape[0], shape_embeddings.shape[0]))
    for i in range(0, scan_embeddings.shape[0], batch_size):
        dists = torch.einsum(
            "qvd,tvd->vqt", scan_embeddings[i : i + batch_size], shape_embeddings
        )  # [N_views, 1, 3049]
        weights = scan_weights[i : i + batch_size]
        weights_norm = weights / (1e-10 + weights.sum(1, keepdims=True))
        dists = torch.einsum("qv,vqt->qt", weights_norm, dists)
        distance_matrix[i : i + batch_size] = dists
    return 1 - distance_matrix


def compute_perceptual():
    scan_dset = EvaluationDataset("./assets/scannet_paths.npy")
    shape_dset_all = EvaluationDataset("./assets/shapenet_paths_all.npy")
    with torch.no_grad():
        # Get the embedding after the 28th layer
        vgg = models.vgg19(pretrained=True).features
        vgg = nn.Sequential(*[l for l in list(vgg.children())[:28]]).to("cuda").eval()

        orientations = [(180, 45), (180, -25), (90, 45), (225, 0), (135, -45)]

        def perceptual_method(sample):
            normal_maps = []
            depth_maps = []
            for azim, elev in orientations:
                normal_maps.append(
                    render_normalmap(
                        sample["vertices"].unsqueeze(0).to("cuda"),
                        sample["faces"].unsqueeze(0).to("cuda"),
                        image_size=128,
                        dist=1,
                        elev=elev,
                        azim=azim,
                    ).to("cuda")
                )
                depth_maps.append(
                    render_depthmap(
                        sample["vertices"].unsqueeze(0).to("cuda"),
                        sample["faces"].unsqueeze(0).to("cuda"),
                        image_size=128,
                        dist=1,
                        elev=elev,
                        azim=azim,
                    ).to("cuda")
                )

            normal_maps = torch.cat(normal_maps)
            depth_maps = torch.stack(depth_maps)
            # Normalize images to sensible range
            normal_maps = normal_maps / 6 + 0.5
            depth_maps = torch.clip(depth_maps, 0) * 2 / 3
            composite = (1 - depth_maps) * normal_maps
            # Uncomment to view composite image
            # from PIL import Image
            # import os
            # os.makedirs(f"outputs/{sample['id'].split('/')[0]}", exist_ok=True)
            # for i in range(5):
            #     im = Image.fromarray(np.transpose(composite.cpu().numpy()[i]*255, (1,2,0)).astype(np.uint8))
            #     im.save(f"outputs/{sample['id']}_{i}.jpeg")

            embedding = F.normalize(
                vgg(composite).reshape(1, normal_maps.shape[0], -1), dim=-1,
            ).cpu()
            weights = 1 - torch.mean((normal_maps == 0.5).float(), dim=(1, 2, 3)).cpu()
            return embedding, weights

        print("Computing Scan Embeddings")
        scan_embeddings_weights = [perceptual_method(scan) for scan in tqdm(scan_dset)]
        scan_embeddings = torch.cat([s[0] for s in scan_embeddings_weights])
        scan_weights = torch.stack([s[1] for s in scan_embeddings_weights])
        del scan_embeddings_weights
        print("Computing Shape Embeddings (All)")
        shape_embeddings_all = torch.cat(
            [perceptual_method(shape)[0] for shape in tqdm(shape_dset_all)]
        )
        dist_mat_all = batched_distance(
            shape_embeddings_all, scan_embeddings, scan_weights
        )
    return dist_mat_all

if __name__ == "__main__":
    dist_mat_all = compute_perceptual()
    np.save("./assets/perceptual_distance_cache_all.npy", dist_mat_all)

    evaluate_distance_matrix(dist_mat_all)

"""This module processes the required shapenet models."""
import argparse
import os

from tqdm import tqdm
from instance_retrieval.data.mesh_to_voxel import mesh_to_voxel2
from instance_retrieval.geometry import apply_tqs

from instance_retrieval.io import load_json, load_shapenet_model
import numpy as np

from instance_retrieval.render import render_thumbnail


def get_ranked_models(root_folder):
    scenes = load_json(f"{root_folder}scan-cad_similarity_public_augmented-100_v0.json")
    paths = set()
    for sample in scenes['samples']:
        for cad in sample['ranked']:
            paths.add(cad['name'][5:])
        for cad in sample['pool']:
            paths.add(cad['name'][5:])
    assert len(paths) == 9297
    return sorted(list(paths))


def get_scan2cad_models(root_folder):
    scenes = load_json(f"{root_folder}/full_annotations.json")
    paths = set()
    for scene in scenes:
        for model in scene["aligned_models"]:
            paths.add(f"{model['catid_cad']}/{model['id_cad']}")
    assert len(paths) == 3049
    return sorted(list(paths))


def process_shapenet_model(id, scale, resolution):
    catid_cad, cad_id = id.split("/")
    # Voxelize and save model
    shapenet_voxel_path = f"{datapaths['shapenet_instances']}voxels/{catid_cad}/"
    shapenet_thumbs_path = f"{datapaths['shapenet_instances']}thumbs/{catid_cad}/"
    shapenet_filled_path = f"{datapaths['shapenet_instances']}filled/{catid_cad}/"
    shapenet_vertex_path = f"{datapaths['shapenet_instances']}vertices/{catid_cad}/"
    shapenet_faces_path = f"{datapaths['shapenet_instances']}faces/{catid_cad}/"

    os.makedirs(shapenet_voxel_path, exist_ok=True)
    os.makedirs(shapenet_filled_path, exist_ok=True)
    os.makedirs(shapenet_thumbs_path, exist_ok=True)
    os.makedirs(shapenet_vertex_path, exist_ok=True)
    os.makedirs(shapenet_faces_path, exist_ok=True)

    cad_vertices, cad_colors, cad_faces, shape_size = load_shapenet_model(
        cad_id, catid_cad, datapaths["shapenet"]
    )
    size = scale * max(np.ptp(cad_vertices, axis=0) / 2)

    cad_voxel = mesh_to_voxel2(cad_vertices, cad_faces, resolution, size=size)
    cad_filled = mesh_to_voxel2(
        cad_vertices, cad_faces, resolution, size=size, fill=True,
    )
    img = render_thumbnail(
        cad_vertices,
        cad_colors,
        cad_faces,
    )

    np.save(f"{shapenet_vertex_path}{cad_id}.npy", cad_vertices)
    np.save(f"{shapenet_faces_path}{cad_id}.npy", cad_faces)
    np.save(f"{shapenet_voxel_path}{cad_id}.npy", cad_voxel * np.clip(shape_size, 0.01, 10))
    np.save(f"{shapenet_filled_path}{cad_id}.npy", cad_filled)
    img.save(f"{shapenet_thumbs_path}{cad_id}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=36 / 32)
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()

    datapaths = load_json("./data/datapaths.json")
    # Scan-CAD Object Similarity Dataset
    paths_ranked = get_ranked_models(datapaths['scan_cad_similarity'])
    np.save("./assets/shapenet_paths_ranked.npy", np.array(paths_ranked))
    # Scan2CAD
    paths_scan2cad = get_scan2cad_models(datapaths['scan2cad'])
    np.save("./assets/shapenet_paths.npy", np.array(paths_scan2cad))
    # Combined
    paths_all = sorted(list(set(paths_ranked).union(set(paths_scan2cad))))
    assert len(paths_all) == 10851
    np.save("./assets/shapenet_paths_all.npy", np.array(paths_all))

    for i, cad in tqdm(enumerate(paths_all)):
        process_shapenet_model(cad, args.scale, args.resolution)



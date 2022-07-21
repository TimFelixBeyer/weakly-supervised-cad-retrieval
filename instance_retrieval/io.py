import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import trimesh
from pytorch3d.ops import cubify
from instance_retrieval.geometry import world_bbox_tf_from_annot, params_from_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_img(mat, path):
    plt.imshow(mat)
    plt.savefig(path)
    plt.close()


def dump_json(path, dict):
    with open(path, "w") as f:
        json.dump(dict, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def load_paths(path):
    return list((a, b) for a, b in pd.read_csv(path, header=None, dtype=str).values)


def load_scannet_scene(scan_id, scannet_root_dir):
    """Load a ScanNet scene into a vertex, color, face representation."""
    scan_file = f"{scannet_root_dir}{scan_id}/{scan_id}_vh_clean_2.ply"
    assert os.path.exists(scan_file), scan_file + " does not exist."

    tri_mesh = trimesh.load(scan_file)
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    colors = tri_mesh.visual.vertex_colors[:, :3]
    return vertices, colors, faces


def load_shapenet_model(model_id, category, shapenet_root_dir, color=[50, 200, 50]):
    cad_file = f"{shapenet_root_dir}{category}/{model_id}/models/model_normalized.obj"
    mesh = trimesh.load(cad_file, force="mesh")
    vertices = mesh.vertices
    vertices = vertices - (vertices.max(0) + vertices.min(0)) / 2
    faces = mesh.faces
    colors = np.array([color] * len(vertices))

    model = load_json(f"{shapenet_root_dir}{category}/{model_id}/models/model_normalized.json")
    shape_size = max(np.array(model["max"]) - np.array(model["min"])) / 2
    return vertices, colors, faces, shape_size


def load_shapenet_voxels(shapenet_path):
    labels = np.load("./assets/shapenet_paths.npy").tolist()
    voxels = [
        load_npy_to_torch(f"{shapenet_path}voxels/{instance}.npy")
        for instance in labels
    ]
    voxels_filled = [
        load_npy_to_torch(f"{shapenet_path}filled/{instance}.npy")
        for instance in labels
    ]
    voxels = torch.stack(voxels).float().unsqueeze(1).to(device)
    voxels_filled = torch.stack(voxels_filled).float().unsqueeze(1).to(device)
    return voxels, voxels_filled, labels


def load_npy_to_torch(path):
    assert path.endswith(".npy"), f"Path has to end in .npy, was {path}"
    return torch.from_numpy(np.load(path))


def save_voxels(path, voxel_grid, resolution=None):
    if resolution is None:
        resolution = max(*voxel_grid.shape)
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_grid, 1 / resolution)
    mesh.export(path)


def save_voxels_cube(path, voxel_grid):
    if isinstance(voxel_grid, list):
        voxs = np.concatenate(voxel_grid)
        voxel_grid = np.zeros((voxs.shape[0], voxs.shape[0], voxs.shape[0]))
        voxel_grid[:, :32, :32] = voxs
    mesh = cubify(torch.from_numpy(np.expand_dims(voxel_grid, 0)), 0.1, align="center")
    save_mesh(path, mesh.verts_packed().numpy(), mesh.faces_packed().numpy())


def save_mesh(path, vertices, faces):
    trimesh.Trimesh(vertices, faces).export(path)


def get_scan_boxes(path):
    bboxes = {}
    scenes = load_json(path)
    for scene in scenes:
        for i, model in enumerate(scene["aligned_models"]):
            M = world_bbox_tf_from_annot(model)
            T, S, R = params_from_matrix(M)
            scan_box_norm = S / max(S)
            bboxes[f"{scene['id_scan']}/{i}"] = scan_box_norm
    return bboxes
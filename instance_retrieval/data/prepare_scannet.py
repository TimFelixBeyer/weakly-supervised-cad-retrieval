"""Loads ScanNet, then aligns OBBs using Scan2CAD annotation, crops, voxelizes and saves.
Note: Splits are created in split_data_by_category.py
"""
import argparse
import os

from tqdm import tqdm
import numpy as np
from instance_retrieval.data.mesh_to_voxel import mesh_to_voxel
from instance_retrieval.geometry import (
    apply_tf,
    apply_tqs,
    get_tf_tqs,
    params_from_matrix,
    world_bbox_tf_from_annot,
)
from instance_retrieval.io import (
    dump_json,
    load_json,
    load_scannet_scene,
)
from instance_retrieval.render import render_thumbnail


def crop_from_bbox(verts, colors, faces, M, scale):
    """Takes a transformation matrix M that represents a bounding box and
    crops thevertices etc. to fit inside that bounding box.

    Parameters
    ----------
    verts : np.ndarray, shape=(n, 3)
    colors : np.ndarray, shape=(n, 3)
    faces : np.ndarray, shape=(m, 3)
    M : np.ndarray, shape=(4, 4)
        The Homogeneous transformation matrix.
    scale : Float
        The scale of the box.
        We need this since we want to scale the bbox such that all vertices
        within the bbox are kept, but the diagonal should be normalized only
        over the actual box size.
    Returns
    -------
    [type]
        [description]
    """
    T, S, R = params_from_matrix(M)
    # Reverse the translation and rotation
    verts = apply_tf(verts, R.T @ get_tf_tqs(t=-T))
    # Keep everthing that's inside the centered, axis-aligned bbox.
    idx = np.all([verts < S, -S < verts], axis=(0, 2))
    change = np.cumsum(np.logical_not(idx))
    keep_faces_idx = np.all(idx[faces], axis=1)
    faces = faces[keep_faces_idx]
    cropped_faces = faces - change[faces]
    verts = verts[idx]
    cropped_colors = colors[idx]

    # normalize such that diagonal is 1 * scale
    diag = np.linalg.norm(S) * 2
    cropped_verts = scale * verts / diag
    bbox = S / diag  # for voxelization
    return cropped_verts, cropped_colors, cropped_faces, bbox


def process_scene(scene, scale, resolution):
    """
    Parameters
    ----------
    scene : Dict
        Scan2CAD annotation of a single scene
    """
    scan_id = scene["id_scan"]

    # Process ScanNet scene
    scene_vertices, scene_colors, scene_faces = load_scannet_scene(
        scan_id, datapaths["scannet"]
    )
    # Transform it into world coordinates.
    scene_vertices = apply_tqs(
        scene_vertices,
        scene["trs"]["translation"],
        scene["trs"]["rotation"],
        scene["trs"]["scale"],
    )
    matches = {}
    for i, model in enumerate(scene["aligned_models"]):
        # Crop from ScanNet scene
        scannet_voxel_path = f"{datapaths['scannet_instances']}voxels/{scan_id}/"
        scannet_thumbs_path = f"{datapaths['scannet_instances']}thumbs/{scan_id}/"
        scannet_vertex_path = f"{datapaths['scannet_instances']}vertices/{scan_id}/"
        scannet_faces_path = f"{datapaths['scannet_instances']}faces/{scan_id}/"
        os.makedirs(scannet_voxel_path, exist_ok=True)
        os.makedirs(scannet_thumbs_path, exist_ok=True)
        os.makedirs(scannet_vertex_path, exist_ok=True)
        os.makedirs(scannet_faces_path, exist_ok=True)

        M = world_bbox_tf_from_annot(model, scale=scale)
        scan_vertices, scan_colors, scan_faces, scannet_bbox = crop_from_bbox(
            scene_vertices, scene_colors, scene_faces, M, scale=scale
        )
        if scan_vertices.shape[0] > 0 and scan_faces.shape[0] > 0:
            scan_voxel = mesh_to_voxel(
                scan_vertices,
                scan_faces,
                resolution,
                size=scale * max(scannet_bbox),
            )
            scan_img = render_thumbnail(
                scan_vertices,
                scan_colors,
                scan_faces
            )

            np.save(f"{scannet_voxel_path}{i}.npy", scan_voxel * np.clip(max(scannet_bbox), 0.01, 10))
            np.save(f"{scannet_vertex_path}{i}.npy", scan_vertices)
            np.save(f"{scannet_faces_path}{i}.npy", scan_faces)
            scan_img.save(f"{scannet_thumbs_path}{i}.png")
            # Commit to match database
            matches[f"{scan_id}/{i}"] = f"{model['catid_cad']}/{model['id_cad']}"
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=36 / 32)
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args()

    # Load dataset paths
    datapaths = load_json("./data/datapaths.json")
    scenes = load_json(f"{datapaths['scan2cad']}full_annotations.json")

    total_matches = {}
    for scene in tqdm(scenes):
        matches = process_scene(scene, args.scale, args.resolution)
        total_matches = {**total_matches, **matches}
    dump_json("./assets/3dmatches_all.json", total_matches)

    np.save("assets/scannet_paths.npy", np.array(sorted(list(total_matches.keys()))))

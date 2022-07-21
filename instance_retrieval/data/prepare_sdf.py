"""Create and store the 32^3 SDFs."""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from instance_retrieval.data.parse_sens import SensorData
from instance_retrieval.data.sdf import populate_sdf
from instance_retrieval.io import load_json

device = "cuda" if torch.cuda.is_available() else "cpu"


def save_sdf(scene, scale=36 / 32, resolution=32, skip=1):
    """Creates and saves all objects in a given scene as SDF grids using
    volumetric fusion.

    Parameters
    ----------
        scene : Dict
            Scan2CAD annotation of a single scene
        scale : Float
            Scale offset factor around object, 1 corresponds to a perfect fit.
        resolution : Int
            Resolution of the resulting SDF. (Default: 32)
        skip : Int
            Pick every n-th frame to create the SDF.
    """
    scan_id = scene["id_scan"]
    scannet_path = f"{datapaths['scannet_sens']}{scan_id}/{scan_id}.sens"
    sens = SensorData(scannet_path, skip)

    for i, model in enumerate(scene["aligned_models"]):
        scannet_path = f"{datapaths['scannet_instances']}voxels/{scan_id}/"
        scannet_path_sdf = f"{datapaths['scannet_instances']}sdf/{scan_id}/"
        # Only create SDFs for existing voxelized scans.
        if os.path.exists(f"{scannet_path}{i}.npy"):
            sdf, _, _ = populate_sdf(sens, resolution, scene, model, scale)
            os.makedirs(scannet_path_sdf, exist_ok=True)
            np.save(f"{scannet_path_sdf}{i}.npy", sdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--scale", type=float, default=36 / 32)
    args = parser.parse_args()

    # Load dataset paths
    datapaths = load_json("./data/datapaths.json")
    scenes = load_json(args.annotations)
    for scene in tqdm(scenes):
        save_sdf(scene, args.scale, args.resolution, args.skip)

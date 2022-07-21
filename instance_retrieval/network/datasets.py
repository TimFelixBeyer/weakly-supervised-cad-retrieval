import numpy as np
import torch
from instance_retrieval.io import (
    load_json,
    load_npy_to_torch,
)
from instance_retrieval.render import get_depth_maps, get_normal_maps
from scipy.ndimage import rotate, zoom
from torchvision import transforms
from torchvision.io import read_image


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        matches,
        resolution=32,
        return_images=False,
        augment_scans=False,
        scannet_root=load_json("data/datapaths.json")["scannet_instances"],
        shapenet_root=load_json("data/datapaths.json")["shapenet_instances"],
    ):
        self.matches = load_json(matches)
        self.resolution = resolution
        self.return_images = return_images
        self.augment_scans = augment_scans

        self.paths = list(self.matches.items())
        self.resize = transforms.Resize(128)

        self.shapenet_paths = np.load("./assets/shapenet_paths.npy")
        self.scannet_paths = np.load("./assets/scannet_paths.npy")
        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root
        self.shapenet_paths = {p: i for i, p in enumerate(self.shapenet_paths)}
        self.scannet_paths = {p: i for i, p in enumerate(self.scannet_paths)}

    def __len__(self):
        return len(self.paths)

    def _augment_scannet(self, scannet_voxel):
        # Augmentation is mostly useful on badly aligned obbs, like predicted ones.
        zoom_factor = np.random.uniform(0.5**0.5, 1)
        scannet_voxel = np.expand_dims(clipped_zoom(scannet_voxel[0], zoom_factor), 0)
        heading = np.random.random() * 360  # 90 - 45
        scannet_voxel = rotate(scannet_voxel, heading, axes=(-3, -1), reshape=False)
        return scannet_voxel

    def __getitem__(self, index):
        scannet_path, shapenet_path = self.paths[index]
        catid_cad = shapenet_path.split("/")[0]
        cad_id = shapenet_path.split("/")[1]

        scannet_voxel_path = (
            f"{self.scannet_root}voxels/{scannet_path}.npy"
        )
        scannet_sdf_path = f"{self.scannet_root}sdf/{scannet_path}.npy"
        shapenet_voxel_path = (
            f"{self.shapenet_root}voxels/{shapenet_path}.npy"
        )
        shapenet_filled_npy_path = (
            f"{self.shapenet_root}filled/{shapenet_path}.npy"
        )
        # Add channel dim
        scannet_voxel = load_npy_to_torch(scannet_voxel_path).float().unsqueeze(0)
        scannet_sdf = load_npy_to_torch(scannet_sdf_path).float().unsqueeze(0)
        shapenet_voxel = load_npy_to_torch(shapenet_voxel_path).float().unsqueeze(0)
        shapenet_voxel_filled = (
            load_npy_to_torch(shapenet_filled_npy_path).float().unsqueeze(0)
        )
        if self.augment_scans:
            scannet_voxel = self._augment_scannet(scannet_voxel)

        sample = {
            "scannet_voxel": scannet_voxel,
            "scannet_sdf": scannet_sdf,
            "shapenet_voxel": shapenet_voxel,
            "shapenet_voxel_filled": shapenet_voxel_filled,
            "cad_id": cad_id,
            "catid_cad": catid_cad,
            "scannet_idx": self.scannet_paths[scannet_path],
            "shapenet_idx": self.shapenet_paths[shapenet_path],
        }
        # Uncomment to do online perceptual loss computation (slow)
        # if self.return_depth:
        #     orientations = [(180, 45), (180, -25), (90, 45), (225, 0), (135, -45)]
        #     sample["scannet_normal"] = []
        #     sample["shapenet_normal"] = []
        #     for azim, elev in orientations:
        #         normal = get_normal_maps([shapenet_path], self.datapaths['shapenet_instances'], image_size=128, elev=elev, azim=azim)
        #         depth = get_depth_maps([shapenet_path], self.datapaths['shapenet_instances'], image_size=128, elev=elev, azim=azim)
        #         sample["scannet_normal"].append((1 - depth) * normal)
        #         normal = get_normal_maps([scannet_path], self.datapaths['scannet_instances'], image_size=128, elev=elev, azim=azim)
        #         depth = get_depth_maps([scannet_path], self.datapaths['scannet_instances'], image_size=128, elev=elev, azim=azim)
        #         sample["shapenet_normal"].append((1 - depth) * normal)
        #     sample["scannet_normal"] = torch.cat(sample["scannet_normal"], dim=0)
        #     sample["shapenet_normal"] = torch.cat(sample["shapenet_normal"], dim=0)

        if self.return_images:
            scannet_thumb_path = f"{self.scannet_root}thumbs/{scannet_path}.png"
            scannet_thumb = self.resize(read_image(scannet_thumb_path)) / 255.0
            # Make black background transparent
            alpha = torch.sum(scannet_thumb, dim=0, keepdim=True) > 0.03
            scannet_thumb = torch.cat([scannet_thumb, alpha], dim=0)
            sample["scannet_thumbnail"] = scannet_thumb
            # shapenet_thumb_path = f"{self.datapaths['shapenet']}thumbs/{shapenet_path}.png"
            # shapenet_thumb = self.resize(read_image(shapenet_thumb_path))
            # sample["shapenet_thumbnail"] = shapenet_thumb
        return sample


def clipped_zoom(img, zoom_factor):
    """Taken from https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
    and modified."""
    h, w, d = img.shape[-3:]
    # Zooming out
    if zoom_factor < 1:
        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        zd = int(np.round(d * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        front = (d - zd) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw, front:front+zd] = zoom(img, zoom_factor)
    return out


if __name__ == "__main__":
    print("Running performance test:")
    path = "./assets/3dmatches_all.json"
    dset = RetrievalDataset(path)
    import time

    m = dset[0]
    t0 = time.time()
    for i in range(100):
        m = dset[i]
    t1 = time.time()
    dset = RetrievalDataset(path, return_images=True)
    t2 = time.time()
    for i in range(100):
        m = dset[i]
    t3 = time.time()
    print(t1 - t0, t3 - t2)

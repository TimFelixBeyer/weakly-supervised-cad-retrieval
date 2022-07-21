import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt as distance


def mesh_to_voxel(vertices, faces, resolution, size, fill=False, crop=False):
    """Voxelize a 0-centered mesh.
    All vertices must lie in a cube: [-size, size]^3

    Parameters
    ----------
    vertices : [type]
    faces : [type]
    resolution : [type]
    size : [type]

    Returns
    -------
    [type]
        [description]
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((resolution, resolution, resolution))

    max_edge = size / resolution
    xyzmin = -np.array([size, size, size])
    xyzmax = np.array([size, size, size])
    x_y_z = np.array([resolution, resolution, resolution])

    vertices, faces = trimesh.remesh.subdivide_to_size(vertices, faces, max_edge)

    if crop:
        mask = np.all(xyzmax >= vertices, axis=1) * np.all(xyzmin <= vertices, axis=1)
        # print(f"Filtering to {np.mean(mask):.3f} of verts.")
        vertices = vertices[mask]

    all_in_box = np.all(xyzmax >= vertices.max(0)) and np.all(xyzmin <= vertices.min(0))
    assert all_in_box, f"Not in box {xyzmax} verts: {vertices.max(0)} {vertices.min(0)}"

    segments = []
    for i in range(3):
        # note the +1 in num
        segments.append(np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1)))

    # find where each point lies in corresponding segmented axis
    # -1 so index are 0-based; clip for edge cases
    voxel_x = np.clip(np.searchsorted(segments[0], vertices[:, 0]) - 1, 0, x_y_z[0])
    voxel_y = np.clip(np.searchsorted(segments[1], vertices[:, 1]) - 1, 0, x_y_z[1])
    voxel_z = np.clip(np.searchsorted(segments[2], vertices[:, 2]) - 1, 0, x_y_z[2])

    voxel = np.zeros((resolution, resolution, resolution))
    voxel[voxel_x, voxel_y, voxel_z] = 1

    if fill:
        # Flood fill around the object by starting from the eight corners,
        # then invert the filled mask to get the object.
        stack = set(
            (
                (0, 0, 0),
                (0, 0, resolution - 1),
                (0, resolution - 1, 0),
                (0, resolution - 1, resolution - 1),
                (resolution - 1, 0, 0),
                (resolution - 1, 0, resolution - 1),
                (resolution - 1, resolution - 1, 0),
                (resolution - 1, resolution - 1, resolution - 1),
            )
        )
        while stack:
            x, y, z = stack.pop()

            if voxel[x, y, z] == 0:
                voxel[x, y, z] = -1
                if x > 0:
                    stack.add((x - 1, y, z))
                if x < (resolution - 1):
                    stack.add((x + 1, y, z))
                if y > 0:
                    stack.add((x, y - 1, z))
                if y < (resolution - 1):
                    stack.add((x, y + 1, z))
                if z > 0:
                    stack.add((x, y, z - 1))
                if z < (resolution - 1):
                    stack.add((x, y, z + 1))
        return voxel != -1

    return voxel

def mesh_to_voxel2(vertices, faces, resolution, size, fill=False, crop=False):
    """Voxelize a 0-centered mesh.
    All vertices must lie in a cube: [-size, size]^3

    Parameters
    ----------
    vertices : [type]
    faces : [type]
    resolution : [type]
    size : [type]

    Returns
    -------
    [type]
        [description]
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((resolution, resolution, resolution))

    max_edge = size / resolution
    xyzmin = -np.array([size, size, size])
    xyzmax = np.array([size, size, size])
    x_y_z = np.array([resolution, resolution, resolution])

    vertices = trimesh.Trimesh(vertices, faces).sample(300000)

    if crop:
        mask = np.all(xyzmax >= vertices, axis=1) * np.all(xyzmin <= vertices, axis=1)
        # print(f"Filtering to {np.mean(mask):.3f} of verts.")
        vertices = vertices[mask]

    all_in_box = np.all(xyzmax >= vertices.max(0)) and np.all(xyzmin <= vertices.min(0))
    assert all_in_box, f"Not in box {xyzmax} verts: {vertices.max(0)} {vertices.min(0)}"

    segments = []
    for i in range(3):
        # note the +1 in num
        segments.append(np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1)))

    # find where each point lies in corresponding segmented axis
    # -1 so index are 0-based; clip for edge cases
    voxel_x = np.clip(np.searchsorted(segments[0], vertices[:, 0]) - 1, 0, x_y_z[0])
    voxel_y = np.clip(np.searchsorted(segments[1], vertices[:, 1]) - 1, 0, x_y_z[1])
    voxel_z = np.clip(np.searchsorted(segments[2], vertices[:, 2]) - 1, 0, x_y_z[2])

    voxel = np.zeros((resolution, resolution, resolution))
    voxel[voxel_x, voxel_y, voxel_z] = 1

    if fill:
        # Flood fill around the object by starting from the eight corners,
        # then invert the filled mask to get the object.
        stack = set(
            (
                (0, 0, 0),
                (0, 0, resolution - 1),
                (0, resolution - 1, 0),
                (0, resolution - 1, resolution - 1),
                (resolution - 1, 0, 0),
                (resolution - 1, 0, resolution - 1),
                (resolution - 1, resolution - 1, 0),
                (resolution - 1, resolution - 1, resolution - 1),
            )
        )
        while stack:
            x, y, z = stack.pop()

            if voxel[x, y, z] == 0:
                voxel[x, y, z] = -1
                if x > 0:
                    stack.add((x - 1, y, z))
                if x < (resolution - 1):
                    stack.add((x + 1, y, z))
                if y > 0:
                    stack.add((x, y - 1, z))
                if y < (resolution - 1):
                    stack.add((x, y + 1, z))
                if z > 0:
                    stack.add((x, y, z - 1))
                if z < (resolution - 1):
                    stack.add((x, y, z + 1))
        return voxel != -1

    return voxel


def mesh_to_sdf(vertices, faces, resolution, size):
    voxel = mesh_to_voxel(vertices, faces, resolution, size)
    sdf = distance(1 - voxel)
    return sdf

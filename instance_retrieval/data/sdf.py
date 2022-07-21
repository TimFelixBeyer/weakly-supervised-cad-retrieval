import numpy as np
from instance_retrieval.geometry import (
    get_tf_tqs,
    params_from_matrix,
    world_bbox_tf_from_annot,
)

DEPTH_MIN = 0.2
DEPTH_MAX = 8.0
WORLD_TO_GRID = None


def depth_to_skeleton(intrinsics, ux, uy, depth):
    if depth == 0 or depth == -np.inf:
        return -np.inf(np.ones((3,)))
    x = (ux - intrinsics[0, 2]) / intrinsics[0, 0]
    y = (uy - intrinsics[1, 2]) / intrinsics[1, 1]
    return depth * np.array([x, y, 1])


def skeleton_to_depth(intrinsics, p):
    """Camera system -> Pixel coordinates"""
    ux = p[:, 0] * intrinsics[0, 0] / p[:, 2] + intrinsics[0, 2]
    uy = p[:, 1] * intrinsics[1, 1] / p[:, 2] + intrinsics[1, 2]
    z = p[:, 2]
    return np.stack([ux, uy, z], axis=1)


def fit_bounding_box(points):
    return points.min(0), points.max(0)


def world_to_voxel(p):
    if p.shape[1] == 3:
        p = np.hstack((p, np.ones((p.shape[0], 1))))
    return np.rint(p.dot(WORLD_TO_GRID.T))[:, :3].astype(np.int64)


def voxel_to_world(p):
    if p.shape[1] == 3:
        p = np.hstack((p, np.ones((p.shape[0], 1))))
    return p.dot(np.linalg.inv(WORLD_TO_GRID).T)[:, :3]


def compute_frustum_bounds(intrinsic, transform, width, height, sdf_shape):
    corner_points = np.empty((8, 3))
    corner_points[0] = depth_to_skeleton(intrinsic, 0, 0, DEPTH_MIN)
    corner_points[1] = depth_to_skeleton(intrinsic, width - 1, 0, DEPTH_MIN)
    corner_points[2] = depth_to_skeleton(intrinsic, width - 1, height - 1, DEPTH_MIN)
    corner_points[3] = depth_to_skeleton(intrinsic, 0, height - 1, DEPTH_MIN)

    corner_points[4] = depth_to_skeleton(intrinsic, 0, 0, DEPTH_MAX)
    corner_points[5] = depth_to_skeleton(intrinsic, width - 1, 0, DEPTH_MAX)
    corner_points[6] = depth_to_skeleton(intrinsic, width - 1, height - 1, DEPTH_MAX)
    corner_points[7] = depth_to_skeleton(intrinsic, 0, height - 1, DEPTH_MAX)

    pts_l = np.floor(np.hstack((corner_points, np.ones((8, 1)))).dot(transform.T))
    pts_u = np.ceil(np.hstack((corner_points, np.ones((8, 1)))).dot(transform.T))
    pts_l_voxel = world_to_voxel(pts_l)
    pts_u_voxel = world_to_voxel(pts_u)
    bbox = fit_bounding_box(np.concatenate((pts_l_voxel, pts_u_voxel), axis=0))
    # Clip to grid size
    bbox = np.maximum(bbox[0], 0), (np.minimum(bbox[1] + 1, sdf_shape) - 1)
    return bbox


def scene_to_world(scene):
    t = scene["trs"]["translation"]
    q = scene["trs"]["rotation"]
    s = scene["trs"]["scale"]
    return get_tf_tqs(t, q, s)


def populate_sdf(sensor_data, size, scene, model, scale):
    sdf = -np.inf * np.ones([size, size, size]).astype(np.int32)
    free_ctr = np.zeros_like(sdf)
    weight = np.zeros_like(sdf)

    intrinsics = sensor_data.intrinsic_depth
    depth_width = sensor_data.depth_width
    depth_height = sensor_data.depth_height
    depth_shift = sensor_data.depth_shift

    global WORLD_TO_GRID
    M = world_bbox_tf_from_annot(model, scale=scale)
    T, S, R = params_from_matrix(M)
    WORLD_TO_GRID = 1 / max(S) * R.T @ get_tf_tqs(t=-T) @ scene_to_world(scene)
    WORLD_TO_GRID[3, 3] = 1

    padding = 0
    mat_s = 0.5 * np.diag(np.array([*sdf.shape, 3]) - 1)
    mat_t = np.eye(4)
    mat_t[:3, 3] = [1, 1, 1]
    WORLD_TO_GRID = mat_s @ mat_t @ WORLD_TO_GRID
    # mat_t = np.eye(4)
    # mat_t[:3, 3] = [padding, padding, padding]
    # sx = ((sdf.shape[0] - 1) - 2 * padding) / (sdf.shape[0] - 1)
    # sy = ((sdf.shape[1] - 1) - 2 * padding) / (sdf.shape[1] - 1)
    # sz = ((sdf.shape[2] - 1) - 2 * padding) / (sdf.shape[2] - 1)
    # mat_s = np.diag([sx, sy, sz, 1])
    # WORLD_TO_GRID = mat_t @ mat_s @ WORLD_TO_GRID

    for frame in sensor_data.frames:
        camera_to_world = frame.camera_to_world
        world_to_camera = np.linalg.inv(camera_to_world)
        depth_image = frame.get_depth_image(depth_height, depth_width).T / depth_shift

        voxel_bounds = compute_frustum_bounds(
            intrinsics, camera_to_world, depth_width, depth_height, sdf.shape
        )
        sdf, weight, free_ctr = update_sdf_with_frame(
            voxel_bounds,
            intrinsics,
            depth_height,
            depth_width,
            depth_image,
            world_to_camera,
            sdf,
            free_ctr,
            weight,
        )
    return sdf, weight, free_ctr


def update_sdf_with_frame(
    voxel_bounds,
    intrinsics,
    depth_height,
    depth_width,
    depth_image,
    world_to_camera,
    sdf,
    free_ctr,
    weight,
):
    """Tested to be equivalent to update_sdf_with_frame_slow."""

    x = range(voxel_bounds[0][0], voxel_bounds[1][0] + 1)
    y = range(voxel_bounds[0][1], voxel_bounds[1][1] + 1)
    z = range(voxel_bounds[0][2], voxel_bounds[1][2] + 1)
    x, y, z = np.meshgrid(x, y, z)
    ijk = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
    ijk = ijk.astype(np.int64)

    pts_world = voxel_to_world(ijk)
    pts_world = np.hstack((pts_world, np.ones((pts_world.shape[0], 1))))
    pts_camera = pts_world.dot(world_to_camera.T)[:, :3]
    pts_depth = skeleton_to_depth(intrinsics, pts_camera)  # to u,v coords
    pts_depth_i = np.round(pts_depth).astype(np.int32)
    p = pts_depth
    pi = pts_depth_i

    width_mask = np.all((0 <= pi[:, 0], pi[:, 0] < depth_width), axis=0)
    height_mask = np.all((0 <= pi[:, 1], pi[:, 1] < depth_height), axis=0)
    depth_mask = 0 <= p[:, 2]  # Filter pts behind the camera
    mask = np.all((width_mask, height_mask, depth_mask), axis=0)

    depths = depth_image[pi[mask][:, 0], pi[mask][:, 1]]
    depth_mask = np.all((DEPTH_MIN <= depths, depths < DEPTH_MAX), axis=0)

    depths = depths[depth_mask]
    ijk = ijk[mask][depth_mask]
    p = pts_depth[mask][depth_mask]

    p_mask = p[:, 2] < depths
    p_idx = ijk[p_mask]
    free_ctr[p_idx[:, 0], p_idx[:, 1], p_idx[:, 2]] += 1

    sdf_ = depths - p[:, 2]
    # Uncomment to visualize the camera trajectory
    # np.linalg.norm(pts_camera[mask][depth_mask], axis=1) < 0.173
    fill_mask = np.abs(sdf_) <= np.abs(sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]])
    fill_idx = ijk[fill_mask]
    sdf[fill_idx[:, 0], fill_idx[:, 1], fill_idx[:, 2]] = sdf_[fill_mask]
    weight[fill_idx[:, 0], fill_idx[:, 1], fill_idx[:, 2]] += 1

    return sdf, weight, free_ctr


# def update_sdf_with_frame_slow(
#     voxel_bounds,
#     intrinsics,
#     depth_height,
#     depth_width,
#     depth_image,
#     world_to_camera,
#     sdf,
#     free_ctr,
#     weight,
# ):

#     for k in range(voxel_bounds[0][2], voxel_bounds[1][2] + 1):
#         print(k / (voxel_bounds[1][2] - voxel_bounds[0][2]))
#         for j in range(voxel_bounds[0][1], voxel_bounds[1][1] + 1):
#             for i in range(voxel_bounds[0][0], voxel_bounds[1][0] + 1):
#                 p = voxel_to_world(np.array([[i, j, k]]))
#                 p = np.hstack((p, np.ones((p.shape[0], 1))))
#                 p = world_to_camera.dot(p.T).T[:, :3]
#                 p = skeleton_to_depth(intrinsics, p)
#                 pi = np.round(p).astype(np.int32)
#                 # print(pi, depth_image.shape, depth_width, depth_height)
#                 if 0 <= pi[0, 0] < depth_width and 0 <= pi[0, 1] < depth_height:
#                     d = depth_image[pi[0, 0], pi[0, 1]]
#                     if DEPTH_MIN <= d <= DEPTH_MAX:
#                         if p[0, 2] < d:
#                             free_ctr[i, j, k] += 1

#                         sdf_ = d - p[0, 2]
#                         if np.abs(sdf_) <= np.abs(sdf[i, j, k]):
#                             sdf[i, j, k] = sdf_
#                             weight[i, j, k] = 1
#     return sdf, weight, free_ctr

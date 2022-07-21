import numpy as np
import quaternion
from skimage.transform import rescale


def get_tf_tqs(t=np.zeros((3)), q=np.array([1, 0, 0, 0]), s=np.ones((3))):
    """Get a homogeneous 4x4 transformation matrix that corresponds to the
    transformation V_trans = R*S*V + T

    Parameters
    ----------
    t : np.ndarray, shape=(3,), optional
        The transformation to be applied.
    q : np.ndarray, shape=(4,), optional
        The rotation to be applied, given as a unit quaternion.
    s : np.ndarray, shape=(3,), optional
        The scaling to be applied.

    Returns
    -------
    M : np.ndarray, shape=(4, 4)
        The transformation matrix.
    """
    M = np.eye(4)
    M[:3, :3] = quaternion.as_rotation_matrix(np.quaternion(*q)) @ np.diag(s)
    M[:3, 3] = t
    return M


def apply_tf(v, M):
    """Applies a homogeneous transformation matrix to a set of 3D points.

    Parameters
    ----------
    v : np.ndarray, shape=(N, 3)
        The coordinates of the points/vertices to be transformed.
    M : np.ndarray, shape=(4, 4)
        The 4x4 transformation matrix.

    Returns
    -------
    v_tf : np.ndarray, shape=(N, 3)
        The coordinates of the points/vertices after the transformation.
    """
    v_h = np.hstack([v, np.ones((v.shape[0], 1))])
    v_tf = v_h.dot(M.T)[:, :3]
    return v_tf


def apply_tqs(v, t=np.zeros((3)), q=np.array([1, 0, 0, 0]), s=np.ones((3))):
    """Applies a transformation given by T(ranslation), Q(uaternion), S(cale).
    V_trans = R*S*V + T

    Parameters
    ----------
    v : np.ndarray, shape=(N, 3)
        The coordinates of the points/vertices to be transformed.
    t : np.ndarray, shape=(3,), optional
        The transformation to be applied.
    q : np.ndarray, shape=(4,), optional
        The rotation to be applied, given as a unit quaternion.
    s : np.ndarray, shape=(3,), optional
        The scaling to be applied.

    Returns
    -------
    v_tf : np.ndarray, shape=(N, 3)
        The coordinates of the points/vertices after the transformation.
    """
    M = get_tf_tqs(t, q, s)
    v_tf = apply_tf(v, M)
    return v_tf


def invert_quaternion(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def params_from_matrix(M):
    """Decomposes a homogeneous transformation matrix into translation, rotation and
    scale, such that applying M to x is equivalent to applying T @ R @ S to x.
    Note that the dimensions of the returned arrays do not match.

    Parameters
    ----------
    M : np.ndarray, shape=(4, 4)
        The Homogeneous transformation matrix.

    Returns
    -------
    T : np.ndarray, shape = (3,)
    S : np.ndarray, shape = (3, 3)
    R : np.ndarray, shape = (4, 4)
    """
    T = M[:3, 3]
    S = np.linalg.norm(M[:3, :3], axis=0)
    R = np.eye(4)
    R[:3, :3] = M[:3, :3] / np.expand_dims(S, 0)
    return T, S, R


def world_bbox_tf_from_annot(model, scale=1.0):
    """Returns a homogeneous transformation matrix that transforms a [-1, 1] ^ 3
    bounding box to the corner points for an annotations world coordinates.

    Parameters
    ----------
    model : Dict
        Alignment params.
    scale : float, optional
        Scale factor (to include more of surroundings), by default 1.0

    Returns
    -------
    M : np.ndarray, shape=(4, 4)
        The Homogeneous transformation matrix.
    """
    trs = model["trs"]
    bbox = np.diag([*(scale * np.array(model["bbox"])), 1])
    center = get_tf_tqs(t=np.array(model["center"]))
    translate = get_tf_tqs(t=np.array(trs["translation"]))
    rotate = get_tf_tqs(q=trs["rotation"])
    scale = get_tf_tqs(s=np.array(trs["scale"]))
    M = translate.dot(rotate).dot(scale).dot(center).dot(bbox)
    return M


def rescale_voxel_grid(grid, scale, res=32):
    """Perform anisotropic rescaling of a voxel grid.
    Due to interpolation, grids returned by this function may not be binary.
    WARNING: Fails silently and returns an all 0 grid if the provided scale
    contains NaNs.

    Parameters
    ----------
    grid : torch.Tensor, shape=(1, N, N, N)
        3D voxel grid to be rescaled.
    scale : np.array, shape=(3,)
        Scale factors, can be any value.
    res : int, optional
        Resolution of the voxel grid, by default 32

    Returns
    -------
    np.ndarray, shape=(res, res, res)
        The rescaled grid.
    """
    assert res % 2 == 0, "Resolution has to be even."
    # Filter out matches where the scaling effort would be too large
    if np.any(np.isnan(scale)):
        return 0 * grid[0].cpu().numpy()
    grid = rescale(grid[0].cpu().numpy().copy(), scale, anti_aliasing=True)
    # Pad out smaller dims
    H, W, D = grid.shape
    pad_a = np.array([(res - H) // 2, (res - W) // 2, (res - D) // 2])
    pad_a = np.clip(pad_a, 0, res)
    pad_b = np.array([res - H - pad_a[0], res - W - pad_a[1], res - D - pad_a[2]])
    pad_b = np.clip(pad_b, 0, res)
    grid = np.pad(
        grid, ((pad_a[0], pad_b[0]), (pad_a[1], pad_b[1]), (pad_a[2], pad_b[2]))
    )
    # Crop back larger dims
    H, W, D = grid.shape
    return grid[
        (H - res) // 2 : (H + res) // 2,
        (W - res) // 2 : (W + res) // 2,
        (D - res) // 2 : (D + res) // 2,
    ]


def rescale_voxel_grids(grids, scales):
    """Rescale multiple voxel grids at the same time.

    Parameters
    ----------
    grids : List[torch.Tensor]
        List of occupancy grids.
    scales : List[np.ndarray]
        List of scales corresponding to the grids.

    Returns
    -------
    np.ndarray, shape=(B, N, N, N)
        Array containing the rescaled occupancy grids.
    """
    return np.stack(
        [rescale_voxel_grid(grid, scale) for grid, scale in zip(grids, scales)]
    )

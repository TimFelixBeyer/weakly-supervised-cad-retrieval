import json

import torch
from PIL import Image
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    HardFlatShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures import Meshes

from instance_retrieval.io import load_npy_to_torch, load_shapenet_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def render_thumbnail(vertices, colors, faces, image_size=64):
    return render_thumbnails(
        torch.as_tensor(vertices, device=device).unsqueeze(0),
        torch.as_tensor(colors, device=device).unsqueeze(0),
        torch.as_tensor(faces, device=device).unsqueeze(0),
        image_size=image_size,
    )[0]


def render_thumbnails(vertices, colors, faces, image_size=64):
    """Render thumbnails of meshes with a given color + resolution.

    Args:
        vertices
        colors
        faces
        save_paths (List[String]): Paths at which the thumbnails should be saved.
        image_size (int, optional): Defaults to 512.
    """
    mesh = Meshes(verts=vertices.float(), faces=faces.float())
    mesh.textures = Textures(verts_rgb=colors.float())
    # create texture
    # Initialize a camera.
    R, T = look_at_view_transform(1.0, 30, 150)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(image_size=image_size, cull_backfaces=True)

    # lights = PointLights(device=device, location=[[0.0, 0.0, -1.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardFlatShader(device=device, cameras=cameras),  # , lights=lights),
    )
    images = renderer(mesh)
    return [
        Image.fromarray((images[i, ..., :3].to(torch.uint8)).cpu().numpy())
        for i in range(len(images))
    ]


def render_depthmap(vertices, faces, image_size=128, dist=1.0, elev=30, azim=150):
    """Render depthmaps of meshes with a given color + resolution.

    Parameters
    ----------
    vertices : torch.Tensor, shape=(B, N, 3)
        Array of vertex coordinates.
    faces : torch.Tensor, shape=(B, M, 3)
        Array of vertex indices for each face.
    image_size : int, optional
        Image resulution, by default 128
    dist : float, optional
        Camera distance from the origin, by default 1.0
    elev : int, optional
        Eelevation of the camera viewpoint in degrees, by default 30
    azim : int, optional
        Azimuth of the camera viewpoint in degrees (180 is from the front),
        by default 150

    Returns
    -------
    depth_maps : torch.Tensor, shape=(B, image_size, image_size)
        The depthmaps from the given viewpoint.
    """
    mesh = Meshes(verts=vertices.float(), faces=faces.float())
    # create texture
    # Initialize a camera.
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size, bin_size=[0, None][0]
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    depth_maps = rasterizer(mesh).zbuf.min(dim=-1)[0]
    return depth_maps


def render_normalmap(vertices, faces, image_size=128, dist=1.0, elev=30, azim=150):
    """Render world-space normal maps of meshes with a given color + resolution.

    Parameters
    ----------
    vertices : torch.Tensor, shape=(B, N, 3)
        Array of vertex coordinates.
    faces : torch.Tensor, shape=(B, M, 3)
        Array of vertex indices for each face.
    image_size : int, optional
        Image resulution, by default 128
    dist : float, optional
        Camera distance from the origin, by default 1.0
    elev : int, optional
        Eelevation of the camera viewpoint in degrees, by default 30
    azim : int, optional
        Azimuth of the camera viewpoint in degrees (180 is from the front),
        by default 150

    Returns
    -------
    normal_maps : torch.Tensor, shape=(B, image_size, image_size)
        The normal maps from the given viewpoint.
    """
    mesh = Meshes(verts=vertices.float(), faces=faces.float())
    # create texture
    # Initialize a camera.
    R, T = look_at_view_transform(dist, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=image_size, bin_size=[0, None][0], cull_backfaces=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    def phong_normal_shading(meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(
            fragments.pix_to_face, ones, faces_normals
        )
        return pixel_normals

    normal_maps = phong_normal_shading(mesh, fragments)
    normal_maps = normal_maps.min(dim=-2)[0].permute((0, 3, 1, 2))
    return normal_maps


def get_depth_maps(labels, root_path, image_size=128, dist=1.0, elev=30, azim=150):
    verts = []
    faces = []
    verts_path = f"{root_path}/vertices/"
    faces_path = f"{root_path}/faces/"
    for i, label in enumerate(labels):
        verts.append(load_npy_to_torch(verts_path + label + ".npy"))
        faces.append(load_npy_to_torch(faces_path + label + ".npy"))

    depth_maps = []
    for i in range(0, len(verts), 32):
        verts_pad = torch.nn.utils.rnn.pad_sequence(
            verts[i : i + 32], batch_first=True
        ).to(device)
        faces_pad = torch.nn.utils.rnn.pad_sequence(
            faces[i : i + 32], batch_first=True
        ).to(device)
        depth_maps.append(
            render_depthmap(
                verts_pad,
                faces_pad,
                image_size=image_size,
                dist=dist,
                elev=elev,
                azim=azim,
            ).unsqueeze(1)
        )
    depth_maps = torch.cat(depth_maps, dim=0)
    depth_maps = torch.clip(depth_maps, 0)
    return depth_maps * 2 / 3


def get_normal_maps(
    labels, root_path, image_size=128, dist=1.0, elev=30, azim=150, batch_size=32
):
    verts = []
    faces = []
    verts_path = f"{root_path}/vertices/"
    faces_path = f"{root_path}/faces/"
    for i, label in enumerate(labels):
        verts.append(load_npy_to_torch(verts_path + label + ".npy"))
        faces.append(load_npy_to_torch(faces_path + label + ".npy"))

    normal_maps = []
    for i in range(0, len(verts), batch_size):
        verts_pad = torch.nn.utils.rnn.pad_sequence(
            verts[i : i + batch_size], batch_first=True
        ).to(device)
        faces_pad = torch.nn.utils.rnn.pad_sequence(
            faces[i : i + batch_size], batch_first=True
        ).to(device)
        normal_maps.append(
            render_normalmap(
                verts_pad,
                faces_pad,
                image_size=image_size,
                dist=dist,
                elev=elev,
                azim=azim,
            )
        )
    normal_maps = torch.cat(normal_maps, dim=0)
    return normal_maps / 6 + 0.5


if __name__ == "__main__":
    with open("./data/datapaths.json", "r") as f:
        datapaths = json.loads(f.read())

    cats = ["02747177", "03001627", "03001627", "03001627", "03001627", "04379243"]
    ids = [
        "85d8a1ad55fa646878725384d6baf445",
        "235c8ef29ef5fc5bafd49046c1129780",
        "b4371c352f96c4d5a6fee8e2140acec9",
        "2c03bcb2a133ce28bb6caad47eee6580",
        "bdc892547cceb2ef34dedfee80b7006",
        "142060f848466cad97ef9a13efb5e3f7",
    ]
    v = []
    c = []
    f = []
    for cat, id in zip(cats, ids):
        cad_vertices, cad_colors, cad_faces, _ = load_shapenet_model(
            id, cat, datapaths["shapenet"]
        )
        v.append(torch.as_tensor(cad_vertices, device=device))
        c.append(torch.as_tensor(cad_colors, device=device))
        f.append(torch.as_tensor(cad_faces, device=device))

    v = torch.nn.utils.rnn.pad_sequence(v, batch_first=True).to(device)
    c = torch.nn.utils.rnn.pad_sequence(c, batch_first=True).to(device)
    f = torch.nn.utils.rnn.pad_sequence(f, batch_first=True).to(device)
    for img, cats, ids in zip(render_thumbnails(v, c, f), cats, ids):
        img.save(f"{cat}_{id}__thumb.png")

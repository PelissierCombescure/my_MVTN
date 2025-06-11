import numpy as np
import open3d as o3d
import torch
import os 
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import look_at_view_transform, OpenGLOrthographicCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes

import sys 
# Get the absolute path of MYPOINTCLIP (two levels up)
root_dir = os.path.abspath("/home/mpelissi/CLIP/myCLIP2Point/")
sys.path.append(root_dir)

from render.render import batch_tensor

def my_render_mesh(points, normals, a_faces, azim, elev, dist, views, image_size, device):
    rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0])).to(points.device)
    rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0])).to(points.device)
    points = points @ rota1 @ rota2
    normals = normals @ rota1 @ rota2

    R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
            elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))

    cameras = OpenGLOrthographicCameras(device=points.device, R=R, T=T, znear=0.01)

    raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=False,
                bin_size=0
            )
    renderer = MeshRasterizer(raster_settings=raster_settings)
    # sommets dans repère monde
    meshes = Meshes(verts=[points.squeeze().to(device)], faces=[torch.tensor(a_faces).to(device)])
    fragments = renderer(meshes.extend(views), cameras=cameras)
    
    return fragments, points, cameras, meshes, normals, R, T

import numpy as np
import open3d as o3d
import torch
import os 
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer import look_at_view_transform, OpenGLOrthographicCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes


def my_render_meshes(meshes, color, azim, elev, dist, lights, background_color=(1.0, 1.0, 1.0)):
        c_batch_size = len(meshes)
        verts = [msh.verts_list()[0].cuda() for msh in meshes]
        faces = [msh.faces_list()[0].cuda() for msh in meshes]
        new_meshes = Meshes(
                verts=verts,
                faces=faces,
                textures=None)
        max_vert = new_meshes.verts_padded().shape[1]

        new_meshes.textures = Textures(
                verts_rgb=color.cuda()*torch.ones((c_batch_size, max_vert, 3)).cuda())

        # Create a Meshes object for the teapot. Here we have only one mesh in the batch.
        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
                elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))
        R, T = check_and_correct_rotation_matrix(
                R, T, EXAHSTION_LIMIT, azim, elev, dist)

        cameras = OpenGLPerspectiveCameras(
                device="cuda:{}".format(torch.cuda.current_device()), R=R, T=T)
        camera = OpenGLPerspectiveCameras(device="cuda:{}".format(torch.cuda.current_device()), R=R[None, 0, ...], T=T[None, 0, ...])

        raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=self.faces_per_pixel,
                cull_backfaces=self.cull_backfaces,
        )

        renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
                shader=HardPhongShader(blend_params=BlendParams(background_color=background_color), device=lights.device, cameras=camera, lights=lights)
        )
        new_meshes = new_meshes.extend(self.nb_views)

        rendered_images = renderer(new_meshes, cameras=cameras, lights=lights)

        rendered_images = unbatch_tensor(
                rendered_images, batch_size=self.nb_views, dim=1, unsqueeze=True).transpose(0, 1)

        rendered_images = rendered_images[..., 0:3].transpose(2, 4).transpose(3, 4)
        return rendered_images, cameras, new_meshes, meshes, R, T

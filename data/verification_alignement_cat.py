import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import ast
import trimesh
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
from scipy.spatial.transform import Rotation 
from tqdm import tqdm 
import random

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    HardPhongShader, OpenGLOrthographicCameras, PointsRasterizationSettings,
    PointsRasterizer, NormWeightedCompositor, DirectionalLights)
from pytorch3d.renderer.mesh import Textures
from pytorch3d.structures import Meshes

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
# Add MYPOINTCLIP to sys.path
sys.path.append(root_dir)
from mvtorch.view_selector import MVTN
from mvtorch.mvrenderer import MVRenderer
from mvtorch.utils import torch_color, torch_center_and_normalize
from utils_mvtn import write_obj_with_color, save_colored_obj_with_faces, calculer_aires_triangles_batch, create_rotation_matrix, get_rotation
from mvtorch.data import ScanObjectNN, CustomDataLoader, ModelNet40, rotation_matrix

nb_views = 12
views_config = "circular"
bs = 1
data_dir = "/media/disk1/mpelissi-data/Aligned/modelnet40_manually_aligned/"#"/home/mpelissi/Dataset/ModelNet40/"
category = "all"
simplified_mesh = True
angles = np.linspace(0, 360, 73)
name_file = None

categories = ['plant',
 'mantel',
 'cup',
 'bookshelf',
 'wardrobe',
 'table',
 'piano',
 'bowl',
 'chair',
 'xbox',
 'tv_stand',
 'laptop',
 'vase',
 'cone',
 'car',
 'tent',
 'stairs',
 'toilet',
 'bottle',
 'keyboard',
 'range_hood',
 'dresser',
 'bathtub',
 'sink',
 'sofa',
 'guitar',
 'bench',
 'door',
 'radio',
 'monitor',
 'person',
 'desk',
 'curtain',
 'airplane',
 'bed',
 'lamp',
 'night_stand',
 'glass_box',
 'stool',
 'flower_pot']
categories = categories+categories
# Shuffle categories
random.shuffle(categories)

# Global parameters
points_per_pixel=1; points_radius=0.02; image_size=224

# Model 3D remeshing iso
dir_remeshing =  "/home/mpelissi/Dataset/ModelNet40_remeshing_iso"
dir_output = "/home/mpelissi/MVTN/my_MVTN/outputs"
all_mesh_iso = glob.glob(os.path.join(dir_remeshing, "*/*/*.obj")); print(f"🔎​​​ Number of meshes found in {dir_remeshing} : {len(all_mesh_iso)}")
# Data loading
dset_train = ModelNet40(data_dir=data_dir, split='train', samples_per_class=None, category=category, simplified_mesh=simplified_mesh)
dset_test = ModelNet40(data_dir=data_dir, split='test', samples_per_class=None, category=category, simplified_mesh=simplified_mesh)
train_loader = CustomDataLoader(dset_train, batch_size=bs, shuffle=True, drop_last=False, pin_memory=True)
test_loader = CustomDataLoader(dset_test, batch_size=bs, shuffle=False, drop_last=False, pin_memory=True)
print(f"🔎​​​ Which categories are used ? 🚨​ {category} 🚨​\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');  print("💻 device : ", device)
mvtn = MVTN(nb_views, views_config).cuda()
mvrenderer = MVRenderer(nb_views=nb_views, return_mapping=False, pc_rendering=False).cuda()

dset = dset_train; loader = train_loader
for category in tqdm(categories):
    print(f"🔎​​​ Category : {category}")
    for i, (targets, ref_mesh, points, names) in enumerate(loader):
        if any([category in p for p in names]) :  # si le fichier recherché est dans le batch
            # reference mesh   
            azim, elev, dist = mvtn(points, c_batch_size=len(targets))
            ref_images, _ = mvrenderer(ref_mesh, points, azim=azim, elev=elev, dist=dist)
            ref_vertices = ref_mesh[0].verts_list()[0].cpu().detach().numpy()

            # Get mesh_iso + projections
            path_mesh_modelnet40 = names[0]; print(f"🔎​​​ Path to the reference mesh : {path_mesh_modelnet40}")
            #path_mesh_iso = [p for p in all_mesh_iso if os.path.basename(path_mesh_modelnet40).split('_SM')[0] in p][0] # avec dataset Modelnet40
            path_mesh_iso = [p for p in all_mesh_iso if os.path.basename(path_mesh_modelnet40).split('.')[0] in p][0] # avec dataset Aligned Modelnet40
            cat = path_mesh_iso.split('/')[-3]; type = path_mesh_iso.split('/')[-2]; name = path_mesh_iso.split('/')[-1].split('.')[0]
            print("Dans dataset Modelnet40 remeshing : ", cat, type, name)
            filename = f"projections_{cat}_{name}.png"
            # Check if file already exists in output directory
            if os.path.exists(os.path.join("/home/mpelissi/MVTN/my_MVTN/data/outputs", filename)):
                print(f"⚠️ File {filename} already exists in the output directory. Skipping...")
                continue
            else :  break

    # load mesh from remeshing iso
    print(f"🔎​​​ Loading mesh from {os.path.join(dir_remeshing, cat, type, name + '.obj')}")
    mesh_iso = trimesh.load(os.path.join(dir_remeshing, cat, type, name + ".obj"), force='mesh')        
    if not dset.is_rotated:
        angle = dset.initial_angle; rot_axis = [1, 0, 0]
    else :
        angle = rotations_list[index]["rot_theta"][0]
        rot_axis = [dset.rotations_list[index]["rot_x"][0], dset.rotations_list[index]["rot_y"][0], dset.rotations_list[index]["rot_z"][0]]

    verts = np.array(mesh_iso.vertices.data.tolist())
    faces = np.array(mesh_iso.faces.data.tolist())
    if dset.cleaned_mesh:
        if dset.correction_factors[index] == -1 and dset.simplified_mesh: # flip the faces 
            faces[:,0] , faces[:,2] = faces[:,2] , faces[:,0]

    verts = rotation_matrix(rot_axis, angle).dot(verts.T).T
    verts = torch_center_and_normalize(torch.from_numpy(verts).to(torch.float), p=dset.dset_norm); nb_verts = verts.shape[0]
    faces = torch.from_numpy(faces); nb_faces = faces.shape[0]; 
    array_faces = faces.cpu().numpy() 
    array_normals = mesh_iso.vertex_normals
    normals = torch.tensor(array_normals, dtype=torch.float32).cuda()

    # Initialize each vertex to be white in color.
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts],faces=[faces],textures=textures)

    ## rotation to aligned 
    proj_vertices = mesh.verts_list()[0].cpu().detach().numpy()
    rotations_list = get_rotation(proj_vertices, ref_vertices)

    for rot in rotations_list:
        if rot[0] == 'Z':
            matrix = create_rotation_matrix('Y', rot[1])
            proj_vertices = torch.from_numpy(proj_vertices @ matrix[:3, :3])
            #proj_vertices = torch_center_and_normalize(proj_vertices.to(torch.float), p=dset.dset_norm)
            mesh_2 = Meshes(verts=[proj_vertices.to(torch.float).to(device)], faces=[faces.to(device)], textures=textures)
            print(f"🔎​​​ Rotation applied to aligned mesh with angle {rot[1]}°")
            
    ## Projection
    background_color = torch_color(mvrenderer.background_color, mvrenderer.background_color, max_lightness=True,).cuda()
    color = mvrenderer.rendering_color(None)

    if not mvrenderer.pc_rendering: # Mesh rendering
        lights = DirectionalLights(device=background_color.device, direction=mvrenderer.light_direction(azim, elev, dist))
        rendered_images, cameras, _, mesh_world, R, T, renderer = mvrenderer.render_meshes(meshes=mesh, color=color, azim=azim, elev=elev, dist=dist, lights=lights, background_color=background_color, projection = True)
        rendered_images_rot, _, _, _, _, _, _ = mvrenderer.render_meshes(meshes=mesh_2, color=color, azim=azim, elev=elev, dist=dist, lights=lights, background_color=background_color, projection = True)
        
    # transform xyz to the camera view coordinates
    cam_points = cameras.get_world_to_view_transform().transform_points(mesh_world.verts_list()[0].unsqueeze(0))
    cam_points_np = cam_points.cpu().numpy()
    # transform xyz to the camera view coordinates
    cam_normals = cameras.get_world_to_view_transform().transform_normals(normals.unsqueeze(0))
    cam_normals_np = cam_normals.cpu().numpy()
    # Cartes de profondeur
    fragments = renderer.rasterizer(mesh_world, cameras=cameras)
    depthmaps_np = fragments.zbuf[:,:,:,0].cpu().numpy()

    fig, axes = plt.subplots(3, nb_views, figsize=(20, 8))
    fig.suptitle(f"{name}: My views from remeshing_iso and reference from {data_dir}", fontsize=16)
    for i in range(bs):
        for j in range(nb_views):
            # Get image from rendered_images and move to CPU, take first 3 channels
            img = rendered_images[i, j].cpu().detach().permute(1, 2, 0)[:, :, :3]
            axes[0,j].imshow(img); axes[0,j].axis('off')
            axes[0,j].set_title(f'My View {j+1}')
            
            img_rot = rendered_images_rot[i, j].cpu().detach().permute(1, 2, 0)[:, :, :3]
            axes[1,j].imshow(img_rot); axes[0,j].axis('off')
            axes[1,j].set_title(f'My View {j+1} rot')
            if j == 0: cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[1, 0], orientation='vertical', fraction=0.046, pad=0.04)
            
            img_ref = ref_images[i, j].cpu().detach().permute(1, 2, 0)[:, :, :3]
            axes[2,j].imshow(img_ref), axes[2,j].axis('off')
            axes[2,j].set_title(f'Reference View {j+1}')
    plt.tight_layout()
    
    plt.savefig(os.path.join("/home/mpelissi/MVTN/my_MVTN/data/outputs", filename), bbox_inches='tight', dpi=300)
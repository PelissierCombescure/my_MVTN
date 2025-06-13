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
from utils_mvtn import write_obj_with_color, save_colored_obj_with_faces, calculer_aires_triangles_batch
from mvtorch.data import ScanObjectNN, CustomDataLoader, ModelNet40, rotation_matrix


# python3 run_projections_mvtn.py -nb_views 12 -view_config circular -dir_output /media/disk1/mpelissi-data/MVTN/circular-12/Projections


parser = argparse.ArgumentParser(description='Train a multi-view network for classification.')
parser.add_argument('-nb_views', '--nb_views', type=int, required=True, help='Number of views')
parser.add_argument('-view_config', '--view_config', required=True, type=str)
parser.add_argument('-dir_output', '--dir_output', required=True, type=str)
parser.add_argument('--split', type=str, choices=['train', 'test'], required=True)
args = parser.parse_args()

nb_views = args.nb_views
views_config = args.view_config

###### PATHS
dir_remeshing =  "/home/mpelissi/Dataset/ModelNet40_remeshing_iso"
dir_output = args.dir_output
all_mesh_iso = glob.glob(os.path.join(dir_remeshing, "*/*/*.obj")); print(f"🔎​​​ Number of meshes found in {dir_remeshing} : {len(all_mesh_iso)}")

###### Global parameters
bs = 1
data_dir = "/home/mpelissi/Dataset/ModelNet40/"
category = "all"
simplified_mesh = True
points_per_pixel=1; points_radius=0.02; image_size=224

###### Dataset
dset_train = ModelNet40(data_dir=data_dir, split='train', samples_per_class=None, category=category, simplified_mesh=simplified_mesh)
dset_test = ModelNet40(data_dir=data_dir, split='test', samples_per_class=None, category=category, simplified_mesh=simplified_mesh)
train_loader = CustomDataLoader(dset_train, batch_size=bs, shuffle=True, drop_last=False, pin_memory=True)
test_loader = CustomDataLoader(dset_test, batch_size=bs, shuffle=False, drop_last=False, pin_memory=True)
print(f"🔎​​​ Which categories are used ? 🚨​ {category} 🚨​\n")
print(f"📦 ​Dataset train : {len(dset_train)} samples")
print(f"📦​​​ Dataset test : {len(dset_test)} samples\n")

#### Networks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');  print("💻 device : ", device)
mvtn = MVTN(nb_views, views_config).cuda()
mvrenderer = MVRenderer(nb_views=nb_views, return_mapping=False, pc_rendering=False).cuda()
background_color = torch_color(mvrenderer.background_color, mvrenderer.background_color, max_lightness=True,).cuda()
color = mvrenderer.rendering_color(None)
print(f"Loading MVTN with {nb_views} views and config : {views_config} -OK\n")

#for dset, loader in zip([dset_train, dset_test], [train_loader, test_loader]):
if True :
    # Get the index of the current dataset split
    if args.split == "train": dset = dset_train; loader = train_loader
    elif args.split == "test": dset = dset_test; loader = test_loader
    print(f"🔎​​​ Dataset {dset.split} - {len(loader)} samples")
    total_batches = len(loader); bar_len = 50
    
    for i, (_, _, points, names) in enumerate(loader):   
        azim, elev, dist = mvtn(points, c_batch_size=len(names))
        #### correspondant de names dans all_mesh_iso
        path_mesh_modelnet40 = names[0]
        path_mesh_iso = [p for p in all_mesh_iso if os.path.basename(path_mesh_modelnet40).split('_SM')[0] in p][0]
        cat = path_mesh_iso.split('/')[-3]; type = path_mesh_iso.split('/')[-2]; name = path_mesh_iso.split('/')[-1].split('.')[0]
        
        ################################
        #### Get item as MVTN
        mesh_iso = trimesh.load(os.path.join(dir_remeshing, cat, type, name + ".obj"), force='mesh')   
        if True :     
            if not dset.is_rotated: angle = dset.initial_angle; rot_axis = [1, 0, 0]
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
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = Textures(verts_rgb=verts_rgb)
        mesh = Meshes(verts=[verts],faces=[faces],textures=textures)
        
        ################################
        #### Projection 
        if not mvrenderer.pc_rendering: # Mesh rendering
            lights = DirectionalLights(device=background_color.device, direction=mvrenderer.light_direction(azim, elev, dist))
            _, cameras, _, mesh_world, R, T, renderer = mvrenderer.render_meshes(meshes=mesh, color=color, azim=azim, elev=elev, dist=dist, lights=lights, background_color=background_color, projection = True)
    
        # transform xyz to the camera view coordinates
        cam_points = cameras.get_world_to_view_transform().transform_points(mesh_world.verts_list()[0].unsqueeze(0))
        cam_points_np = cam_points.cpu().numpy()
        # transform xyz to the camera view coordinates
        cam_normals = cameras.get_world_to_view_transform().transform_normals(normals.unsqueeze(0))
        cam_normals_np = cam_normals.cpu().numpy()
        # Cartes de profondeur
        fragments = renderer.rasterizer(mesh_world, cameras=cameras)
        depthmaps_np = fragments.zbuf[:,:,:,0].cpu().numpy()
        
        # pix_to_face is of shape (N, H, W, 1)
        pix_to_face = fragments.pix_to_face
        pix_to_face = fragments.pix_to_face[..., 0] 
        
        ################################
        #### Data par vue
        visible_faces_per_view = []
        visible_verts_per_view = []

        for b in range(pix_to_face.shape[0]):
            array_pt_cloud_b = cam_points_np[b]
            array_normals_b = cam_normals_np[b]
            
            # 1. Get valid face indices (ignore -1 = background)
            face_ids = pix_to_face[b]
            idx_visible_faces_b = torch.unique(face_ids[face_ids >= 0]) % nb_faces  # remove -1 (background)
            idx_visible_faces_b_np = idx_visible_faces_b.cpu().numpy()
            visible_faces_per_view.append(idx_visible_faces_b)
             
            
            # 2. Extract visible vertices from those faces
            idx_visible_verts_b = torch.unique(torch.tensor(array_faces)[idx_visible_faces_b])
            idx_visible_verts_b_np = idx_visible_verts_b.cpu().numpy()
            visible_verts_per_view.append(idx_visible_verts_b)   
                    
            obj_filename = f"{os.path.join(dir_output, cat, type, name)}_myview_{b+1}_color.obj"
            write_obj_with_color(array_pt_cloud_b, array_faces, idx_visible_verts_b_np, obj_filename)
            
            ## Angle sommets visibles
            cos_angle_visible_vertex_b = np.full(array_pt_cloud_b.shape[0], np.nan)
            # sommets et normales visibles
            v_visible = array_pt_cloud_b[idx_visible_verts_b_np]
            n_visible = array_normals_b[idx_visible_verts_b_np]
            n_norms = np.linalg.norm(n_visible, axis=1, keepdims=True)
            n_norms[np.where(n_norms == 0)] = 1e-10 # quelques normales sont nulles
            # normalisation
            n_visible_norm = n_visible / n_norms
            # vecteurs directeurs
            D = -v_visible
            D /= np.linalg.norm(D, axis=1, keepdims=True)
            cos_alpha = np.abs(np.sum(D * n_visible_norm, axis=1))
            cos_angle_visible_vertex_b[idx_visible_verts_b_np] = cos_alpha   
            obj_filename_angle = f"{os.path.join(dir_output, cat, type, name)}_myview_{b+1}_angles.obj"
            save_colored_obj_with_faces(obj_filename_angle, array_pt_cloud_b, cos_angle_visible_vertex_b, array_faces)
            
            
            # Surface Totale 3D
            surface3D_b = np.sum(calculer_aires_triangles_batch(array_pt_cloud_b, array_faces))
            # Surface visible de la projection courante
            surface3D_visible_b = np.sum(calculer_aires_triangles_batch(array_pt_cloud_b, array_faces[idx_visible_faces_b_np]))
            
            data_output_path = os.path.join(dir_output, cat, type, name+"_cam"+str(b+1)+"_data.npz")

            np.savez_compressed(
                data_output_path,
                array_pt_cloud = array_pt_cloud_b,
                array_normals = array_normals_b,
                dephtmap = depthmaps_np[b],
                visible_vertex_idx = idx_visible_verts_b_np,
                visible_faces = idx_visible_faces_b_np,
                cos_angles = cos_angle_visible_vertex_b,
                surface3D = surface3D_b,
                surface3D_visible = surface3D_visible_b)
    
        # Update progress bar once per 12-name batch
        filled_len = int(bar_len * i // total_batches)
        bar = '█' * filled_len + '-' * (bar_len - filled_len)
        percent = (i / total_batches) * 100
        sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% ({i}/{total_batches})')
        sys.stdout.flush()

                        
        


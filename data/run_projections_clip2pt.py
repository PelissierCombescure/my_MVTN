import os 
import sys
import torch
import ast
import trimesh
from tqdm import tqdm
import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime 


from data.functions_pov_mvtn import *

# Get the absolute path of MYPOINTCLIP (two levels up)
root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add MYPOINTCLIP to sys.path
sys.path.append(root_dir)

from datasets import ModelNet10, ModelNet40Align, ModelNet40Ply, ScanObjectNN
from datasets.modelnet40_align import ModelNet40Ply_limper
from datasets.modelnet40_align import cats, idx_cats
from render.selector import Selector
from render.render import *
from utils_clip2point import read_state_dict, read_ply, create_rotation_matrix, pcshow, write_obj_with_color, calculer_aires_triangles_batch, save_colored_obj_with_faces, read_paths_from_txt


# Model 3D remeshing iso
dir_remeshing =  "/home/mpelissi/Dataset/ModelNet40_remeshing_iso"
dir_output = "/media/disk1/mpelissi-data/CLIP/CLIP2Point/Projections-24pov/Limper-3046"
print("dir_output : ", dir_output)

# correspondance entre les paths de obj et ply 
correspondence_df = pd.read_csv("/home/mpelissi/CLIP/myCLIP2Point/data/correspodnances-obj-ply.csv")

# correspondance entre obj et ply : rotation
rotations = pd.read_csv("/home/mpelissi/CLIP/myCLIP2Point/data/modelnet40-remeshing_to_ply-limper.csv")

# Global parameters
views = 24; points_per_pixel=1; points_radius=0.02; image_size=224

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');  print("💻 device : ", device)
selector = Selector(views, 0).to(device)

def run_process(path_ply):
    try :
    
        path_obj = correspondence_df[correspondence_df['path_ply'] == path_ply]['path_obj'].values[0]
        cat = path_obj.split('/')[-3]; type = path_obj.split('/')[-2]; name = path_obj.split('/')[-1]
        
        ###################################################################
        ### Aligner OBJ et PLY
        ## Récupération des coordonnées des sommets, des faces et des normales
        mesh_init = trimesh.load_mesh(os.path.join(dir_remeshing, path_obj+".obj"))
        ## Rotation pour aligner avec les PLY-limper
        rotation_to_apply = ast.literal_eval(rotations[rotations['path_ply'] == path_ply]['transformation'].values[0])
        mesh_aligned = mesh_init.copy()
        for transf in rotation_to_apply:
            angle = transf[1]
            matrix = create_rotation_matrix('Z', angle)    
            mesh_aligned.apply_transform(matrix)
            
        ## Récupération des coordonnées des sommets, des faces et des normales
        array_coords_aligned = np.array(mesh_aligned.vertices); nb_vertices = len(array_coords_aligned)
        array_normals_aligned = np.array(mesh_aligned.vertex_normals)
        array_faces = np.array(mesh_aligned.faces); nb_faces = len(array_faces)
        
        ###################################################################
        ## Convert to a NumPy array and take only XYZ coordinates
        points_aligned = torch.tensor(array_coords_aligned, dtype=torch.float32).cuda()
        normals_aligned = torch.tensor(array_normals_aligned, dtype=torch.float32).cuda()
        # Ensure the point cloud has the correct shape [B, N, 3]
        B = 1  # Batch size
        points_aligned = points_aligned.unsqueeze(0)  # Add batch dimension -> [1, N, 3]
        normals_aligned = normals_aligned.unsqueeze(0)  # Add batch dimension -> [1, N, 3]
        
        ###################################################################
        ### Projection
        # Cameras positions
        c_views_azim, c_views_elev, c_views_dist = selector(points_aligned)
        # Depthmaps
        fragments, world_points, cameras, world_mesh, world_normals,  Rs, Ts  = my_render_mesh(points_aligned, normals_aligned, array_faces, c_views_azim, c_views_elev, c_views_dist, views, image_size, device)
        # transform xyz to the camera view coordinates
        cam_points = cameras.get_world_to_view_transform().transform_points(world_points)
        cam_points_np = cam_points.cpu().numpy()
        # transform xyz to the camera view coordinates
        cam_normals = cameras.get_world_to_view_transform().transform_normals(world_normals)
        cam_normals_np = cam_normals.cpu().numpy()
        # Cartes de profondeur
        depthmaps_np = fragments.zbuf[:,:,:,0].cpu().numpy()
            
        # pix_to_face is of shape (N, H, W, 1)
        pix_to_face = fragments.pix_to_face  # Expected: [N, H, W, 1]
        pix_to_face = fragments.pix_to_face[..., 0]  # Expected: [N, H, W]

        visible_faces_per_view = []
        visible_verts_per_view = []
        #faces = world_mesh.faces_packed()  # (F, 3)

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
            # Step 1: Save all array data in a compressed .npz file
            np.savez_compressed(
                data_output_path,
                array_pt_cloud = array_pt_cloud_b,
                array_normals = array_normals_b,
                dephtmap = depthmaps_np[b],
                visible_vertex_idx = idx_visible_verts_b_np,
                visible_faces = idx_visible_faces_b_np,
                cos_angles = cos_angle_visible_vertex_b,
                surface3D = surface3D_b,
                urface3D_visible = surface3D_visible_b)
        
        return 'ok', path_ply, path_obj, None
    
    except Exception as e:
        print(f"Error processing {path_ply}: {e}")
        return 'error', path_ply, path_obj, str(e)
        
    
#############################################################"
#### MAIN
paths_files = read_paths_from_txt("/home/mpelissi/CLIP/myCLIP2Point/data/paths_files_Ply_limper.txt")
results = []
for path_ply in tqdm(paths_files):
    results.append(run_process(path_ply))

# Create filename with current timestamp
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_name = os.path.join("/home/mpelissi/CLIP/myCLIP2Point/weights/error", f"projections_{timestamp}.txt")
with open(file_name, "w") as file:
    file.write(f"Date: {timestamp} - Error during saving\n")
    for verdict, path_ply, path_obj, pbl in results:
        file.write(f"{verdict}: {path_ply} -- {path_obj} -- {pbl}\n")


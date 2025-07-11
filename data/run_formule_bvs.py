import sys
import pickle
import tqdm 
import glob
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime 
import argparse


parser = argparse.ArgumentParser(description='Train a multi-view network for classification.')
parser.add_argument('-nb_views', '--nb_views', type=int, required=True, help='Number of views')
parser.add_argument('-view_config', '--view_config', required=True, type=str)
parser.add_argument('-dir_projection', '--dir_projection', required=True, type=str)
parser.add_argument('-dir_output', '--dir_output', required=True, type=str)
args = parser.parse_args()

# nb vue
nb_view = args.nb_views
view_config = args.view_config
dir_projection = args.dir_projection
dir_output = args.dir_output

# Data sur les projections
# Output
print(f"Input directory: {dir_projection}")
print(f"Output directory: {dir_output}")

# Model 3D remeshing iso
dir_remeshing = "/home/mpelissi/Dataset/ModelNet40_remeshing_iso"
paths_iso = glob.glob(os.path.join(dir_remeshing, "*/*/*.obj")); 
paths_projection = glob.glob(os.path.join(dir_projection, "*/*/*cam1_data.npz"))

dir_saillance = "/media/disk1/mpelissi-data/Modelnet40_limper_remeshing_iso_lce"
paths_limper = glob.glob(os.path.join(dir_saillance, "*/*/*.csv")); print(f"Number of saillance files: {len(paths_limper)}")
print(f"🔎​​​ Parmi les {len(paths_iso)} mesh :\nOn a les projections des {nb_view} vues {view_config} de {len(paths_projection)} mesh")

names_limper = [os.path.basename(path).split("_lce")[0] for path in paths_limper]
names_projection = [os.path.basename(path).split("_cam1_data.npz")[0] for path in paths_projection]
names_commun = list(set(names_limper) & set(names_projection))
print(f"Parmi ces {len(paths_projection)} mesh, on a les saillances de {len(names_commun)} mesh")


paths_NAN = []
paths_NAN_ok = []
seuil_NAN = 0.1

results = []
for path_limper in tqdm.tqdm(paths_limper):
    name = os.path.basename(path_limper).split('_lce')[0]    
    try:
        #if True:
        # Informations sur le mesh
        cat = path_limper.split('/')[-3]; type = path_limper.split('/')[-2]
        if not os.path.exists(os.path.join(dir_output, cat, type, name+"_bvs.pkl")):            
            # Saillance limper associée
            df_saillance_limper = pd.read_csv(path_limper, header=None)
            # colonne Combined_Saliency_Norm
            saillance_limper = np.array([float(s) for s in list(df_saillance_limper[1])[1:]])
            ## les valeurs des saillances sont déjà normalisées entre 0 et 1

            dict_scores = {}
            too_much_nan = False; contain_nan = False
            suffix = ""
            # Pour chaque pov
            for i in range(0, nb_view): 
                #print(k)
                # data du pov k
                path_npz_cam_i = os.path.join(dir_projection, cat, type, name+"_cam"+str(i+1)+"_data.npz")
                # Load the .npz file and pkl file
                data_cam_i = np.load(path_npz_cam_i)
                # sommets visibles
                sommets_visible = data_cam_i['visible_vertex_idx']#; print(len(sommets_visible))
                #mask_sommets_visible = data_cam_i['visible_vertex_bin']
                cos_angles = data_cam_i['cos_angles']
                # Somme [limper*angle]
                saillance_limper_visible = saillance_limper[sommets_visible]
                # Vérification des NaN
                if np.isnan(saillance_limper_visible).any() : 
                    contain_nan = True
                    taux_nan = len(np.where(np.isnan(saillance_limper_visible))[0])/len(saillance_limper_visible)
                    saillance_limper_visible[np.isnan(saillance_limper_visible)] = 0.0
                else : taux_nan = 0
                # Terme de saillance
                terme_somme_saillance = np.sum(saillance_limper_visible*cos_angles[sommets_visible])
                # Surface 3D + 'normalisation' : on divise par la surface 3D totale de l'objet
                terme_surface3d = data_cam_i['surface3D_visible']/data_cam_i['surface3D']
                
                # Sauvegardes des termes
                dict_scores['cam_'+str(i+1)] = {'terme_surface3d': terme_surface3d, 'saillance': terme_somme_saillance}
                                
                # Trop de NAN
                if contain_nan & (taux_nan > seuil_NAN):
                    too_much_nan = True
                    break
            
            if not too_much_nan:
                # Normalisation du terme de saillance pour les nb_view povs
                max_terme_saillance = np.max([dict_scores['cam_'+str(i+1)]['saillance'] for i in range(0, nb_view)])
                dict_scores = {k: {'terme_surface3d': v['terme_surface3d'], 'saillance': v['saillance'], 
                                    'terme_saillance': v['saillance']/max_terme_saillance} for k, v in dict_scores.items()}

                # Scores = (surface3D + saillance) pour les nb_view povs
                for k, v in dict_scores.items():
                    dict_scores[k]['score'] =  v['terme_surface3d'] + v['terme_saillance']
                    
                # Normalisation des score --> / par la somme des scores
                sum_scores = np.sum([dict_scores[k]['score'] for k in dict_scores.keys()])
                for k, v in dict_scores.items():
                    dict_scores[k]['score_norm'] =  v['score']/sum_scores
                    
                # BVS 
                score_max = np.max([dict_scores[k]['score'] for k in dict_scores.keys()])
                dict_scores['bvs'] = [k for k in dict_scores.keys() if dict_scores[k]['score'] == score_max][0]
                # si plusieurs 
                if len([k for k in dict_scores.keys() if (('cam' in k) and (dict_scores[k]['score'] == score_max))]) > 1: print("Plusieurs pov", name)
                metadata = {
                    "name_limper": path_limper, "categorie": cat, "type": type, "name": name,
                    "bvs" : dict_scores['bvs'], "score_max": score_max, "score_max_norm": dict_scores[dict_scores['bvs']]['score_norm'], 
                    "scores": dict_scores}
                
                # Contient des nan MAIS pas bcp pour être ici
                if contain_nan : 
                    paths_NAN_ok.append(path_limper)
                    results.append(('nan mais ok', path_limper, "nan < seuil", too_much_nan))
                    suffix = "fewnan"
            
                else : # RAS pas de NAN
                    suffix = "ras"
                    results.append(('ok', path_limper, "RAS", too_much_nan))
                    
                with open(os.path.join(dir_output, cat, type, name+f"_bvs_{suffix}.pkl"), "wb") as f: pickle.dump(metadata, f)
                #print("Enregistrement de", os.path.join(dir_output, cat, type, name+"_bvs.pkl"))
                
            # Trop de NaN
            else:
                results.append(('too much nan', path_limper, "Contient TROP de NaN", too_much_nan))
                paths_NAN.append((path_limper, taux_nan))
                
        # Déjà traité
        else :
            results.append(("skip", path_limper, "Déjà traité", False))         
    
    except Exception as e:
        results.append(("pbl",path_limper, e, too_much_nan)) 
        
##############################################################################
# Write results to file
# List all files in thee current directory
files_in_directory = os.listdir('/home/mpelissi/MVTN/my_MVTN/data/error')
# Filter files containing the word 'error'
nb_error_files = len([file for file in files_in_directory if ((view_config in file) and ('bvs' in file))])

# fichiers avec nan mis à 0
with open(os.path.join('/home/mpelissi/MVTN/my_MVTN/data/error', f"bvs_nan_run{nb_error_files+1}_{view_config}_{nb_view}.txt"), "w") as f:
    f.write("Date: {:%Y-%m-%d %H:%M:%S} - Error during saving\n".format(datetime.now()))
    for p in paths_NAN:
        f.write(f"{p[0]} - Taux de NaN : {p[1]}\n")

if nb_error_files >= 0: 
    file_name = os.path.join('/home/mpelissi/MVTN/my_MVTN/data/error', f"error_bvs_run{nb_error_files+1}_{view_config}_{nb_view}.txt")
    
with open(file_name, "w") as file:
    file.write("Date: {:%Y-%m-%d %H:%M:%S} - Error during saving\n".format(datetime.now()))
    for verdict, name, err, bool_nan in results:
       file.write(f"{verdict}: {name} : {err} -- NAN ? : {bool_nan}\n")


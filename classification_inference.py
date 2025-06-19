import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import argparse
import pandas as pd
from mvtorch.data import ModelNet40, CustomDataLoader
from mvtorch.networks import MVNetwork
from mvtorch.view_selector import MVTN
from mvtorch.mvrenderer import MVRenderer
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json
import glob 

#  CUDA_VISIBLE_DEVICES=1 python3 classification_inference.py -nb_views 8

dir_results = "/home/mpelissi/MVTN/my_MVTN/results/"
data_dir = "/home/mpelissi/Dataset/ModelNet40"

def load_models(weights_dir, nb_views, num_classes=40, df=None):
    """
    Load the trained models with their best weights
    """
    # Initialize models
    mvnetwork = MVNetwork(num_classes, num_parts=None, mode='cls', net_name='resnet18').cuda()
    mvtn = MVTN(nb_views, list(df['views_config'])[0], canonical_distance=list(df['canonical_dist'])[0]).cuda()
    mvrenderer = MVRenderer(nb_views=nb_views, return_mapping=False, pc_rendering=list(df['pc_rendering'])[0]).cuda()

    # Load weights
    mvnetwork.load_state_dict(torch.load(os.path.join(weights_dir, 'mvnetwork_best.pth')))
    mvtn.load_state_dict(torch.load(os.path.join(weights_dir, 'mvtn_best.pth')))
    mvrenderer.load_state_dict(torch.load(os.path.join(weights_dir, 'mvrenderer_best.pth')))

    # Set to evaluation mode
    mvnetwork.eval()
    mvtn.eval()
    mvrenderer.eval()

    return mvnetwork, mvtn, mvrenderer

def evaluate_test_set(mvnetwork, mvtn, mvrenderer, test_loader, save_dir=None, nb_views=None, dir_weights=None, data_dir=None, category=None):
    """
    Evaluate the model on the test set and optionally save view parameters and rendered images
    """
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    views_parameters = {'save_dir': save_dir, 'nb_views': nb_views, 'dir_weights': dir_weights, 'data_dir': data_dir, 'category': category}
    
    with torch.no_grad():
        for _, (targets, meshes, points, names) in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Get view parameters
            azim, elev, dist = mvtn(points, c_batch_size=len(targets))
            
            # Render images
            rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
            
            # Get predictions
            outputs = mvnetwork(rendered_images)[0]
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets.cuda()).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

            # Save view parameters and rendered images for each sample in the batch if save_dir is provided
            if save_dir is not None:
                for i in range(len(targets)):
                    n = os.path.basename(names[i])
                    views_parameters[n] = {
                        'mesh_name': n,
                        'azimuth': azim[i].cpu().numpy().tolist(),
                        'elevation': elev[i].cpu().numpy().tolist(),
                        'distance': dist[i].cpu().numpy().tolist(),
                        #'rendered_images': rendered_images[i].cpu().numpy().tolist(),
                        'predicted': predicted[i].cpu().numpy().tolist(),
                        'target': targets[i].cpu().numpy().tolist()
                    }
    
    accuracy_inference = 100 * correct / total
    accuracy_bvs = -1
    views_parameters['accuracy'] = accuracy_inference

    # Save view parameters
    with open(os.path.join(save_dir, 'view_parameters.json'), 'w') as f:
        json.dump(views_parameters, f, indent=4)

    return accuracy_inference, accuracy_bvs, all_predictions, all_targets

def plot_confusion_matrix(predictions, targets, classes, save_dir):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(targets, predictions)
    
    # Create figure for raw counts confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_counts.png'))
    plt.close()
    
    # Calculate and plot percentage confusion matrix
    cm_percentage = np.full(cm.shape, 0.0)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            cm_percentage[r, c] = cm[r, c] / cm[r].sum() * 100
    
    # Create figure for percentage confusion matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Percentages)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_percentages.png'))
    plt.close()

def main():
    # Dossier de résultats
    current_time = datetime.now().strftime("%m-%d_%Hh%Mm%S")
    dir_inference = os.path.join(dir_results, 'inference', f"{current_time}")
    os.makedirs(dir_inference, exist_ok=True)
    print(f"​📁​  Results saved in: {dir_inference}")
    
    # fichier d'erreur 
    with open(os.path.join(dir_inference, 'errors.txt'), 'w') as error_file:
        error_file.write(f"{current_time} -- Errors during inference will be logged here.\n")
      
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply LCE to a mesh and save saliency values to CSV.')
    parser.add_argument('-nb_views', '--nb_views', type=int, default = -1, help='Number of views')
    parser.add_argument('-category', '--category', default='all', type=str, help='Model name')
    parser.add_argument('-view_config', '--view_config', default='toto', type=str, help='Model name')
    args = parser.parse_args()
    nb_views = args.nb_views
    category = args.category
    view_config = args.view_config
    
    if nb_views == -1: # inference sur tous les dossiers de overview.csv
        overview_file = "results/train/overview.csv"  
        overview_df = pd.read_csv(overview_file)
        # Create a copy of the overview dataframe to avoid modifying the original
        overview_df_update = overview_df.copy()
        overview_df_update['acc_bvs'] = None
        overview_df_update['acc_inference'] = None
        folders = [(p, f) for p in glob.glob(os.path.join(dir_results, 'train', '*')) 
                        for f in overview_df['folder_name'].unique() if f in p]
    #else : # un dossier en particulier
        ## TODO
    
    for folder_path, folder_name in tqdm(folders):
        #try :
        if True:
            print(f"\nProcessing folder: {folder_name} at path: {folder_path}")  
            # creation du dossier avec le meme nom que le dossier de l'overview
            save_dir = os.path.join(dir_inference, os.path.basename(folder_path))     
            os.makedirs(save_dir, exist_ok=True)
            row_idx = overview_df_update[overview_df_update['folder_name'] == folder_name].index[0]
            row_df = overview_df[overview_df['folder_name'] == folder_name]   
            # subfolder of best weights
            dir_best_weights = os.path.join(folder_path, 'best')
            if not os.path.exists(dir_best_weights):
                print(f"​❌​  No best weights found in {dir_best_weights}. Skipping this folder.")
                continue
        
            # Load dataset for class names
            dset_test = ModelNet40(data_dir=data_dir, split='test', samples_per_class=None, category=category, simplified_mesh=list(row_df['simplified_mesh'])[0])
            test_loader = CustomDataLoader(dset_test, batch_size=int(row_df['batch_size']), shuffle=False, drop_last=False, pin_memory=True)
        
            # Load models
            print("​🔃​  Loading models...")
            mvnetwork, mvtn, mvrenderer = load_models(dir_best_weights, nb_views=list(row_df['nb_views'])[0], num_classes=len(dset_test.classes), df=row_df)
            
            # Evaluate on test set and save view parameters
            print("🔎​  Evaluating on test set...")
            acc_inference, acc_bvs, predictions, targets = evaluate_test_set(mvnetwork, mvtn, mvrenderer, test_loader, save_dir=save_dir, nb_views=list(row_df['nb_views'])[0], dir_weights=dir_best_weights, data_dir = data_dir, category = category)
            print(f"🚀​  Test Accuracy inference: {acc_inference:.2f}%")
            print(f"🚀​  Test Accuracy with BVS: {acc_bvs:.2f}%")
            overview_df_update.loc[row_idx, 'acc_inference'] = acc_inference
            overview_df_update.loc[row_idx, 'acc_bvs'] = acc_bvs
            
            
            # Plot confusion matrices
            print("Generating confusion matrices...")
            plot_confusion_matrix(predictions, targets, dset_test.classes, save_dir=save_dir)
            print(f"​📊​  Confusion matrices saved at {save_dir} ")

            # Save the updated overview dataframe
            overview_df_update.to_csv(os.path.join(dir_inference, 'overview_inference.csv'), index=False)
                                              
        # except Exception as e:
        #     print(f"​❌​  Error processing folder {folder_name}: {e}")
        #     with open(os.path.join(dir_inference, 'errors.txt'), 'a') as error_file:
        #         error_file.write(f"Error processing folder {folder_name}: {e}\n")
        #     continue

if __name__ == "__main__":
    main()

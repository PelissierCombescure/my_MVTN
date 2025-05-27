import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import argparse
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

#  CUDA_VISIBLE_DEVICES=1 python3 classification_inference.py -nb_views 8


def load_models(weights_dir, nb_views, num_classes=40):
    """
    Load the trained models with their best weights
    """
    # Initialize models
    mvnetwork = MVNetwork(num_classes=num_classes, num_parts=None, mode='cls', net_name='resnet18').cuda()
    mvtn = MVTN(nb_views=nb_views).cuda()
    mvrenderer = MVRenderer(nb_views=nb_views, return_mapping=False)

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
        for batch_idx, (targets, meshes, points) in enumerate(tqdm(test_loader, desc="Evaluating")):
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
                    # Get mesh name from the dataset
                    mesh_name = test_loader.dataset.data_list[batch_idx * test_loader.batch_size + i].split('/')[-1].split('.')[0]
                    views_parameters[mesh_name] = {
                        'mesh_name': mesh_name,
                        'azimuth': azim[i].cpu().numpy(),
                        'elevation': elev[i].cpu().numpy(),
                        'distance': dist[i].cpu().numpy(),
                        'rendered_images': rendered_images[i].cpu().numpy().tolist(),
                        'predicted': predicted[i].cpu().numpy(),
                        'target': targets[i].cpu().numpy()
                    }
    
    accuracy = 100 * correct / total
    views_parameters['accuracy'] = accuracy

    # Save view parameters
    with open(os.path.join(save_dir, 'view_parameters.json'), 'w') as f:
        json.dump(views_parameters, f, indent=4)

    return accuracy, all_predictions, all_targets

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
    parser = argparse.ArgumentParser(description='Apply LCE to a mesh and save saliency values to CSV.')
    parser.add_argument('-nb_views', '--nb_views', type=int, required=True, help='Number of views')
    parser.add_argument('-category', '--category', default='all', type=str, required=True, help='Model name')
    args = parser.parse_args()
    nb_views = args.nb_views
    category = args.category

    # Set up paths
    dir_results = "/home/mpelissi/MVTN/my_MVTN/results/"
    current_time = datetime.now().strftime("%d-%m_%Hh%Mm%S")
    dir_inference = os.path.join(dir_results, 'inference', current_time)
    os.makedirs(dir_inference, exist_ok=True)
    print(f"\n​📁​  Results saved in: {dir_inference}")
    
    if nb_views == 1:
        dir_best_weights = os.path.join(dir_results, 'train/results_19-05_18h20m13/best')
        print(f"​🚨​  Using weights from: {dir_best_weights} because nb_views is 1")
    elif nb_views == 8:
        dir_best_weights = os.path.join(dir_results, 'train/results_14-05_18h05m48/best')
        print(f"🚨​  ​Using weights from: {dir_best_weights} because nb_views is 8")
    else : 
        raise ValueError(f"Invalid number of views: {nb_views}")
    
    # Load dataset for class names
    data_dir = "/home/mpelissi/Dataset/ModelNet40"
    #data_dir = "/home/mpelissi/Dataset/ModelNet40_remeshing_iso"
    if category != 'all':
        print(f"Category : {category}")
        data_dir = os.path.join(data_dir, category)
        
    dset_test = ModelNet40(data_dir=data_dir, split='test')
    test_loader = CustomDataLoader(dset_test, batch_size=16, shuffle=False, pin_memory=True)
    
    # Load models
    print("​🔃​  Loading models...")
    mvnetwork, mvtn, mvrenderer = load_models(dir_best_weights, nb_views, num_classes=len(dset_test.classes))
    
    # Evaluate on test set and save view parameters
    print("\n🔎​  Evaluating on test set...")
    accuracy, predictions, targets = evaluate_test_set(mvnetwork, mvtn, mvrenderer, test_loader, save_dir=dir_inference, nb_views=nb_views, dir_weights=dir_best_weights, data_dir = data_dir, category = category)
    print(f"🚀​  Test Accuracy: {accuracy:.2f}%")
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(predictions, targets, dset_test.classes, save_dir=dir_inference)
    


if __name__ == "__main__":
    main()

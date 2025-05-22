import os
import torch
import numpy as np
from mvtorch.data import ModelNet40, CustomDataLoader
from mvtorch.networks import MVNetwork
from mvtorch.view_selector import MVTN
from mvtorch.mvrenderer import MVRenderer
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

def predict_single_sample(mvnetwork, mvtn, mvrenderer, mesh, points):
    """
    Perform inference on a single sample
    """
    with torch.no_grad():
        # Get view parameters
        azim, elev, dist = mvtn(points.unsqueeze(0), c_batch_size=1)
        
        # Render images
        rendered_images, _ = mvrenderer(mesh.unsqueeze(0), points.unsqueeze(0), 
                                      azim=azim, elev=elev, dist=dist)
        
        # Get predictions
        outputs = mvnetwork(rendered_images)[0]
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        
        return predicted_class.item(), probabilities[0].cpu().numpy()

def evaluate_test_set(mvnetwork, mvtn, mvrenderer, test_loader):
    """
    Evaluate the model on the test set
    """
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for targets, meshes, points in tqdm(test_loader, desc="Evaluating"):
            azim, elev, dist = mvtn(points, c_batch_size=len(targets))
            rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
            outputs = mvnetwork(rendered_images)[0]
            
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets.cuda()).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    accuracy = 100 * correct / total
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
    current_time = datetime.now().strftime("%d-%m_%Hh%Mm%S")
    # Set up paths
    results_dir = "/home/mpelissi/MVTN/my_MVTN/results/"
    weights_dir = os.path.join(results_dir, 'train/results_14-05_18h05m48/best')
    dir_results = os.path.join(results_dir, 'inference', current_time)
    os.makedirs(dir_results, exist_ok=True)
    print(dir_results,'created')
    
    # Load dataset for class names
    data_dir = "/home/mpelissi/Dataset/ModelNet40"
    dset_test = ModelNet40(data_dir=data_dir, split='test')
    test_loader = CustomDataLoader(dset_test, batch_size=16, shuffle=False, pin_memory=True)
    
    # Load models
    print("Loading models...")
    mvnetwork, mvtn, mvrenderer = load_models(weights_dir, nb_views=8, num_classes=len(dset_test.classes))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    accuracy, predictions, targets = evaluate_test_set(mvnetwork, mvtn, mvrenderer, test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Plot confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix(predictions, targets, dset_test.classes, dir_results)
    
    print(f"\nResults saved in: {dir_results}")

if __name__ == "__main__":
    main()

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tqdm
import time 
import json
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import argparse
import pandas as pd
import shutil

import torch
print("CUDA available: ", torch.cuda.is_available())
gpu_name = torch.cuda.get_device_name(0)
print("GPU name: ", gpu_name)

from mvtorch.data import ScanObjectNN, CustomDataLoader, ModelNet40
from mvtorch.networks import MVNetwork
from mvtorch.view_selector import MVTN
from mvtorch.mvrenderer import MVRenderer
from earlyStop import EarlyStopping

from utils_mvtn import *

# CUDA_VISIBLE_DEVICES=1 python3 classification_train.py -nb_views 2 -epochs 100 -batch_size 16 -category all -data_dir /home/mpelissi/Dataset/ModelNet40/ -log_suffix circular -view_config spherical -mesh_rendering

parser = argparse.ArgumentParser(description='Train a multi-view network for classification.')
parser.add_argument('-nb_views', '--nb_views', type=int, required=True, help='Number of views')
parser.add_argument('-epochs', '--epochs', default=100, type=int, required=True, help='Number of epochs')
parser.add_argument('-batch_size', '--batch_size', default=1, type=int, required=True, help='Batch size')
parser.add_argument('-category', '--category',  type=str)
parser.add_argument('-data_dir', '--data_dir', required=True,  help='path to 3D dataset')
parser.add_argument('-log_suffix', '--log_suffix', required=True, type=str, help='Pour copier les bns logs')
parser.add_argument('-view_config', '--view_config', required=True, type=str)
# Add the new arguments for rendering
parser.add_argument('-pc_rendering', '--pc_rendering', action='store_true', help='Set pc_rendering to True (for point cloud rendering)')
parser.add_argument('-mesh_rendering', '--mesh_rendering', action='store_true', help='Set pc_rendering to False (for mesh rendering)')

args = parser.parse_args()
nb_views = args.nb_views
epochs = args.epochs
bs = args.batch_size
data_dir = args.data_dir
category = args.category
log_suffix = args.log_suffix
views_config = args.view_config

pc_rendering = None
if args.pc_rendering and args.mesh_rendering:
    parser.error("You cannot provide both -pc_rendering and -mesh_rendering. Please choose one.")
elif args.pc_rendering:
    pc_rendering = True
elif args.mesh_rendering:
    pc_rendering = False
else:
    # Raise an error if neither is provided
    parser.error("You must specify either -pc_rendering or -mesh_rendering.")

print(f"🔧​ Training configuration: {nb_views} views, {epochs} epochs, batch size {bs}")
print(f"​ Data directory used: {data_dir}\n")

########################################################################################## DATA
# Number of samples to load per class (for faster experimentation)
samples_per_class_train = None  # Adjust this number as needed
samples_per_class_test = None # Adjust this number as needed
########################################################################################## DATA

## Data loading
simplified_mesh=True
dset_train = ModelNet40(data_dir=data_dir, split='train', samples_per_class=samples_per_class_train, category=category, simplified_mesh=simplified_mesh)
dset_test = ModelNet40(data_dir=data_dir, split='test', samples_per_class=samples_per_class_test, category=category, simplified_mesh=simplified_mesh)
print(f"🔎​​​ Which categories are used ? 🚨​ {category} 🚨​\n")


if samples_per_class_train is not None:
    print(f"​Using {samples_per_class_train} samples per class in the training set")
    print(f"​Using {samples_per_class_test} samples per class in the testing set")
else :
    print(f"Using all samples in the training set : {len(dset_train)}")
    print(f"Using all samples in the testing set : {len(dset_test)}")
print(f"📝​​ Loaded {len(dset_train)} training samples and {len(dset_test)} testing samples\n")

## Optimize code 1/3
train_loader = CustomDataLoader(dset_train, batch_size=bs, shuffle=True, drop_last=False, pin_memory=True)
test_loader = CustomDataLoader(dset_test, batch_size=bs, shuffle=False, drop_last=False, pin_memory=True)

print((f"📦​ There are {len(train_loader)} batches of {bs} samples in the training set"))
print((f"📦 ​There are {len(test_loader)} batches of {bs} samples in the testing set"))

##########################################################################################
# Models x3
if True :
    ## Network initialization
    # Create backbone multi-view network (ResNet18)
    mvnetwork = MVNetwork(num_classes=len(dset_train.classes), num_parts=None, mode='cls', net_name='resnet18').cuda()

    # Create backbone optimizer
    weight_decay = 0.03  # Adjust weight decay as needed
    lr_opti = 0.00001
    optimizer = torch.optim.AdamW(mvnetwork.parameters(), lr=lr_opti, weight_decay=weight_decay)

    # Create view selector
    canonical_dist = 1.1 * 2  # Adjust canonical distance as needed
    mvtn = MVTN(nb_views, views_config, canonical_distance=canonical_dist).cuda()
    print(f"🔍​ View selector configuration: {views_config} with {nb_views} views")

    # Create optimizer for view selector (In case views are not fixed, otherwise set to None)
    if 'learned' in views_config:
        print("Using learned view selector")
        lr_mvtn_optimizer = 0.0001
        mvtn_optimizer = torch.optim.AdamW(mvtn.parameters(), lr=lr_mvtn_optimizer, weight_decay=weight_decay)
        opti_mvtn = 'setup'
    else :
        print("mvtn_optimizer is None !!!!!")
        mvtn_optimizer = None
        opti_mvtn = None
        lr_mvtn_optimizer = None         
 
    # Create multi-view renderer
    if pc_rendering:
        print("\n 🚨 Using point cloud rendering")
    else:
        print("\n 🚨 Using mesh rendering")
    mvrenderer = MVRenderer(nb_views=nb_views, return_mapping=False, pc_rendering=pc_rendering).cuda()
    # Create loss function for training
    criterion = torch.nn.CrossEntropyLoss()
    # Create early stopping mechanism
    patience = 10; min_delta = 0.001
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

##########################################################################################
## Training
# Create a directory with the current date and time
current_time = datetime.now().strftime("%m-%d_%Hh%Mm%S")
results_dir_current = os.path.join("results/train/", f'results_{current_time}-{gpu_name}')
os.makedirs(results_dir_current, exist_ok=True)
os.makedirs(os.path.join(results_dir_current, "best"), exist_ok=True)
#os.makedirs(results_dir_current+"/by_epoch", exist_ok=True)
print(f"\n📁​ Results directory: {results_dir_current}")
#path_model_configs = '/home/mpelissi/MVTN/my_MVTN/results/model_config.csv'


# Variables to track the best accuracy
best_accuracy = 0.0
best_epoch = 0

# Lists to store loss and accuracy for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Dictionary to store training parameters and results
training_info = {
    'gpu_name': gpu_name,
    'folder_name': f'results_{current_time}',
    'nb_views': nb_views,
    'epochs': epochs,
    'patience': patience,
    'min_delta': min_delta,
    'data_dir': data_dir,
    'category': category,
    'log_suffix': log_suffix,
    'epoch_earlystop': None,
    'views_config': views_config,
    'opti_mvtn' : opti_mvtn,
    'lr_opti': lr_opti,
    'lr_mvtn_optimizer': lr_mvtn_optimizer,
    'weight_decay': weight_decay,
    'batch_size': bs,
    'pc_rendering': pc_rendering,
    'canonical_dist': canonical_dist,
    'simplified_mesh': simplified_mesh,
    'train_losses': [],
    'train_accuracies': [],
    'test_losses': [],
    'test_accuracies': [],
    'best_accuracy': best_accuracy,
    'best_epoch': best_epoch
}

# Create a text file with training setup details
with open(os.path.join(results_dir_current, 'training_setup.txt'), 'w') as f:
    f.write(f"Paths\n")

# Mesh qui ne cause pas de CUDA out of memory
df_mesh_ok = pd.DataFrame(columns=['Epoch', 'Batch', 'Name', 'Mesh_Vertices', 'Target', 'Points_Shape'])
df_mesh_ok.to_csv(os.path.join(results_dir_current, 'mesh_ok.csv'), index=False)
###############################################################################################
for epoch in range(epochs):
    print(f"\n ➰​ Epoch {epoch + 1}/{epochs}")
    torch.cuda.empty_cache()
    ############################################ Training
    mvnetwork.train(); mvtn.train(); mvrenderer.train()    
    running_loss = 0
    correct = 0.0
    
    train_pbar = tqdm.tqdm(total=len(train_loader), desc=f"Training")
    
    for i, (targets, meshes, points, names) in enumerate(train_loader):    
        try :
            azim, elev, dist = mvtn(points, c_batch_size=len(targets))
            rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
            outputs = mvnetwork(rendered_images)[0]

            loss = criterion(outputs, targets.cuda())
            running_loss += loss.item()
            loss.backward()
            correct += (torch.max(outputs, dim=1)[1] == targets.cuda()).to(torch.int32).sum().item()
            
            optimizer.step()
            optimizer.zero_grad()
            
            if mvtn_optimizer is not None:
                mvtn_optimizer.step()
                mvtn_optimizer.zero_grad()
                
            for n in range(len(names)):
                # Add data to the df_mesh_ok dataframe
                new_row = pd.DataFrame({
                    'Epoch': [f"{epoch + 1}"],
                    'Batch': [f"{i + 1}"],
                    'Name': [names[n]],
                    'Mesh_Vertices': [meshes[n].verts_packed().shape[0]],
                    'Target': [targets[n].item()],
                    'Points_Shape': [str(points[n].shape)]
                })
                df_mesh_ok = pd.concat([df_mesh_ok, new_row], ignore_index=True)
                # Save the dataframe after each update
                df_mesh_ok.to_csv(os.path.join(results_dir_current, 'mesh_ok.csv'), index=False)
        
        except RuntimeError as e: # Catch RuntimeError
            if "CUDA out of memory" in str(e):
                print(f"\n🚨 CUDA OUT OF MEMORY ERROR DETECTED DURING TRAINING!")
                print(f"  - Epoch: {epoch + 1}, Batch: {i + 1}")
                # Save these details to a log file
                with open(os.path.join(results_dir_current, 'cuda_error_meshes_train.txt'), 'a') as f:
                    f.write(f"Epoch {epoch + 1}, Batch {i + 1}:\n")
                    for i in range(len(meshes)):
                        line = f"Name: {names[i]}, Mesh: {meshes[i].verts_packed().shape[0]}, Target: {targets[i]}, Points shape: {points[i].shape}"
                        f.write(line + "\n")
                raise e
            else: raise e
            
        # Update progress bar with current loss and accuracy
        current_loss = running_loss / (i + 1)
        current_acc = 100.0 * correct / ((i + 1) * train_loader.batch_size)
        train_pbar.set_postfix({'loss': f'{current_loss:.5f}', 'acc': f'{current_acc:.2f}%'})
        train_pbar.update(1)

    train_pbar.close()
    avg_train_loss = running_loss / len(train_loader)
    avg_train_accuracy = 100.0 * correct / len(dset_train)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    print(f"🏋️​ Training - Final Loss: {avg_train_loss:.5f}, Final Accuracy: {avg_train_accuracy:.2f}%")
    
    ############################################ Testing
    mvnetwork.eval(); mvtn.eval(); mvrenderer.eval()
    running_loss = 0
    correct_test = 0.0
    
    test_pbar = tqdm.tqdm(total=len(test_loader), desc=f"Testing")
    
    for i, (targets, meshes, points, names) in enumerate(test_loader):
        try :
            with torch.no_grad():
                azim, elev, dist = mvtn(points, c_batch_size=len(targets))
                rendered_images, _ = mvrenderer(meshes, points, azim=azim, elev=elev, dist=dist)
                outputs = mvnetwork(rendered_images)[0]

                loss = criterion(outputs, targets.cuda())
                running_loss += loss.item()
                correct_test += (torch.max(outputs, dim=1)[1] == targets.cuda()).to(torch.int32).sum().item()
                
                # Update progress bar with current loss and accuracy
                current_loss = running_loss / (i + 1)
                current_acc = 100.0 * correct_test / ((i + 1) * test_loader.batch_size)
                test_pbar.set_postfix({'loss': f'{current_loss:.5f}', 'acc': f'{current_acc:.2f}%'})
                test_pbar.update(1)
                
        except RuntimeError as e: # Catch RuntimeError
            if "CUDA out of memory" in str(e):
                print(f"\n🚨 CUDA OUT OF MEMORY ERROR DETECTED DURING TESTING!")
                print(f"  - Epoch: {epoch + 1}, Batch: {i + 1}")
                with open(os.path.join(results_dir_current, 'cuda_error_meshes_test.txt'), 'a') as f:
                    f.write(f"Epoch {epoch + 1}, Batch {i + 1}:\n")
                    for i in range(len(meshes)):
                        line = f"Name: {names[i]}, Mesh: {meshes[i].verts_packed().shape[0]}, Target: {targets[i]}, Points shape: {points[i].shape}"
                        f.write(line + "\n")
                raise e
            else: raise e

    test_pbar.close()
    avg_test_loss = running_loss / len(test_loader)
    avg_test_accuracy = 100.0 * correct_test / len(dset_test)
    test_losses.append(avg_test_loss)
    test_accuracies.append(avg_test_accuracy)
    # Early stopping check
            
    print(f"🔍​ Testing - Final Loss: {avg_test_loss:.5f}, Final Accuracy: {avg_test_accuracy:.2f}%")
    
    # Save best models if we achieve better accuracy
    if avg_test_accuracy > best_accuracy:
        best_accuracy = avg_test_accuracy
        best_epoch = epoch + 1
        
        # Save MVTN weights
        torch.save(mvtn.state_dict(), os.path.join(results_dir_current, 'best', 'mvtn_best.pth'))
        
        # Save MVNetwork weights
        torch.save(mvnetwork.state_dict(), os.path.join(results_dir_current, 'best', 'mvnetwork_best.pth'))
        
        # Save MVRenderer weights
        torch.save(mvrenderer.state_dict(), os.path.join(results_dir_current, 'best', 'mvrenderer_best.pth'))
        
        print(f"\n🔝​ New best accuracy: {best_accuracy:.2f}% at epoch {best_epoch}")
        print(f"Saved best model weights in {os.path.join(results_dir_current, 'best')}")
    
    print("-" * 80)  # Add a separator line between epochs
    
    # Update training info dictionary
    training_info['train_losses'] = train_losses
    training_info['train_accuracies'] = train_accuracies
    training_info['test_losses'] = test_losses
    training_info['test_accuracies'] = test_accuracies
    training_info['best_accuracy'] = best_accuracy
    training_info['best_epoch'] = best_epoch
    

    ############################################################ Save training info after each epoch
    with open(os.path.join(results_dir_current, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=4)
        
    # Plot and save training/testing curves
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, color='#2E86C1', linestyle='-', label='Training Loss')
    plt.plot(range(1, epoch + 2), test_losses, color='#E74C3C', linestyle='-', label='Testing Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Testing Losses'); plt.legend(); plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, color='#2E86C1', linestyle='-', label='Training Accuracy')
    plt.plot(range(1, epoch + 2), test_accuracies, color='#E74C3C', linestyle='-', label='Testing Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Training and Testing Accuracies'); plt.legend(); plt.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir_current, 'training_curves.png'))
    plt.close()
    
    # Move logs file
    shutil.copy("logs.out", os.path.join(results_dir_current, 'logs-'+log_suffix+'.out'))
    shutil.copy("logs.err", os.path.join(results_dir_current, 'logs-'+log_suffix+'.err'))
    
    # Early stopping check
    early_stopping(avg_test_loss)
    if early_stopping.early_stop:
        print("🚨 Early stopping triggered.")
        training_info['epoch_earlystop'] = epoch + 1
        # si early stop : update 
        with open(os.path.join(results_dir_current, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=4)
        break


# update model_configs.csv
model_config = pd.read_csv('model_config.csv')
# Define new row data as a dictionary
new_row = {
    'Name': 'init-git',
    'Nb_views': nb_views,
    'Weights_path': f'results_{current_time}',
    'dir_modelnet40': data_dir,
    'config': views_config
}
model_config = model_config.append(new_row, ignore_index=True)
#model_config.to_csv('/home/mpelissi/MVTN/my_MVTN/results/train/model_config.csv', index=False)
    
print(f"\n🎉​ Training completed! \n​📂​ Results saved in {results_dir_current}")

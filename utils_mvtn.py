import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import os

def save_loss_acc(path_to_save, train_losses, test_losses, train_accuracies, test_accuracies, plot_best = False, best_epoch = None):
    
    mess_attention = "Dans json epoch commence à 1, \ndonc ici best_epoch = best_epoch_json - 1"
    # Plotting the loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot training and testing loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.title('Loss Over Epochs\n'+mess_attention)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Best epoch
    if plot_best:
        plt.scatter(x=best_epoch-1, y=test_losses[best_epoch-1], color='red', linestyle='--', label='Best Epoch')
        plt.scatter(x=best_epoch-1, y=train_losses[best_epoch-1], color='red', linestyle='--')
        plt.text(best_epoch-1, test_losses[best_epoch-1], f'({best_epoch-1}, {test_losses[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.text(best_epoch-1, train_losses[best_epoch-1], f'({best_epoch-1}, {train_losses[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.legend()

    # Plot training and testing accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='orange')
    plt.title('Accuracy Over Epochs\n'+mess_attention)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%) ')
    plt.legend()
    
        # Best epoch
    if plot_best:
        plt.scatter(x=best_epoch-1, y=test_accuracies[best_epoch-1], color='red', linestyle='--', label='Best Epoch')
        plt.scatter(x=best_epoch-1, y=train_accuracies[best_epoch-1], color='red', linestyle='--')
        plt.text(best_epoch-1, test_accuracies[best_epoch-1], f'({best_epoch-1}, {test_accuracies[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.text(best_epoch-1, train_accuracies[best_epoch-1], f'({best_epoch-1}, {train_accuracies[best_epoch-1]:.2f})', fontsize=9, color='red')
        plt.legend()

    plt.tight_layout()
    plt.savefig(path_to_save+'/training_metrics.png')  # Save the figure
    plt.show()  # Display the figure
    
def write_obj_with_color(vertices, faces, indices, obj_filename):
    # Open the OBJ file for writing
    with open(obj_filename, "w") as obj_file:     
        # Write each vertex to the OBJ file
        for i, v in enumerate(vertices):
            # Use the red material for vertices in the indices list
            if i in indices:
                obj_file.write(f"v {v[0]} {v[1]} {v[2]} 255 0 0\n")
            else:
                obj_file.write(f"v {v[0]} {v[1]} {v[2]} 128 128 128\n")

        obj_file.write(f"v {0} {0} {0} 0 255 255\n")   
        for face in faces:
            # OBJ format uses 1-based indexing, so we add 1 to the vertex indices
            obj_file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
            
    add_black_cube(obj_filename)

def add_black_cube(obj_filename, size=0.05, offset=[0.0, 0.0, 0.0]):
    """
    Add a black cube to an existing OBJ file
    
    Parameters:
    - obj_filename: path to the OBJ file
    - size: size of the cube
    - offset: position offset for the cube [x, y, z]
    """
    # Define cube vertices
    cube_vertices = [
        [-size + offset[0], -size + offset[1], -size + offset[2], 0, 0, 0],  # Black color
        [size + offset[0], -size + offset[1], -size + offset[2], 0, 0, 0],
        [size + offset[0], size + offset[1], -size + offset[2], 0, 0, 0],
        [-size + offset[0], size + offset[1], -size + offset[2], 0, 0, 0],
        [-size + offset[0], -size + offset[1], size + offset[2], 0, 0, 0],
        [size + offset[0], -size + offset[1], size + offset[2], 0, 0, 0],
        [size + offset[0], size + offset[1], size + offset[2], 0, 0, 0],
        [-size + offset[0], size + offset[1], size + offset[2], 0, 0, 0]
    ]
    
    # Read the original file to count vertices
    with open(obj_filename, 'r') as f:
        lines = f.readlines()
    
    # Count vertices in original file
    vertex_count = sum(1 for line in lines if line.startswith('v '))
    
    # Define cube faces (with adjusted indices)
    cube_faces = [
        [vertex_count + 0, vertex_count + 1, vertex_count + 2],
        [vertex_count + 0, vertex_count + 2, vertex_count + 3],
        [vertex_count + 4, vertex_count + 5, vertex_count + 6],
        [vertex_count + 4, vertex_count + 6, vertex_count + 7],
        [vertex_count + 0, vertex_count + 4, vertex_count + 7],
        [vertex_count + 0, vertex_count + 7, vertex_count + 3],
        [vertex_count + 1, vertex_count + 5, vertex_count + 6],
        [vertex_count + 1, vertex_count + 6, vertex_count + 2],
        [vertex_count + 0, vertex_count + 1, vertex_count + 5],
        [vertex_count + 0, vertex_count + 5, vertex_count + 4],
        [vertex_count + 3, vertex_count + 2, vertex_count + 6],
        [vertex_count + 3, vertex_count + 6, vertex_count + 7]
    ]
    
    # Append cube to the file
    with open(obj_filename, 'a') as f:
        for v in cube_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]} {v[3]} {v[4]} {v[5]}\n")
        for face in cube_faces:
            # OBJ format uses 1-based indexing
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
                      
           
def save_colored_obj_with_faces(filename, vertices, coeffs, faces, vmin=None, vmax=None):
    """
    Save a colored mesh to an OBJ file:
    - vertices: (N, 3)
    - coeffs: (N,) with scalar values (NaN for magenta)
    - faces: (F, 3) or (F, 4), indices into `vertices` (0-based)
    """
    assert vertices.shape[0] == coeffs.shape[0], "Mismatch between vertices and coeffs"
    assert faces.ndim == 2 and (faces.shape[1] == 3 or faces.shape[1] == 4), "Faces must be triangles or quads"

    # Normalize coeffs for colormap
    coeffs_valid = coeffs[~np.isnan(coeffs)]
    if vmin is None:
        vmin = np.min(coeffs_valid)
    if vmax is None:
        vmax = np.max(coeffs_valid)

    norm_coeffs = (coeffs - vmin) / (vmax - vmin)
    norm_coeffs = np.clip(norm_coeffs, 0, 1)

    # Apply rainbow colormap
    cmap = cm.get_cmap('rainbow')
    colors = cmap(norm_coeffs)[:, :3]

    # Set NaN colors to magenta
    nan_mask = np.isnan(coeffs)
    colors[nan_mask] = np.array([1.0, 0.0, 1.0])

    # Write .obj file
    with open(filename, 'w') as f:
        # Vertices with RGB
        for v, c in zip(vertices, colors):
            f.write(f"v {v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        
        # Faces (1-based indexing)
        for face in faces:
            indices = face + 1  # OBJ format is 1-indexed
            f.write("f " + " ".join(map(str, indices)) + "\n")

    #print(f"Saved colored mesh to: {filename}")
    
def calculer_aires_triangles_batch(pt_cloud, arr_faces):
    v0s = pt_cloud[arr_faces[:, 0]]
    v1s = pt_cloud[arr_faces[:, 1]]
    v2s = pt_cloud[arr_faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1s - v0s, v2s - v0s), axis=1)
# tactile_preprocessing.py

import os
import numpy as np
import torch

def load_tactile_file(file_path):
    """
    Load tactile data from a text file.
    """
    # This assumes tactile data is stored in plain text with one measurement per line
    data = np.loadtxt(file_path)
    return data

def normalize_data(data):
    """
    Normalize the tactile data to have zero mean and unit variance.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

def preprocess_tactile(file_path):
    """
    Preprocess tactile file to create a normalized data tensor.
    """
    # Load the data
    data = load_tactile_file(file_path)
    
    # Normalize the data
    normalized_data = normalize_data(data)
    
    # Convert the normalized data to a PyTorch tensor
    data_tensor = torch.from_numpy(normalized_data).float()
    
    # Add a batch dimension (B x C x L) if you're using 1D convolutional network
    data_tensor = data_tensor.unsqueeze(0)
    
    return data_tensor

def preprocess_tactile_directory(tactile_dir):
    """
    Process all tactile data files in the specified directory.
    """
    tactile_tensors = []
    file_names = []

    for file_name in os.listdir(tactile_dir):
        if file_name.endswith('.txt'):  # Assuming .txt format
            print("Pre-processing - ", file_name)
            file_path = os.path.join(tactile_dir, file_name)
            tactile_tensor = preprocess_tactile(file_path)
            tactile_tensors.append(tactile_tensor)
            file_names.append(file_name)

    # Stack all tactile tensors
    if tactile_tensors:
        tactile_data = torch.stack(tactile_tensors, dim=0)
    else:
        tactile_data = torch.empty(0)

    return tactile_data, file_names

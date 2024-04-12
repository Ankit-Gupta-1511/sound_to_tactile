import numpy as np
import torch
import os
from itertools import product

def preprocess_tactile_data(directory):
    """
    Preprocess all tactile data files in the specified directory.
    
    Args:
    - directory (str): Directory containing the tactile data files.
    
    Returns:
    - all_data (dict): Dictionary with keys as 'prefix_trial' and values as preprocessed tensor data.
    - file_names (list): List of filenames in the order they were processed.
    """
    # This assumes file naming convention is consistent as shown in your directory structure
    prefixes = sorted(set('_'.join(f.split('_')[:-2]) for f in os.listdir(directory) if f.endswith('.txt')))
    trials = sorted(set(f.split('_')[-1] for f in os.listdir(directory) if f.endswith('.txt')))
    axes = ['X', 'Y', 'Z']

    all_data = []
    file_names = []  # List to keep track of file names

    for prefix, trial in product(prefixes, trials):
        # Initialize a list to hold data from all three axes for the current prefix and trial
        trial_data = []
        for axis in axes:
            filename = f"{prefix}_{axis}_{trial}"
            print("Pre-processing - ", filename)
            file_path = os.path.join(directory, filename)
            axis_data = np.loadtxt(file_path)
            normalized_axis_data = (axis_data - np.mean(axis_data)) / np.std(axis_data)
            trial_data.append(normalized_axis_data)
            file_names.append(filename)  # Append filename to the list
        
        # Stack X, Y, Z axis data along the first dimension to create a single tensor
        tactile_tensors = torch.tensor(trial_data, dtype=torch.float32)
        
        # The key for the dictionary is the combination of prefix and trial
        key = f"{prefix}_{trial}"
        all_data.append(tactile_tensors.unsqueeze(0))   # Add batch dimension

    # Stack all tactile tensors
    tactile_data = torch.stack(all_data, dim=0)
    

    return tactile_data, file_names


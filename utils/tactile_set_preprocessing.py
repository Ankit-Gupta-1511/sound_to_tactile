import numpy as np
import torch
import os
from itertools import product
import librosa


def mel_spectrogram(data, sr=10000, n_fft=2048, hop_length=512, n_mels=128):
    """ Placeholder for tactile data conversion to Mel-spectrogram """
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)




def linear_prediction_coefficients(data, order=2):
    """ Placeholder function for calculating LPCs of a signal """
    
    a = librosa.lpc(data, order=order)
    return a

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    
    # Check if standard deviation is zero
    if std == 0:
        print("Standard deviation is zero. Returning zeroed data to avoid division by zero.")
        return np.zeros(data.shape)
    else:
        return (data - mean) / std

def preprocess(file_path):
    """ Process tactile file to create feature vectors """
    data = np.loadtxt(file_path)
    sr = 10000  # Sample rate for tactile data might need to be defined based on data acquisition
    mel_spec = mel_spectrogram(data, sr=sr)
    lpc_coeffs = linear_prediction_coefficients(data)

    # Normalize LPC coefficients
    normalized_lpc =normalize_data(lpc_coeffs)
    # feature_tensor = torch.tensor(normalized_lpc, dtype=torch.float32)

    # return feature_tensor.unsqueeze(0)  # Adding batch dimension
    return normalized_lpc

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
            normalized_axis_data = preprocess(file_path)
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


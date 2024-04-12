import os
import numpy as np
import librosa  # Used for audio, but placeholder for tactile signal processing
import torch

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

def preprocess_tactile_data(file_path):
    """ Process tactile file to create feature vectors """
    data = np.loadtxt(file_path)
    sr = 10000  # Sample rate for tactile data might need to be defined based on data acquisition
    mel_spec = mel_spectrogram(data, sr=sr)
    lpc_coeffs = linear_prediction_coefficients(data)

    # Normalize LPC coefficients
    normalized_lpc =normalize_data(lpc_coeffs)
    feature_tensor = torch.tensor(normalized_lpc, dtype=torch.float32)

    return feature_tensor.unsqueeze(0)  # Adding batch dimension

def preprocess_tactile_directory(directory):
    """
    Process all tactile data files in the specified directory.
    
    Args:
    - directory (str): Directory containing the tactile data files.
    
    Returns:
    - all_data (torch.Tensor): Tensor containing all the preprocessed tactile data.
    - file_names (list): List of filenames in the order they were processed.
    """
    file_names = []  # List to keep track of file names
    all_features = []

    for file_name in os.listdir(directory):
        if file_name.endswith('.txt'):
            print("Preprocessing - ", file_name)
            file_path = os.path.join(directory, file_name)
            features = preprocess_tactile_data(file_path)
            all_features.append(features)
            file_names.append(file_name)

    # Stack all features along the first dimension
    tactile_data = torch.cat(all_features, dim=0)

    return tactile_data, file_names


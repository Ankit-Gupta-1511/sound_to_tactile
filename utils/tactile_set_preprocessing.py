import numpy as np
import torch
import os
from itertools import product
import librosa

def mel_spectrogram(data, sr=10000, n_fft=1024, hop_length=None, n_mels=256):
    """ Convert data into Mel-spectrogram """
    if hop_length is None:
        hop_length = (data.shape[0] - n_fft) // (n_mels - 1)
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max)

def normalize_data(data):
    """ Normalize data """
    return librosa.util.normalize(data)

def preprocess(file_path, duration=3, sr=10000, n_mels=256):
    """ Process tactile file to create feature vectors """
    # Load the data
    data = np.loadtxt(file_path)
    # Resample or truncate to 3 seconds
    num_samples = sr * duration
    if data.shape[0] > num_samples:
        data = data[:num_samples]
    elif data.shape[0] < num_samples:
        data = np.pad(data, (0, num_samples - data.shape[0]), mode='constant')

    # Create Mel-spectrogram
    mel_spec = mel_spectrogram(data, sr=sr, n_mels=n_mels)

    # Normalize
    mel_spec_norm = normalize_data(mel_spec)

    # Resize to 256x256 if necessary
    if mel_spec_norm.shape[1] < 256:
        # If fewer than 256 time-steps, pad with zeros
        padding_amount = 256 - mel_spec_norm.shape[1]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, padding_amount)), mode='constant')

    return mel_spec_norm

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
        trial_data = []
        for axis in axes:
            filename = f"{prefix}_{axis}_{trial}"
            print("Pre-processing - ", filename)
            file_path = os.path.join(directory, filename)
            # Here you might concatenate your x, y, z data if they are separate
            # For now, we will process them individually
            mel_spec = preprocess(file_path)
            trial_data.append(mel_spec)

        # Convert list of Mel-spectrograms into a single tensor per trial
        trial_tensor = torch.tensor(np.stack(trial_data), dtype=torch.float32)
        all_data.append(trial_tensor.unsqueeze(0))  # Add batch dimension

    # Stack all trial tensors
    tactile_data = torch.cat(all_data, dim=0)

    return tactile_data, file_names


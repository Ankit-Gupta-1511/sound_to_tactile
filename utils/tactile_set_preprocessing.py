import numpy as np
import torch
import os
from itertools import product
import librosa
import matplotlib.pyplot as plt

save_dir = 'output/mel_frequency_spectrogram/tactile'

def mel_spectrogram(data, sr=10000, n_fft=1024, hop_length=None, n_mels=256):
    """ Convert data into Mel-spectrogram """
    if hop_length is None:
        # Calculate the number of samples for the 4s duration of the file
        num_samples = sr * 4

        # Calculate hop_length to get 256 time steps (one less than n_mels because the first frame is centered at time zero)
        hop_length = max(1, num_samples // (n_mels - 1))
    S = librosa.feature.melspectrogram(y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(S, ref=np.max)
    
    # Ensure mel_spec_db has the shape (n_mels, 256) by trimming or padding
    mel_spec_db = mel_spec_db[:, :256] if mel_spec_db.shape[1] > 256 else np.pad(mel_spec_db, [(0, 0), (0, max(0, 256 - mel_spec_db.shape[1]))], mode='constant')

    return mel_spec_db


def mel_spectrogram_to_tactile(mel_spec_db, sr=10000, n_fft=1024, hop_length=None, n_mels=256):
    """ Convert Mel spectrogram back to waveform """
    if hop_length is None:
        # Calculate the number of samples for the 4s duration of the file
        num_samples = sr * 4
        # Calculate hop_length to get 256 time steps
        hop_length = max(1, num_samples // (n_mels - 1))

    # Check if the tensor is on CUDA and move it to CPU
    if mel_spec_db.is_cuda:
        mel_spec_db = mel_spec_db.cpu()

    # Convert the tensor to numpy array after ensuring it's on the CPU
    mel_spec_db_np = mel_spec_db.numpy()

    # Convert decibel to power spectrogram
    mel_spec = librosa.db_to_power(mel_spec_db_np)
    # Inverse Mel spectrogram
    y_inv = librosa.feature.inverse.mel_to_audio(mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32)
    return y_inv


def normalize_data(data):
    """ Normalize data """
    return librosa.util.normalize(data)

def preprocess(file_path, duration=4, sr=10000, n_mels=256):
    """ Process tactile file to create feature vectors """
    # Load the data
    data = np.loadtxt(file_path)
    # Resample or truncate to 4 seconds
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
    if mel_spec_norm.shape[0] < 256:
        padding_amount = 256 - mel_spec_norm.shape[0]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, padding_amount), (0, 0)), mode='constant')
    if mel_spec_norm.shape[1] < 256:
        padding_amount = 256 - mel_spec_norm.shape[1]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, padding_amount)), mode='constant')

    # Calculate the number of samples for the 4s duration of the file
    num_samples = sr * 4

    # Calculate hop_length to get 256 time steps (one less than n_mels because the first frame is centered at time zero)
    hop_length = max(1, num_samples // (n_mels - 1))

    if save_dir is not None:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_norm, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.splitext(os.path.basename(file_path))[0] + '_spectrogram.png'
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path)
        plt.close()

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


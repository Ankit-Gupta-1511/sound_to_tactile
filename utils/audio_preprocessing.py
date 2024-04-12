# audio_preprocessing.py

import os
import librosa
import torch
import numpy as np

def load_audio_file(file_path, sample_rate=22050):
    """
    Load an audio file.
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def audio_to_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    Convert audio to a mel-spectrogram.
    """
    # Ensure all parameters are passed as keyword arguments
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(S=spectrogram, ref=np.max)
    return spectrogram_db

def normalize_data(data):
    """
    Normalize the spectrogram data.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def preprocess_audio(file_path):
    """
    Preprocess audio file to create a normalized mel-spectrogram tensor.
    """
    audio, sr = load_audio_file(file_path)
    spectrogram_db = audio_to_spectrogram(audio, sr)
    normalized_spectrogram = normalize_data(spectrogram_db)
    spectrogram_tensor = torch.tensor(normalized_spectrogram).float()
    spectrogram_tensor = spectrogram_tensor.unsqueeze(0)  # Add batch dimension
    return spectrogram_tensor

def preprocess_directory(audio_dir, sample_rate=22050):
    """
    Process all audio files in the specified directory.
    """
    audio_tensors = []
    file_names = []

    # Process each file in directory
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):  # Assuming .wav format
            print("Pre-processing - ", file_name)
            file_path = os.path.join(audio_dir, file_name)
            spectrogram_tensor = preprocess_audio(file_path)
            audio_tensors.append(spectrogram_tensor)
            file_names.append(file_name)

    # Stack all spectrogram tensors along the first dimension
    audio_data = torch.cat(audio_tensors, dim=0)
    return audio_data, file_names


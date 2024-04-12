import os
import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt

save_dir_path = 'output/mel_frequency_spectrogram'

def load_audio_file(file_path, sample_rate=44100):
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=4)  # Make sure to load only 4s of audio
    return audio, sr

def audio_to_spectrogram(audio, sr, n_fft=2048, hop_length=512, n_mels=256):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def preprocess_audio(file_path, save_dir=None):
    audio, sr = load_audio_file(file_path)
    
    # Calculate the total number of samples in a 4s clip at the given sample rate
    total_samples = 4 * sr
    # Calculate the hop_length to get 256 frames
    hop_length = total_samples // (256 - 1)

    spectrogram_db = audio_to_spectrogram(audio, sr, hop_length=hop_length)
    normalized_spectrogram = normalize_data(spectrogram_db)

    # Ensure spectrogram is 256x256
    if normalized_spectrogram.shape[1] > 256:
        # If more than 256 time-steps, truncate excess
        normalized_spectrogram = normalized_spectrogram[:, :256]
    elif normalized_spectrogram.shape[1] < 256:
        # If fewer than 256 time-steps, pad with zeros
        padding_amount = 256 - normalized_spectrogram.shape[1]
        normalized_spectrogram = np.pad(normalized_spectrogram, ((0, 0), (0, padding_amount)), mode='constant')

    spectrogram_tensor = torch.tensor(normalized_spectrogram).float().unsqueeze(0)  # Add batch dimension

    if save_dir is not None:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(normalized_spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plot_filename = os.path.splitext(os.path.basename(file_path))[0] + '_spectrogram.png'
        save_path = os.path.join(save_dir, plot_filename)
        plt.savefig(save_path)
        plt.close()

    return spectrogram_tensor

# Updated to process 4s audio with sample_rate of 44100
def preprocess_directory(audio_dir, sample_rate=44100):
    audio_tensors = []
    file_names = []

    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            print("Pre-processing - ", file_name)
            file_path = os.path.join(audio_dir, file_name)
            spectrogram_tensor = preprocess_audio(file_path, save_dir=None)
            audio_tensors.append(spectrogram_tensor)
            file_names.append(file_name)

    audio_data = torch.cat(audio_tensors, dim=0)
    return audio_data, file_names

import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch.nn as nn

from utils.audio_preprocessing import preprocess_directory
from utils.tactile_preprocessing import preprocess_tactile_directory
from utils.tactile_set_preprocessing import preprocess_tactile_data
from residual_unet import ResidualUNet

model_path = 'output/model/model_weights.pth'

# Paths for data
audio_tensor_path = 'output/preprocessing/audio_data_test.pt'
audio_file_names_path = 'output/preprocessing/file_names_test.pkl'
audio_dir = 'data/SoundScans/Movement/Testing'

tactile_tensor_path = 'output/preprocessing/tactile_data_test.pt'
tactile_file_names_path = 'output/preprocessing/tactile_file_names_test.pkl'
tactile_dir  = 'data/AccelScansComponents/Movement/Testing'

def spectral_similarity(x, y):
    """Compute the spectral similarity between two sets of signals."""
    # Assuming x and y are of shape [batch, channels, height, width]
    # and we need to convert them to complex numbers before FFT
    
    # Expand dimensions to add an imaginary part
    zero_imag = torch.zeros_like(x)
    
    # Stack along the last dimension to form [batch, channels, height, width, 2]
    x_complex = torch.stack((x, zero_imag), dim=-1)
    y_complex = torch.stack((y, zero_imag), dim=-1)
    
    # Compute the FFT
    X = torch.fft.fftn(x_complex, dim=(2, 3))
    Y = torch.fft.fftn(y_complex, dim=(2, 3))
    
    # Compute magnitude
    magX = torch.sqrt(X[..., 0]**2 + X[..., 1]**2)
    magY = torch.sqrt(Y[..., 0]**2 + Y[..., 1]**2)
    
    # Normalize magnitudes to prevent large dynamic range issues
    magX = (magX - magX.mean(dim=(2,3), keepdim=True)) / (magX.std(dim=(2,3), keepdim=True) + 1e-8)
    magY = (magY - magY.mean(dim=(2,3), keepdim=True)) / (magY.std(dim=(2,3), keepdim=True) + 1e-8)
    
    # Calculate similarity
    num = (magX * magY).sum(dim=(2, 3))  # sum over spatial dimensions
    denom = torch.sqrt((magX**2).sum(dim=(2, 3)) * (magY**2).sum(dim=(2, 3)))
    similarity = num / (denom + 1e-8)
    
    return similarity.mean()

def temporal_similarity(x, y):
    """Compute Mean Subtracted Contrast Normalized (MSCN) coefficients and similarity."""
    def mscn(image):
        mu = F.avg_pool2d(image, 3, stride=1, padding=1)
        sigma = torch.sqrt(F.avg_pool2d((image - mu)**2, 3, stride=1, padding=1) + 1e-8)
        return (image - mu) / (sigma + 1)
    
    x_mscn = mscn(x)
    y_mscn = mscn(y)
    
    similarity = (x_mscn * y_mscn).sum(dim=(-2, -1)) / torch.sqrt((x_mscn**2).sum(dim=(-2, -1)) * (y_mscn**2).sum(dim=(-2, -1)))
    return similarity.mean()

def st_sim(original_signals, predicted_signals, eta=0.5):
    """Compute ST-SIM combining spectral and temporal measures."""
    spectral_sim = spectral_similarity(original_signals, predicted_signals)
    temporal_sim = temporal_similarity(original_signals, predicted_signals)
    st_sim_score = (spectral_sim**eta) * (temporal_sim**(1 - eta))
    return st_sim_score

def load_preprocessed_data(audio_tensor_path, file_names_path):
    audio_data = torch.load(audio_tensor_path)
    with open(file_names_path, 'rb') as f:
        file_names = pickle.load(f)
    return audio_data, file_names

def save_preprocessed_data(audio_data, file_names, audio_tensor_path, file_names_path):
    torch.save(audio_data, audio_tensor_path)
    with open(file_names_path, 'wb') as f:
        pickle.dump(file_names, f)

'''
Preprocess audio data
'''
# Check if the preprocessed files exist
if os.path.exists(audio_tensor_path) and os.path.exists(audio_file_names_path):
    # Load the preprocessed data
    audio_data, file_names = load_preprocessed_data(audio_tensor_path, audio_file_names_path)
    print("Loaded preprocessed audio data.")
else:
    # Preprocess the data since it doesn't exist
    audio_data, file_names = preprocess_directory(audio_dir)
    # Save the preprocessed data for future use
    save_preprocessed_data(audio_data, file_names, audio_tensor_path, audio_file_names_path)
    print("Preprocessed and saved new audio data.")


'''
Preprocess tactile data
'''
# Check if the preprocessed files exist
if os.path.exists(tactile_tensor_path) and os.path.exists(tactile_file_names_path):
    # Load the preprocessed data
    tactile_data, tactile_file_names = load_preprocessed_data(tactile_tensor_path, tactile_file_names_path)
    print("Loaded preprocessed tactile data.")
else:
    # Preprocess the data since it doesn't exist
    # tactile_data, tactile_file_names = preprocess_tactile_directory(tactile_dir)
    tactile_data, tactile_file_names = preprocess_tactile_data(tactile_dir)
    # Save the preprocessed data for future use
    save_preprocessed_data(tactile_data, tactile_file_names, tactile_tensor_path, tactile_file_names_path)
    print("Preprocessed and saved new tactile data.")


# Create a TensorDataset and DataLoader
test_dataset = TensorDataset(audio_data, tactile_data)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = ResidualUNet()
model.load_state_dict(torch.load(model_path))
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    for original_audio_signals, original_tactile_signal in test_dataloader:

        original_audio_signals = original_audio_signals.to(device).unsqueeze(1)
        original_tactile_signal = original_tactile_signal.to(device)
        predicted_signals = model(original_audio_signals)
        # print(original_tactile_signal.shape, predicted_signals.shape)
        score = st_sim(original_tactile_signal, predicted_signals)
        print(f"ST-SIM Score: {score.item()}")

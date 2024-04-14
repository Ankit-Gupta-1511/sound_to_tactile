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

def apply_psychometric_function(magnitude, threshold, tau, theta):
    """ Apply the psychometric function to convert magnitudes to detection probabilities.
        magnitude: Magnitude of the FFT
        threshold: Detection threshold (dT)
        tau: Modulation threshold
        theta: Standard deviation for the psychometric function
    """
    # Convert magnitude to acceleration units if needed
    magnitude = 9.81 * magnitude  # Convert to m/s^2 if your magnitude is not already in these units
    # Make sure constants are tensors and on the same device and type as magnitude
    two = torch.tensor(2.0, dtype=magnitude.dtype, device=magnitude.device)
    # Compute probability density using the cumulative distribution function of a normal distribution
    probability_density = 0.5 * (1 + torch.erf((magnitude - tau * threshold) / (theta * torch.sqrt(two))))
    return probability_density

def spectral_similarity(x, y, threshold=1.0, tau=1.0, theta=0.1):
    """ Compute the spectral similarity between two sets of signals using ST-SIM measure. """
    # Assuming x and y are of shape [batch, channels, height, width]
    # FFT to convert signals to frequency domain
    X = torch.fft.fftn(x, dim=(2, 3))
    Y = torch.fft.fftn(y, dim=(2, 3))
    
    # Compute magnitude from real and imaginary parts
    magX = torch.sqrt(X.real**2 + X.imag**2)
    magY = torch.sqrt(Y.real**2 + Y.imag**2)
    
    # Apply psychometric function to magnitudes
    pX = apply_psychometric_function(magX, threshold, tau, theta)
    pY = apply_psychometric_function(magY, threshold, tau, theta)
    
    # Calculate spectral similarity
    num = (pX * pY).sum(dim=(2, 3))  # sum over spatial dimensions
    denom = torch.sqrt((pX**2).sum(dim=(2, 3)) * (pY**2).sum(dim=(2, 3)))
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

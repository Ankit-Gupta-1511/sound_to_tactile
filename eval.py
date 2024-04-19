import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import torch.nn as nn

from utils.audio_preprocessing import preprocess_directory
from utils.tactile_preprocessing import preprocess_tactile_directory
from utils.tactile_set_preprocessing import preprocess_tactile_data, mel_spectrogram_to_tactile
from utils.plot_utils import plot_signals
from residual_unet import ResidualUNet

model_path = 'output/model/model_weights.pth'

# Paths for data
audio_tensor_path = 'output/preprocessing/audio_data_test.pt'
audio_file_names_path = 'output/preprocessing/file_names_test.pkl'
audio_dir = 'data/SoundScans/Movement/Testing'

tactile_tensor_path = 'output/preprocessing/tactile_data_test.pt'
tactile_file_names_path = 'output/preprocessing/tactile_file_names_test.pkl'
tactile_dir  = 'data/AccelScansComponents/Movement/Testing'

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
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Loading trained model
model = ResidualUNet()
model.load_state_dict(torch.load(model_path))
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function
criterion = nn.MSELoss()

# Initialize loss storage for averaging later
total_loss = 0.0
count = 0

# No gradient computation during testing

with torch.no_grad():
    for audio, tactile in test_dataloader:
        # Move tensors to the configured device
        audio = audio.to(device).unsqueeze(1)  # Add channel dimension if necessary
        tactile = tactile.to(device)

        # Forward pass
        predictions = model(audio)

        # Calculate loss
        loss = criterion(predictions, tactile)
        total_loss += loss.item()
        count += 1

        print(count)
        reconstructed_waveform = mel_spectrogram_to_tactile(predictions)
        tactile_waveform = mel_spectrogram_to_tactile(tactile)
        plot_signals(original=tactile_waveform, reconstructed=reconstructed_waveform, sr=10000, file_name=file_names[count - 1])

# Average loss
average_loss = total_loss / count
print(f'Final average loss on test set: {average_loss}')
import os
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils.audio_preprocessing import preprocess_directory
from utils.tactile_preprocessing import preprocess_tactile_directory
from utils.tactile_set_preprocessing import preprocess_tactile_data
from residual_unet import ResidualUNet

def load_preprocessed_data(audio_tensor_path, file_names_path):
    audio_data = torch.load(audio_tensor_path)
    with open(file_names_path, 'rb') as f:
        file_names = pickle.load(f)
    return audio_data, file_names

def save_preprocessed_data(audio_data, file_names, audio_tensor_path, file_names_path):
    torch.save(audio_data, audio_tensor_path)
    with open(file_names_path, 'wb') as f:
        pickle.dump(file_names, f)

# Paths for data
audio_tensor_path = 'output/preprocessing/audio_data.pt'
audio_file_names_path = 'output/preprocessing/file_names.pkl'
audio_dir = 'data/SoundScans/Movement/Training'

tactile_tensor_path = 'output/preprocessing/tactile_data.pt'
tactile_file_names_path = 'output/preprocessing/tactile_file_names.pkl'
tactile_dir  = 'data/AccelScansComponents/Movement/Training'

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


print(audio_data.shape, tactile_data.shape)

# Create datasets and dataloaders
dataset = TensorDataset(audio_data, tactile_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = ResidualUNet(in_channels=128)

# Define a loss function and optimizer
criterion = nn.MSELoss()  # Example loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Example optimizer

# Device configuration - using CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for i, (audio, tactile) in enumerate(dataloader):
        # Move tensors to the configured device
        audio = audio.to(device)
        tactile = tactile.to(device)
        
        # Forward pass
        predictions = model(audio)
        
        # Compute loss
        loss = criterion(predictions, tactile)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

# Save the model's state_dict
torch.save(model.state_dict(), 'output/model/model_weights.pth')
print('Model trained and saved successfully.')
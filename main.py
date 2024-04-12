import os
import torch
import pickle
from utils.audio_preprocessing import preprocess_directory

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
file_names_path = 'output/preprocessing/file_names.pkl'
audio_dir = 'data/SoundScans/Movement/Training'

# Check if the preprocessed files exist
if os.path.exists(audio_tensor_path) and os.path.exists(file_names_path):
    # Load the preprocessed data
    audio_data, file_names = load_preprocessed_data(audio_tensor_path, file_names_path)
    print("Loaded preprocessed data.")
else:
    # Preprocess the data since it doesn't exist
    audio_data, file_names = preprocess_directory(audio_dir)
    # Save the preprocessed data for future use
    save_preprocessed_data(audio_data, file_names, audio_tensor_path, file_names_path)
    print("Preprocessed and saved new data.")

# You can now use `audio_data` and `file_names` for further processing, training, etc.
print(audio_data)
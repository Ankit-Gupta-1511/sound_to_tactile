# main.py
import pickle
import torch
from utils.audio_preprocessing import preprocess_directory

# Path to the directory containing audio files
audio_dir = 'data/SoundScans/Movement/Training'

# Preprocess all audio files in the directory
audio_data, file_names = preprocess_directory(audio_dir)

# print(audio_data)

# Save the tensor data
torch.save(audio_data, 'output/preprocessing/audio_data.pt')
with open('output/preprocessing/file_names.pkl', 'wb') as f:
    pickle.dump(file_names, f)

# Now, `audio_data` contains all the spectrogram tensors ready for training,
# and `file_names` lists the names of the processed files.

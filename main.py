# main.py

from utils.audio_preprocessing import preprocess_directory

# Path to the directory containing audio files
audio_dir = 'data/SoundScans/Movement/Training'

# Preprocess all audio files in the directory
audio_data, file_names = preprocess_directory(audio_dir)

print(audio_data)

# Now, `audio_data` contains all the spectrogram tensors ready for training,
# and `file_names` lists the names of the processed files.

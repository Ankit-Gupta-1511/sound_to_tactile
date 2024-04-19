import matplotlib.pyplot as plt
import os
import librosa
import numpy as np

# Define the path where you want to save the plots
save_directory = 'output/signal_plots'
os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

# Plot and save the first example in the batch for channel 0

def plot_signals(original, reconstructed, sr, file_name):
    """ Plot original and reconstructed signals """
    plot_path = os.path.join(save_directory, file_name + '.png')
    
    # Ensure data is 1D by selecting the first element in the batch and the first channel
    original = original[0, 0] if original.ndim > 1 else original
    reconstructed = reconstructed[0, 0] if reconstructed.ndim > 1 else reconstructed
    
    # Calculate time axis for plotting
    time_axis_original = np.linspace(0, len(original) / sr, num=len(original))
    time_axis_reconstructed = np.linspace(0, len(reconstructed) / sr, num=len(reconstructed))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_axis_original, original, alpha=0.5, label='Original')
    plt.title('Original Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(time_axis_reconstructed, reconstructed, alpha=0.5, color='r', label='Reconstructed')
    plt.title('Reconstructed Signal')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()  # Ensure the plot is closed after saving
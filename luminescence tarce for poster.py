""" Luminescence trace for poster """
#%%
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Define the relative path of the file in your cwd
cwd = os.getcwd()
relative_path = r"data/Batch_B/Batch_B_2022_935_CFC_GPIO/Data_Miniscope_PP.mat"
file_path = os.path.join(cwd, relative_path)

def load_c_raw_all_to_dataframe(file_path):
    """
    Load 'C_raw_all' from the HDF5 .mat file and convert it to a Pandas DataFrame.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'Data' in f and 'C_Raw_all' in f['Data']:
                # Extract the C_raw_all data
                c_raw_all = f['Data']['C_Raw_all'][:]
                
                # Convert to DataFrame
                df = pd.DataFrame(c_raw_all.T)
                
                # Add labels (optional, can be customized)
                num_neurons, num_frames = df.shape
                df.columns = [f"Frame_{i+1}" for i in range(num_frames)]
                df.index = [f"Neuron_{i+1}" for i in range(num_neurons)]
                
                return df
            else:
                print("The dataset 'C_Raw_all' was not found in the file.")
                return None
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def print_shape(df):
    """
    Print the number of rows and columns of the DataFrame.
    """
    if df is not None:
        num_neurons, num_frames = df.shape
        print(f"Number of rows (neurons): {num_neurons}")
        print(f"Number of columns (frames/time): {num_frames}")
    else:
        print("DataFrame is empty or not loaded.")

def plot_neuron_signal(df, neuron_index):
    """
    Plot the signal of a given neuron over all frames.
    """
    if df is not None and 0 <= neuron_index < len(df):
        plt.figure(figsize=(10, 6))
        plt.plot(df.iloc[neuron_index], label=f"Neuron {neuron_index + 1}")
        plt.xlabel("Frame / Time")
        plt.ylabel("Signal Intensity")
        plt.title(f"Neuron {neuron_index + 1} Signal Over Time")
        plt.legend()
        plt.show()
    else:
        print("Invalid neuron index or DataFrame is not loaded.")

def plot_neuron_signal_with_range(df, neuron_index, start_frame, end_frame):
    """
    Plot the signal of a given neuron over a specified frame range.
    """
    if df is not None and 0 <= neuron_index < len(df):
        if 0 <= start_frame < end_frame <= df.shape[1]:
            plt.figure(figsize=(10, 6))
            plt.plot(df.iloc[neuron_index, start_frame:end_frame], label=f"Neuron {neuron_index + 1}")
            plt.xlabel("Frame / Time")
            plt.ylabel("Signal Intensity")
            plt.title(f"Neuron {neuron_index + 1} Signal from Frame {start_frame + 1} to {end_frame}")
            plt.legend()
            plt.show()
        else:
            print("Invalid frame range.")
    else:
        print("Invalid neuron index or DataFrame is not loaded.")

def plot_three_neurons(df, neuron_indices, start_frame, end_frame):
    """
    Plot signals of three neurons over a specified frame range.
    """
    if df is not None:
        if len(neuron_indices) == 3 and all(0 <= idx < len(df) for idx in neuron_indices):
            if 0 <= start_frame < end_frame <= df.shape[1]:
                plt.figure(figsize=(12, 8))
                for idx in neuron_indices:
                    plt.plot(df.iloc[idx, start_frame:end_frame], label=f"Neuron {idx + 1}")
                
                plt.xlabel("Frame / Time")
                plt.ylabel("Signal Intensity")
                plt.title(f"Signals of Neurons {neuron_indices[0] + 1}, {neuron_indices[1] + 1}, and {neuron_indices[2] + 1}")
                plt.legend()
                plt.show()
            else:
                print("Invalid frame range.")
        else:
            print("Invalid neuron indices or DataFrame is not loaded.")
    else:
        print("DataFrame is empty or not loaded.")

# 1. Load the data
df_c_raw_all = load_c_raw_all_to_dataframe(file_path)

# 2. Print number of rows and columns
print_shape(df_c_raw_all)

# 3. Plotting example: Plot a signal of a neuron (neuron index 0)
plot_neuron_signal(df_c_raw_all, neuron_index=0)

# 4. Plotting example with start and end frame: Plot signal of a neuron with frame range (neuron index 0, frames 100 to 200)
plot_neuron_signal_with_range(df_c_raw_all, neuron_index=0, start_frame=100, end_frame=200)

# 5. Plotting example for 3 neurons: Neuron indices [0, 1, 2], frame range 100 to 200
plot_three_neurons(df_c_raw_all, neuron_indices=[0, 1, 2], start_frame=100, end_frame=200)

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_action_potential_signal(length=500, peak_frame=200, peak_value=1.0, width=20):
    """
    Generate a synthetic action potential signal with a given peak position.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: The frame where the peak occurs.
    - peak_value: The value at the peak of the action potential.
    - width: The width of the action potential peak.
    
    Returns:
    - A NumPy array representing the action potential signal.
    """
    signal = np.zeros(length)
    start_frame = max(0, peak_frame - width // 2)
    end_frame = min(length, peak_frame + width // 2)
    
    # Create a simple Gaussian-like peak for the action potential
    signal[start_frame:end_frame] = np.exp(-((np.arange(start_frame, end_frame) - peak_frame) ** 2) / (2 * (width // 4) ** 2)) * peak_value
    return signal

def plot_similar_signals_with_shift(length=500, peak_frame=200, shift=20, width=20):
    """
    Plot two signals with similar action potentials, where the second signal is delayed.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: Frame at which the first signal peaks.
    - shift: The number of frames to shift the action potential in the second signal.
    - width: Width of the action potential.
    """
    # Generate two similar action potentials
    signal_1 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width)
    signal_2 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width)
    
    # Generate the third signal with the shifted peak
    signal_3 = generate_action_potential_signal(length=length, peak_frame=peak_frame + shift, peak_value=1.0, width=width)
    
    # Plot the signals
    plt.figure(figsize=(12, 8))

    # Plot signal 1
    plt.subplot(3, 1, 1)
    plt.plot(signal_1, label='Neuron 1 (Original Signal)', color='blue')
    plt.title('Neuron 1 Signal')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot signal 2 directly below
    plt.subplot(3, 1, 2)
    plt.plot(signal_2, label='Neuron 2 (Similar Action Potential)', color='green')
    plt.title('Neuron 2 Signal')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot the delayed signal 3
    plt.subplot(3, 1, 3)
    plt.plot(signal_3, label='Neuron 3 (Shifted Action Potential)', color='red')
    plt.title('Neuron 3 Signal with Delayed Action Potential')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the signals
plot_similar_signals_with_shift(length=500, peak_frame=200, shift=20, width=20)

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_action_potential_signal(length=500, peak_frame=200, peak_value=1.0, width=20, noise_level=0.1):
    """
    Generate a synthetic action potential signal with a given peak position and add noise.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: The frame where the peak occurs.
    - peak_value: The value at the peak of the action potential.
    - width: The width of the action potential peak.
    - noise_level: The standard deviation of the Gaussian noise added to the signal.
    
    Returns:
    - A NumPy array representing the action potential signal with added noise.
    """
    signal = np.zeros(length)
    start_frame = max(0, peak_frame - width // 2)
    end_frame = min(length, peak_frame + width // 2)
    
    # Create a Gaussian-like peak for the action potential
    signal[start_frame:end_frame] = np.exp(-((np.arange(start_frame, end_frame) - peak_frame) ** 2) / (2 * (width // 4) ** 2)) * peak_value
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, length)
    signal += noise
    
    return signal

def plot_similar_signals_with_shift(length=500, peak_frame=200, shift=40, width=20, noise_level=0.1, shorter_width=10):
    """
    Plot two signals with similar action potentials, where the second signal is delayed and shorter with added noise.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: Frame at which the first signal peaks.
    - shift: The number of frames to shift the action potential in the second signal.
    - width: Width of the action potential for the first two signals.
    - noise_level: The noise level to add to the signals.
    - shorter_width: The width of the action potential for the third signal.
    """
    # Generate two similar action potentials with noise
    signal_1 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width, noise_level=noise_level)
    signal_2 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width, noise_level=noise_level)
    
    # Generate the third signal with the shifted and shorter action potential
    signal_3 = generate_action_potential_signal(length=length, peak_frame=peak_frame + shift, peak_value=1.0, width=shorter_width, noise_level=noise_level)
    
    # Plot the signals
    plt.figure(figsize=(12, 10))

    # Plot signal 1
    plt.subplot(3, 1, 1)
    plt.plot(signal_1, label='Neuron 1 (Original Signal)', color='blue')
    plt.title('Neuron 1 Signal with Noise')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot signal 2 directly below
    plt.subplot(3, 1, 2)
    plt.plot(signal_2, label='Neuron 2 (Similar Action Potential with Noise)', color='green')
    plt.title('Neuron 2 Signal with Noise')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot the delayed and shorter signal 3
    plt.subplot(3, 1, 3)
    plt.plot(signal_3, label='Neuron 3 (Delayed and Shorter Action Potential with Noise)', color='red')
    plt.title('Neuron 3 Signal with Delayed & Shorter Action Potential')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the signals with noise and shifted action potential
plot_similar_signals_with_shift(length=500, peak_frame=200, shift=40, width=20, noise_level=0.1, shorter_width=10)

# %%

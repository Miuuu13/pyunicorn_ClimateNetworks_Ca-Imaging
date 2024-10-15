
#%% [0]

# Imports
import os
import re     
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

# Abbreviations:
#df = dataFrame (pandas)
#np = NumPy array
#cwd = current working directory
#nw network

 #%% [1]
""" I.) Access data from Ca-Imaging, stored in .mat files """

def walk_through_data_and_extract_data_from_matlab_file():
    """  Access .mat files in all sub folders of "data" in cwd 
    @params: not needed, starts at cwd
    @return: dictionary, containing: 
    - key: <animal_id>_Batch<letter from A-F>, where following is stored:
        - c_raw_all (luminescence of neurons over frames)
        - start_frame_session (info about start of a session)
    """
    # initialize dict to store data
    c_raw_all_dict = {}
    
    # regex pattern (regular expression) to capture both letter (A-F) and digits (3-4 digit number)
    regex_pattern = r"Batch_([A-F])_2022_(\d{3,4})_CFC_GPIO"
    # ([A-F]) captures the letter (A-F), could be adjusted/extended
    # (\d{3,4}) captures the 3- or 4-digit number
    
    # start at cwd (current working directory)
    for root, dirs, files in os.walk(os.getcwd()): 
        for file in files:
            # check if the file is named "Data_Miniscope_PP.mat"
            if file == "Data_Miniscope_PP.mat":
                # get the folder name
                folder_name = os.path.basename(root) #os.path.dirname(path) not useful!
                # match folder name with the regex pattern
                match = re.search(regex_pattern, folder_name)
                
                if match:
                    # extract the letter (e.g., A, B, etc.) and the 3- or 4-digit number
                    batch_letter = match.group(1)  # the letter part (A-F)
                    animal_id = match.group(2)  # the digit part (3- or 4-digit number)

                    # construct the key in the format '###_A', '####_B', etc.
                    key = f"{animal_id}_{batch_letter}"

                    # construct full file path
                    file_path = os.path.join(root, file)

                    # open the .mat file using h5py (because of the file version)
                    with h5py.File(file_path, 'r') as h5_file:
                        # Access the "Data" group and extract the "C_raw_all" dataset
                        data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
                        # Access the "Data" group and extract the 'Start_Frame_Session' dataset
                        start_frame_session = h5_file['Data']['Start_Frame_Session'][:]
                        
                        # store the data in the dictionary using the key
                        c_raw_all_dict[key] = {
                            'C_Raw_all': data_c_raw_all,
                            'Start_Frame_Session': start_frame_session
                        }
                        

    # return the dictionary with all extracted data
    return c_raw_all_dict
c_raw_all_dict = walk_through_data_and_extract_data_from_matlab_file()

#%% [2] - check dict 
#Check dict and keys
print(c_raw_all_dict)
c_raw_all_dict.keys()

#%% [3] - 
""" II.) Generate dictionary, capturing session-specific info about Ca-Imaging data """
# use "c_raw_all_dict" and split info in key "data_c_raw_all according" to key "start_frame_session"

"changed split completely (8am 20SEP)"
def split_c_raw_all_into_sessions_dict(c_raw_all_dict):
    # Initialize new dict to store the splited sessions
    session_dict = {}
    
    # Loop through each key in original dict (c_raw_all_dict)
    for key, data in c_raw_all_dict.items():
        # Get the 'C_Raw_all' matrix and 'Start_Frame_Session' data
        c_raw_all = data['C_Raw_all']
        start_frames = data['Start_Frame_Session'][0]
        
        # Get the total number of rows (frames) in 'C_Raw_all'
        #needed for having end of last session
        total_frames = c_raw_all.shape[0]
        
        # Append the total number of frames as the end of the last session
        start_frames = list(start_frames) + [total_frames]
        
        # Loop through each session and split the 'C_Raw_all' matrix
        for i in range(len(start_frames) - 1):
            # Define start and end frame for each session
            start_frame = int(start_frames[i])
            end_frame = int(start_frames[i + 1])
            
            # Slice the 'C_Raw_all' matrix for the current session
            session_data = c_raw_all[start_frame:end_frame, :]
            
            # Create a new key for the session (e.g., '1022_B_s1', '1022_B_s2', etc.)
            #no muli-indexing here for easier access in the analysis function (later)
            session_key = f"{key}_s{i + 1}"
            
            # Store the splitted session data in the new dict
            session_dict[session_key] = session_data
    
    return session_dict

#function call on c_raw_all_dict
split_sessions_dict = split_c_raw_all_into_sessions_dict(c_raw_all_dict)

#%% [4]
# check'split_sessions_dict'
split_sessions_dict

#check keys: are all sessions processed?
for key in split_sessions_dict:
    print(key)
for key, session_data in split_sessions_dict.items():
    rows, cols = session_data.shape
    print(f"Session {key}: Rows/Frames: {rows}; Columns/Neurons: {cols} ")


# Until this point, data from matlab file is extracted and saved into a dictionary.
# Each animal id includes now 10 elements where the data from c_raw_all is 
# splitted into the 10 experimental sessions
# %%

# Get Number of Rows (Frames) and Columns (Neurons) for Each Session
def print_session_shapes(session_dict):
    """
    Prints the number of rows (frames) and columns (neurons) for each session in the session_dict.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    """
    for session_key, session_data in session_dict.items():
        # Get number of rows and columns for the current session
        rows, cols = session_data.shape
        
        # format
        print(f"Session {session_key}: Rows/Frames: {rows}; Columns/Neurons: {cols}")

print_session_shapes(split_sessions_dict)

#%%
### Restart ###

def plot_first_10_neurons_separately_same_scale(session_dict, session_key):
    """
    Plots the first 10 neurons' activity (luminescence over frames) in separate plots for a given session,
    with the same y-axis scale (max value of all neurons).
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(50, session_data.shape[1])
    
    # Calculate the maximum value across the first 10 neurons for y-axis scaling
    max_value = np.max(session_data[:, :num_neurons])
    
    # Create individual plots for each neuron with the same y-axis scale
    for neuron_id in range(num_neurons):
        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(session_data[:, neuron_id], label=f'Neuron {neuron_id}')
        
        plt.title(f"Neuron {neuron_id} Activity for {session_key}")
        plt.xlabel("Frames")
        plt.ylabel("Luminescence (Neuron Activity)")
        plt.ylim(0, max_value)  # Set y-axis limits based on the maximum value
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage:
# Plot the first 10 neurons in separate plots for session '1022_B_s1' with the same y-axis scale
plot_first_10_neurons_separately_same_scale(split_sessions_dict, '1022_B_s6')


# %%

""" handle Nan"""
def plot_first_10_neurons_separately_skip_invalid(session_dict, session_key):
    """
    Plots the first 10 neurons' activity (luminescence over frames) in separate plots for a given session,
    with the same y-axis scale (max value of all neurons) and skips neurons with NaN/Inf values.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(100, session_data.shape[1])
    
    # Calculate the maximum value across the first 10 neurons that do not have NaN or Inf values
    valid_data = session_data[:, :num_neurons]
    valid_data = valid_data[~np.isnan(valid_data).any(axis=1)]  # Remove rows with NaN values
    valid_data = valid_data[~np.isinf(valid_data).any(axis=1)]  # Remove rows with Inf values

    if valid_data.size == 0:
        print(f"All neurons in {session_key} contain NaN or Inf values.")
        return
    
    max_value = np.max(valid_data)  # Get the maximum value after excluding NaN/Inf
    
    # Create individual plots for each neuron with the same y-axis scale
    for neuron_id in range(num_neurons):
        neuron_data = session_data[:, neuron_id]
        
        # Check if the neuron contains NaN or Inf values
        if np.isnan(neuron_data).any() or np.isinf(neuron_data).any():
            print(f"Neuron {neuron_id} in session {session_key} contains NaN or Inf values and was skipped.")
            continue  # Skip plotting this neuron

        # Plot only if valid
        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(neuron_data, label=f'Neuron {neuron_id}')
        
        plt.title(f"Neuron {neuron_id} Activity for {session_key}")
        plt.xlabel("Frames")
        plt.ylabel("Luminescence (Neuron Activity)")
        plt.ylim(0, max_value)  # Set y-axis limits based on the maximum valid value
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage:
# Plot the first 10 neurons in separate plots for session '1022_B_s1' and skip invalid neurons
plot_first_10_neurons_separately_skip_invalid(split_sessions_dict, '1022_B_s1')
#%%

""" count nan values per session """

def count_nan_values_in_sessions(session_dict):
    """
    Counts and prints the number of NaN values in each session in the session_dict.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    """
    for session_key, session_data in session_dict.items():
        # Count the total number of NaN values in the session data
        nan_count = np.isnan(session_data).sum()
        
        # Print the result
        if nan_count > 0:
            print(f"Session {session_key} contains {nan_count} NaN values.")
        else:
            print(f"Session {session_key} contains no NaN values.")

# Example usage:
# Count and print the number of NaN values for each session in the split_sessions_dict
count_nan_values_in_sessions(split_sessions_dict)

# %%
""" nans per neuron in one session """

def display_nan_count_per_neuron(session_dict, session_key):
    """
    Displays the number of NaN values for each neuron in a specific session.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to check (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Loop through each neuron (column) and count the NaN values
    for neuron_id in range(session_data.shape[1]):
        nan_count = np.isnan(session_data[:, neuron_id]).sum()
        
        # Print the neuron ID and its respective NaN count
        if nan_count > 0:
            print(f"Neuron {neuron_id} in session {session_key} contains {nan_count} NaN values.")
        else:
            print(f"Neuron {neuron_id} in session {session_key} contains no NaN values.")

# Example usage:
# Display the number of NaN values per neuron in session '1022_B_s1'
display_nan_count_per_neuron(split_sessions_dict, '1022_B_s5')

#%%

""" total values (incl. nan)"""
def display_total_values_per_neuron(session_dict, session_key):
    """
    Displays the total number of values (frames) for each neuron in a specific session.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to check (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # The total number of values per neuron is the number of rows (frames)
    total_frames = session_data.shape[0]

    # Loop through each neuron (column) and print the total number of values
    for neuron_id in range(session_data.shape[1]):
        print(f"Neuron {neuron_id} in session {session_key} has {total_frames} total values (frames).")

# Example usage:
# Display the total number of values per neuron in session '1022_B_s1'
display_total_values_per_neuron(split_sessions_dict, '1022_B_s1')

# %%

""" Combine count total and count nan values per neuron"""

def display_total_and_nan_values_per_neuron(session_dict, session_key):
    """
    Displays the total number of values (frames) and the number of NaN values for each neuron
    in a specific session.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to check (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # The total number of values per neuron is the number of rows (frames)
    total_frames = session_data.shape[0]

    # Loop through each neuron (column)
    for neuron_id in range(session_data.shape[1]):
        # Count the NaN values for the current neuron
        nan_count = np.isnan(session_data[:, neuron_id]).sum()

        # Prepare the message based on whether the neuron has NaN values or not
        if nan_count > 0:
            print(f"Neuron {neuron_id} in session {session_key} has {total_frames} total values (frames) and contains {nan_count} NaN values.")
        else:
            print(f"Neuron {neuron_id} in session {session_key} has {total_frames} total values (frames) and contains no NaN values.")

# Example usage:
# Display the total number of values and NaN counts per neuron in session '1022_B_s1'
display_total_and_nan_values_per_neuron(split_sessions_dict, '1022_B_s7')

# %%


#######################CHECK dff and z score normalization ##################################################

def check_deltaf_f_and_zscore(c_raw_all_dict):
    """
    Checks whether deltaF/F and z-score have been applied to the data in c_raw_all_dict.
    
    Parameters:
    - c_raw_all_dict: dictionary containing the raw calcium imaging data.
    
    Returns:
    - Dictionary with session keys and booleans indicating whether deltaF/F or z-score is likely applied.
    """
    results = {}

    for session_key, data in c_raw_all_dict.items():
        c_raw_all = data['C_Raw_all']

        # Check for deltaF/F by inspecting if the data has both positive and negative values?
        #TODO is this already enough???
        has_positive_values = (c_raw_all > 0).any()
        has_negative_values = (c_raw_all < 0).any()
        deltaf_f_applied = has_positive_values and has_negative_values

        # Check for z-score by inspecting if the data has:
        # 1. mean around 0 and 
        # 2. standard deviation around 1
        mean_value = np.mean(c_raw_all)
        std_value = np.std(c_raw_all)
        zscore_applied = np.isclose(mean_value, 0, atol=0.1) and np.isclose(std_value, 1, atol=0.1)

        # Store the results for this session
        results[session_key] = {
            'deltaF/F_applied': deltaf_f_applied,
            'zscore_applied': zscore_applied
        }

        # Print results for the session
        print(f"Session {session_key}:")
        print(f"  - DeltaF/F applied: {'Yes' if deltaf_f_applied else 'No'}")
        print(f"  - Z-score applied: {'Yes' if zscore_applied else 'No'}")
        print(f"  - Mean of data: {mean_value:.4f}, Standard deviation of data: {std_value:.4f}")
    
    return results

# Example usage:
# Check for deltaF/F and z-score in the c_raw_all_dict
check_deltaf_f_and_zscore(c_raw_all_dict)
#%%
""" Excluded global way, does not make sense! """
""" Z score - 3 ways 

# Global Z-Score for c_raw_all_dict:

#     Concatenate all C_Raw_all data across all sessions into a single matrix (all_data), 
#     compute the global mean and standard deviation (mean_all and std_all), 
#     and then apply the z-score normalization across all frames and neurons. 
#     The normalized data is stored in c_raw_all_dict_z.

Z-Score for Each Session in split_sessions_dict:

    For each session, compute the mean and standard deviation of the entire session and 
    apply the z-score normalization. 
    The normalized session-specific data is stored in split_sessions_dict_z.

Z-Score for Each Neuron in Each Session:

    For each neuron (i.e., each column) within each session, 
    compute the mean and standard deviation column-wise and apply the z-score normalization. 
    The result is stored in split_sessions_dict_z_per_neuron.
"""

def apply_zscore_all_modes(c_raw_all_dict, split_sessions_dict):
    """
    Applies z-score normalization in three modes:
    1. On the entire dataset in c_raw_all_dict across all frames and neurons.
    2. On each session in split_sessions_dict.
    3. On each neuron within each session in split_sessions_dict.
    
    Parameters:
    - c_raw_all_dict: dictionary containing the raw calcium imaging data.
    - split_sessions_dict: dictionary containing session-specific data.
    
    Returns:
    - c_raw_all_dict_z: z-scored data for the entire dataset (c_raw_all_dict).
    - split_sessions_dict_z: z-scored data for each session (split_sessions_dict).
    - split_sessions_dict_z_per_neuron: z-scored data for each neuron per session (split_sessions_dict).
    """

    """ Excluded global way, does not make sense! """
    
    # # 1. Z-score on the entire c_raw_all_dict (over all columns and frames)
    # all_data = np.concatenate([data['C_Raw_all'] for data in c_raw_all_dict.values()], axis=0)
    # mean_all = np.mean(all_data)
    # std_all = np.std(all_data)
    
    # c_raw_all_dict_z = {}
    # for session_key, data in c_raw_all_dict.items():
    #     c_raw_all = data['C_Raw_all']
        
    #     # Apply global z-score normalization
    #     c_raw_all_z = (c_raw_all - mean_all) / std_all
        
    #     # Save in the new dictionary
    #     c_raw_all_dict_z[session_key] = {
    #         'C_Raw_all': c_raw_all_z,
    #         'Start_Frame_Session': data['Start_Frame_Session']
    #     }

    # 2. Z-score on each session independently (across all neurons in the session)
    split_sessions_dict_z = {}
    for session_key, session_data in split_sessions_dict.items():
        mean_session = np.mean(session_data)
        std_session = np.std(session_data)
        
        # Apply session-level z-score normalization
        session_data_z = (session_data - mean_session) / std_session
        
        # Save in the new dictionary
        split_sessions_dict_z[session_key] = session_data_z

    # 3. Z-score on each neuron individually within each session
    split_sessions_dict_z_per_neuron = {}
    for session_key, session_data in split_sessions_dict.items():
        # Apply z-score normalization to each neuron (column)
        session_data_z_per_neuron = (session_data - np.mean(session_data, axis=0)) / np.std(session_data, axis=0)
        
        # Save in the new dictionary
        split_sessions_dict_z_per_neuron[session_key] = session_data_z_per_neuron

    return split_sessions_dict_z, split_sessions_dict_z_per_neuron #c_raw_all_dict_z, 

# Example usage:
# Apply z-score normalization in all three modes
#c_raw_all_dict_z, 
split_sessions_dict_z, split_sessions_dict_z_per_neuron = apply_zscore_all_modes(c_raw_all_dict, split_sessions_dict)

#%%

#check with check_deltaf_f_and_zscore

def check_deltaf_f_and_zscore(session_dict):
    """
    Checks whether deltaF/F and z-score have been applied to the data in session_dict.
    
    Parameters:
    - session_dict: dictionary containing session-specific data (as NumPy arrays).
    
    Returns:
    - Dictionary with session keys and booleans indicating whether deltaF/F or z-score is likely applied.
    """
    results = {}

    for session_key, session_data in session_dict.items():
        # Check for deltaF/F by inspecting if the data has both positive and negative values
        has_positive_values = (session_data > 0).any()
        has_negative_values = (session_data < 0).any()
        deltaf_f_applied = has_positive_values and has_negative_values

        # Check for z-score by inspecting if the data has mean ~0 and standard deviation ~1
        mean_value = np.mean(session_data)
        std_value = np.std(session_data)
        zscore_applied = np.isclose(mean_value, 0, atol=0.1) and np.isclose(std_value, 1, atol=0.1)

        # Store the results for this session
        results[session_key] = {
            'deltaF/F_applied': deltaf_f_applied,
            'zscore_applied': zscore_applied
        }

        # Print results for the session
        print(f"Session {session_key}:")
        print(f"  - DeltaF/F applied: {'Yes' if deltaf_f_applied else 'No'}")
        print(f"  - Z-score applied: {'Yes' if zscore_applied else 'No'}")
        print(f"  - Mean of data: {mean_value:.4f}, Standard deviation of data: {std_value:.4f}")
    
    return results

# Example usage:
# Check for deltaF/F and z-score in split_sessions_dict_z and split_sessions_dict_z_per_neuron
check_deltaf_f_and_zscore(split_sessions_dict_z)

#%%
check_deltaf_f_and_zscore(split_sessions_dict_z_per_neuron)

#%%

""" plot z scored neuron traces"""



def plot_first_10_neurons_separately_same_scale_zscore_skip_nan(session_dict_z, session_key):
    """
    Plots the first 10 neurons' activity (z-scored data) in separate plots for a given session,
    skipping NaN values and using the same y-axis scale (based on the non-NaN min and max).
    
    Parameters:
    - session_dict_z: dictionary containing the z-scored data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    """
    if session_key not in session_dict_z:
        print(f"Session {session_key} not found.")
        return

    # Get the z-scored data for the specified session
    session_data_z = session_dict_z[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(10, session_data_z.shape[1])

    # Calculate the global min and max across the first 10 neurons, ignoring NaNs
    min_value = np.nanmin(session_data_z[:, :num_neurons])
    max_value = np.nanmax(session_data_z[:, :num_neurons])

    # Create individual plots for each neuron with the same y-axis scale
    for neuron_id in range(num_neurons):
        neuron_data = session_data_z[:, neuron_id]
        
        # Check for NaN values and skip them
        if np.isnan(neuron_data).any():
            print(f"Neuron {neuron_id} in session {session_key} contains NaN values and they will be skipped.")
            neuron_data = neuron_data[~np.isnan(neuron_data)]  # Filter out NaN values

        if neuron_data.size == 0:
            print(f"Neuron {neuron_id} has no valid data points after removing NaN values.")
            continue  # Skip plotting if no valid data remains

        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(neuron_data, label=f'Neuron {neuron_id}')
        
        plt.title(f"Neuron {neuron_id} Activity (Z-Scored) for {session_key}")
        plt.xlabel("Frames")
        plt.ylabel("Z-Scored Activity")
        plt.ylim(min_value, max_value)  # Set y-axis limits based on the global min/max of valid data
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage:
# Plot the first 10 neurons in separate plots for z-scored data in session '1022_B_s1'
#plot_first_10_neurons_separately_same_scale_zscore_skip_nan(split_sessions_dict_z_per_neuron, '1022_B_s1')

# Example usage:
# Plot the first 10 neurons in separate plots for z-scored data in session '1022_B_s1'
#plot_first_10_neurons_separately_same_scale_zscore(split_sessions_dict_z_per_neuron, '1022_B_s1')




#%%
# FOURIER

def fourier_smooth(signal, cutoff_frequency):
    """
    Applies Fourier transform to smooth the signal by filtering out high-frequency components.
    
    Parameters:
    - signal: 1D array of the signal to be smoothed (e.g., neuron activity).
    - cutoff_frequency: frequency threshold to remove high-frequency components.
    
    Returns:
    - smoothed_signal: the smoothed signal in the time domain.
    """
    # Apply Fourier Transform (FFT) to convert to frequency domain
    fft_coeffs = np.fft.fft(signal)
    
    # Generate the frequency array
    frequencies = np.fft.fftfreq(len(signal))
    
    # Filter out high frequencies by zeroing out frequencies higher than cutoff_frequency
    fft_coeffs[np.abs(frequencies) > cutoff_frequency] = 0
    
    # Apply the Inverse Fourier Transform to get back the smoothed signal
    smoothed_signal = np.fft.ifft(fft_coeffs)
    
    # Return only the real part (since the imaginary part should be negligible after IFFT)
    return np.real(smoothed_signal)

# Function to plot neurons with Fourier smoothing
def plot_first_10_neurons_fourier_smoothing(session_dict, session_key, cutoff_frequency):
    """
    Plots the first 10 neurons' activity after applying Fourier smoothing.
    
    Parameters:
    - session_dict: dictionary containing session-specific data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    - cutoff_frequency: frequency threshold for Fourier smoothing.
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(10, session_data.shape[1])

    # Calculate the global min and max across the first 10 neurons for y-axis scaling
    min_value = np.nanmin(session_data[:, :num_neurons])
    max_value = np.nanmax(session_data[:, :num_neurons])

    # Create individual plots for each neuron with Fourier smoothing
    for neuron_id in range(num_neurons):
        neuron_data = session_data[:, neuron_id]
        
        # Check for NaN values and skip them
        if np.isnan(neuron_data).any():
            print(f"Neuron {neuron_id} in session {session_key} contains NaN values and they will be skipped.")
            neuron_data = neuron_data[~np.isnan(neuron_data)]  # Filter out NaN values

        if neuron_data.size == 0:
            print(f"Neuron {neuron_id} has no valid data points after removing NaN values.")
            continue  # Skip plotting if no valid data remains

        # Apply Fourier smoothing to the neuron data
        smoothed_data = fourier_smooth(neuron_data, cutoff_frequency)

        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(smoothed_data, label=f'Neuron {neuron_id} (Smoothed)')
        
        plt.title(f"Neuron {neuron_id} Activity (Fourier-Smoothed) for {session_key}")
        plt.xlabel("Frames")
        plt.ylabel("Smoothed Activity")
        plt.ylim(min_value, max_value)  # Set y-axis limits based on the global min/max of valid data
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage:
# Plot the first 10 neurons in separate plots with Fourier smoothing for z-scored data
plot_first_10_neurons_fourier_smoothing(split_sessions_dict_z_per_neuron, '1022_B_s1', cutoff_frequency=0.1)
#%%

import matplotlib.pyplot as plt
import numpy as np

def plot_unsmoothed_vs_smoothed(session_dict, session_key, cutoff_frequency):
    """
    Plots the first 10 neurons' activity, both unsmoothed and smoothed, in a 2x10 grid.
    
    Parameters:
    - session_dict: dictionary containing session-specific data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    - cutoff_frequency: frequency threshold for Fourier smoothing.
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(10, session_data.shape[1])

    # Create a figure with a 2x10 grid for the plots
    fig, axes = plt.subplots(2, 10, figsize=(200, 6))  # 2 rows, 10 columns for unsmoothed and smoothed
    
    # Iterate over the first 10 neurons
    for neuron_id in range(num_neurons):
        neuron_data = session_data[:, neuron_id]
        
        # Check for NaN values and skip them
        if np.isnan(neuron_data).any():
            print(f"Neuron {neuron_id} in session {session_key} contains NaN values and they will be skipped.")
            neuron_data = neuron_data[~np.isnan(neuron_data)]  # Filter out NaN values

        if neuron_data.size == 0:
            print(f"Neuron {neuron_id} has no valid data points after removing NaN values.")
            continue  # Skip plotting if no valid data remains

        # Apply Fourier smoothing to the neuron data
        smoothed_data = fourier_smooth(neuron_data, cutoff_frequency)

        # Plot the original (unsmoothed) data
        axes[0, neuron_id].plot(neuron_data, label=f'Neuron {neuron_id} (Unsmoothed)')
        axes[0, neuron_id].set_title(f"Neuron {neuron_id} Unsmoothed")
        axes[0, neuron_id].set_xlabel("Frames")
        axes[0, neuron_id].set_ylabel("Activity")
        axes[0, neuron_id].legend(loc='upper right')

        # Plot the smoothed data
        axes[1, neuron_id].plot(smoothed_data, label=f'Neuron {neuron_id} (Smoothed)', color='orange')
        axes[1, neuron_id].set_title(f"Neuron {neuron_id} Smoothed")
        axes[1, neuron_id].set_xlabel("Frames")
        axes[1, neuron_id].set_ylabel("Activity")
        axes[1, neuron_id].legend(loc='upper right')

    # Adjust the layout to ensure there is no overlap
    plt.tight_layout()
    plt.show()

# Example usage:
# Plot unsmoothed vs smoothed for the first 10 neurons with Fourier smoothing
plot_unsmoothed_vs_smoothed(split_sessions_dict_z_per_neuron, '1022_B_s1', cutoff_frequency=0.1)

#%%
#plot_first_10_neurons_separately_same_scale_zscore(split_sessions_dict_z, '1022_B_s1')
#%%
""" rerun analysis for unnormalized and each 3 z score normalization "ways" to compare """

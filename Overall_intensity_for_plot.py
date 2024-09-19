""" sum up activity for each frame """

#Previous analysis of activity only included:

# session_key,neurons_all_nan,neurons_active
# s1_1-22360,       74,         197
# s2_22361-44916,   123,        148
# s3_44917-91942,   101,        170
# s4_91943-124993,  110,        161
# s5_124994-158590, 108,        163
# s6_158591-191577, 120,        151
# s7_191578-225080, 125,        146
# s8_225081-243944, 109,        162
# s9_243945-290515, 116,        155
# s10_290516-309202,173,        98


#%%
""" Network analysis for one animal id (Batch A or B)"""
# Analysis start: SEP24
# collaboration: AG Lutz, Larglinda Islami
# Ca-Imaging, 1-Photon
# Batches organized in folder "data" and subfolders "Batch_A", "Batch_B"
# analysis of .mat, using info about when tone is played for "alignment"
# Info about data and project: https://gitlab.rlp.net/computationalresilience/ca-imaging



#%% [0]
""" Imports """
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os
import re
from scipy.stats import zscore

#%% [1]
""" 1. Data Access and Preprocessing """

def walk_through_folders_and_extract_data():
    c_raw_all_dict = {}
    pattern = r"Batch_[A-F]_2022_(\d{3,4})_CFC_GPIO"

    for root, dirs, files in os.walk(os.getcwd()):
        for file in files:
            if file == "Data_Miniscope_PP.mat":
                folder_name = os.path.basename(root)
                match = re.search(pattern, folder_name)

                if match:
                    key = match.group(1)

                    file_path = os.path.join(root, file)
                    with h5py.File(file_path, 'r') as h5_file:
                        data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
                        start_frame_session = h5_file['Data']['Start_Frame_Session'][:]
                        
                        c_raw_all_dict[key] = {
                            'C_Raw_all': data_c_raw_all,
                            'Start_Frame_Session': start_frame_session
                        }

    return c_raw_all_dict

# Step 2: Z-Score Normalization
def zscore_normalize_c_raw_all(c_raw_all_dict):
    c_raw_all_zscore_dict = {}

    for key, data in c_raw_all_dict.items():
        c_raw_all_matrix = data['C_Raw_all']
        start_frame_session = data['Start_Frame_Session']
        
        # Z-score normalize the C_Raw_all matrix, skip NaNs
        c_raw_all_matrix_zscore = zscore(c_raw_all_matrix, axis=0, nan_policy='omit')

        # Save normalized data into new dictionary
        c_raw_all_zscore_dict[key] = {
            'C_Raw_all': c_raw_all_matrix_zscore,
            'Start_Frame_Session': start_frame_session
        }

    return c_raw_all_zscore_dict

# Step 3: Sum across neurons for each frame and cut at the minimum frame length
def sum_across_neurons(c_raw_all_dict):
    min_frame_length = min([data['C_Raw_all'].shape[0] for data in c_raw_all_dict.values()])

    summed_data = {}

    for key, data in c_raw_all_dict.items():
        c_raw_all_matrix = data['C_Raw_all'][:min_frame_length, :]
        
        # Sum across neurons (columns) for each frame (row), ignoring NaNs
        summed_frames = np.nansum(c_raw_all_matrix, axis=1)

        summed_data[key] = summed_frames

    return summed_data

# Step 4: Repeat the same summing operation for z-scored data
def sum_across_neurons_zscore(c_raw_all_zscore_dict):
    min_frame_length = min([data['C_Raw_all'].shape[0] for data in c_raw_all_zscore_dict.values()])

    summed_zscore_data = {}

    for key, data in c_raw_all_zscore_dict.items():
        c_raw_all_matrix = data['C_Raw_all'][:min_frame_length, :]
        
        # Sum across neurons (columns) for each frame (row), ignoring NaNs
        summed_frames = np.nansum(c_raw_all_matrix, axis=1)

        summed_zscore_data[key] = summed_frames

    return summed_zscore_data

# Running the functions
c_raw_all_dict = walk_through_folders_and_extract_data()
print("C_Raw_all Dictionary:")
print(c_raw_all_dict)

c_raw_all_zscore_dict = zscore_normalize_c_raw_all(c_raw_all_dict)
print("\nC_Raw_all Z-Score Normalized Dictionary:")
print(c_raw_all_zscore_dict)

summed_data = sum_across_neurons(c_raw_all_dict)
print("\nSummed Data Across Neurons (Original):")
print(summed_data)

summed_zscore_data = sum_across_neurons_zscore(c_raw_all_zscore_dict)
print("\nSummed Data Across Neurons (Z-Scored):")
print(summed_zscore_data)
# %%
# Step 5: Sum across all data for intensity over time (across all neurons in all datasets)
def sum_across_all_datasets(c_raw_all_dict):
    min_frame_length = min([data['C_Raw_all'].shape[0] for data in c_raw_all_dict.values()])

    summed_all_data = np.zeros(min_frame_length)

    for key, data in c_raw_all_dict.items():
        c_raw_all_matrix = data['C_Raw_all'][:min_frame_length, :]
        
        # Sum across neurons (columns) for each frame (row), ignoring NaNs
        summed_frames = np.nansum(c_raw_all_matrix, axis=1)

        # Sum across all datasets
        summed_all_data += summed_frames

    return summed_all_data

# Step 6: Repeat the same operation for z-scored data
def sum_across_all_datasets_zscore(c_raw_all_zscore_dict):
    min_frame_length = min([data['C_Raw_all'].shape[0] for data in c_raw_all_zscore_dict.values()])

    summed_all_zscore_data = np.zeros(min_frame_length)

    for key, data in c_raw_all_zscore_dict.items():
        c_raw_all_matrix = data['C_Raw_all'][:min_frame_length, :]
        
        # Sum across neurons (columns) for each frame (row), ignoring NaNs
        summed_frames = np.nansum(c_raw_all_matrix, axis=1)

        # Sum across all datasets
        summed_all_zscore_data += summed_frames

    return summed_all_zscore_data

# Running the new sum operations
summed_all_data = sum_across_all_datasets(c_raw_all_dict)
print("\nSummed Intensity Across All Neurons (Original Data):")
print(summed_all_data)

summed_all_zscore_data = sum_across_all_datasets_zscore(c_raw_all_zscore_dict)
print("\nSummed Intensity Across All Neurons (Z-Scored Data):")
print(summed_all_zscore_data)

#%%

""" Check frame counts """

# Let's write a function that checks the number of frames (rows) for each animal
def check_frame_counts(c_raw_all_dict):
    frame_counts = {}
    for key, data in c_raw_all_dict.items():
        frame_counts[key] = data['C_Raw_all'].shape[0]
    return frame_counts

# Assuming c_raw_all_dict was populated, let's check the number of frames for each animal
frame_counts = check_frame_counts(c_raw_all_dict)

# Displaying the frame counts for each animal
frame_counts

# %%
# Function to check the number of neurons (columns) for each animal's data matrix
def check_neuron_counts(c_raw_all_dict):
    neuron_counts = {}
    for key, data in c_raw_all_dict.items():
        neuron_counts[key] = data['C_Raw_all'].shape[1]  # Number of columns corresponds to the neurons
    return neuron_counts

# Assuming c_raw_all_dict was populated, let's check the number of neurons for each animal
neuron_counts = check_neuron_counts(c_raw_all_dict)

# Displaying the neuron counts for each animal
neuron_counts

# %%
""" new try"""
# Function to sum the intensity of each neuron over all frames for each animal
def sum_neuron_intensity(c_raw_all_dict):
    neuron_summed_intensity = {}
    for key, data in c_raw_all_dict.items():
        c_raw_all_matrix = data['C_Raw_all']
        
        # Sum across all frames (rows) for each neuron (columns), ignoring NaNs
        summed_neurons = np.nansum(c_raw_all_matrix, axis=0)

        neuron_summed_intensity[key] = summed_neurons
    
    return neuron_summed_intensity

# Assuming c_raw_all_dict was populated, let's sum the neuron intensity for each animal
neuron_intensity_sums = sum_neuron_intensity(c_raw_all_dict)

# Displaying the summed intensity for each neuron per animal
neuron_intensity_sums

# %%
neuron_intensity_sums
# %%
# Placeholder example neuron intensity sums for plotting

#neuron_intensity_sums

def plot_neuron_intensity_sums(neuron_intensity_sums):
    # Plot individual plots for each animal's neuron intensity sums
    for key, summed_neurons in neuron_intensity_sums.items():
        plt.figure()
        plt.plot(summed_neurons)
        plt.title(f'Animal {key} - Neuron Intensity Sums')
        plt.xlabel('Neuron Index')
        plt.ylabel('Summed Intensity')
        plt.grid(True)
        plt.show()

    # Combined plot for all animals
    plt.figure()
    
    for key, summed_neurons in neuron_intensity_sums.items():
        plt.plot(summed_neurons, label=f'Animal {key}')
    
    plt.title('Neuron Intensity Sums for All Animals')
    plt.xlabel('Neuron Index')
    plt.ylabel('Summed Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the results with example data
plot_neuron_intensity_sums(neuron_intensity_sums)

# %%

# # Function to sum intensity for each frame across all neurons and all animals, cutting at the minimum frame length
# def sum_intensity_across_frames_and_animals(summed_data):
#     # Determine the minimum frame length across all animals
#     min_frame_length = min([len(summed_frames) for summed_frames in summed_data.values()])

#     # Initialize an array to store the total summed intensity across all animals
#     total_intensity_across_frames = np.zeros(min_frame_length)

#     # Sum intensity for each frame across all animals
#     for key, summed_frames in summed_data.items():
#         total_intensity_across_frames += summed_frames[:min_frame_length]  # Cut at min_frame_length

#     return total_intensity_across_frames

# # Summing the intensity across all neurons and animals for each frame using the example data
# total_intensity_across_frames = sum_intensity_across_frames_and_animals(summed_data_example)

# # Plot the total intensity across all animals for each frame
# plt.figure()
# plt.plot(total_intensity_across_frames, label='Total Intensity Across All Animals', color='r')
# plt.title('Total Intensity Over Frames (Across All Animals)')
# plt.xlabel('Frames')
# plt.ylabel('Total Summed Intensity')
# plt.grid(True)
# plt.show()

# %%

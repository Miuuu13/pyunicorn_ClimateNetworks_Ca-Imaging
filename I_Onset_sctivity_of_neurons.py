""" Calculate the onset activity of neurons """

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

""" Animal ids """
#batchB_ids = [935, 990, 1002, 1012, 1022, 1037] # remaining in batch B

#TODO Add info about R+/R-
#R+
#R-

# discarded 934 and 1031 into backup folder due to missing frames (in total/ for 10 sessions)


#%% [0]
""" Imports """
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

#%% [1]
""" 1. Data Access and Preprocessing """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis

# Define the path to the .mat or HDF5 file 
# file format trivially, stick to .mat as this is the original

path_1012 = "/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/Batch_B/Batch_B_2022_1012_CFC_GPIO/Batch_B_2022_1012_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the .mat file (load mat does not work for this .mat version) 
# and extract C_Raw_all data (matrix)
with h5py.File(path_1012, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# Convert the data  for 'C_Raw_all' and start frames to pandas DataFrames
df_c_raw_all = pd.DataFrame(data_c_raw_all)
df_start_frame_session = pd.DataFrame(start_frame_session)
# header, nr of rows/columns of the DataFrame
print(f"C_Raw_all: \n{df_c_raw_all.head()}")
rows, columns = df_c_raw_all.shape
print(f"\nNumber of rows: {rows}")
print(f"Number of columns: {columns}")
print(f"Start Frame Sessions: \n{df_start_frame_session.head()}")


#%% [2]
""" 2. Session Key Generation

Generate sessions_to_process list automatically """

# list to store the session keys
sessions_to_process = []

# Loop through the start_frame_session
for i in range(1, len(df_start_frame_session.columns)):  # Start from session 1 
    start = int(df_start_frame_session.iloc[0, i-1])
    end = int(df_start_frame_session.iloc[0, i]) - 1
    
    # Create a session key 
    session_key = f"s{i}_{start}-{end}"
    sessions_to_process.append(session_key)

# Add the final session (->goes until the last row of df_c_raw_all)
final_start = int(df_start_frame_session.iloc[0, -1])
final_end = len(df_c_raw_all) - 1
final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
sessions_to_process.append(final_session_key)

print("Generated sessions_to_process:")
print(sessions_to_process)


# ['s1_0-22360', 's2_22361-44916', ..., 's7_191578-225080']

#%% [2b]
""" Create windows and store them in a dictionary """

# dictionary to store the windows
c_raw_all_sessions = {}

# Loop through each start frame session to create the windows
for i in range(len(df_start_frame_session.columns)):
    if i == 0:
        # First window starts at index 0
        start = 0
    else:
        # Subsequent windows start at the current session index
        start = int(df_start_frame_session.iloc[0, i])

    # If not the last index, 
    #take the next start and subtract 1
    if i < len(df_start_frame_session.columns) - 1:
        end = int(df_start_frame_session.iloc[0, i+1]) - 1
    else:
        # Last window ends at the last row of df_c_raw_all
        end = rows - 1
    
    # Create a key like 's1_0-22485', 's2_22486-44647'...
    key = f"s{i+1}_{start}-{end}"
    
    # Store the corresponding rows in the dictionary
    c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

# check content
for key, df in c_raw_all_sessions.items():
    print(f"{key}: {df.shape}")

    """Check for normalization"""


# # Function to check if ΔF/F normalization or z-scoring was performed
# def check_dff_and_zscoring(df):
#     # Check if the data values are indicative of ΔF/F
#     # ΔF/F values are usually close to zero, mostly positive with occasional small negative values
#     dff_check = df.mean().min() >= -1 and df.mean().mean() > 0  
#     # General def. of ΔF/F

#     # Check if the data is z-scored (mean ~ 0 and std ~ 1 for each neuron)
#     zscore_check = (df.mean().abs() < 0.1).all() and (df.std().between(0.9, 1.1)).all()

#     if dff_check:
#         print("The data appears to be ΔF/F normalized.")
#     else:
#         print("The data does not appear to be ΔF/F normalized.")
    
#     if zscore_check:
#         print("The data appears to be z-scored.")
#     else:
#         print("The data does not appear to be z-scored.")

#check on the raw data
# check_dff_and_zscoring(df_c_raw_all)

#%%
def zscore_normalization(df):
    """Applies Z-score normalization to each column (neuron) of the DataFrame, handling NaN values properly."""
    
    #  apply Z-score normalization column-wise,
    #  handling NaN values
    return df.apply(lambda col: (col - col.mean(skipna=True)) / col.std(skipna=True), axis=0)

# Dictionary to store Z-score normalized data 
c_raw_all_sessions_zscore = {}

# Iterate over each session 
# apply Z-score normalization
for key, df in c_raw_all_sessions.items():
    print(f"Applying Z-score normalization to {key}")
    c_raw_all_sessions_zscore[key] = zscore_normalization(df)

# Check the content of the normalized data
for key, df in c_raw_all_sessions_zscore.items():
    print(f"{key}: {df.shape}")


#%%
#check keys
print(c_raw_all_sessions_zscore.keys())

# %%
# Step 1: Select the DataFrame corresponding to session 's4'
df_s4 = c_raw_all_sessions_zscore['s4_91943-124993']

# Initialize lists to store results
neurons = []
nan_counts = []
onset_counts = []

# Step 2: Traverse each column (neuron) to check for NaN values and count onsets
for neuron in df_s4.columns:
    # Step 2.1: Check if the column contains only NaN values
    nan_count = df_s4[neuron].isna().sum()
    if nan_count == len(df_s4):
        # If the neuron is entirely NaN, we discard it (skip further processing)
        continue
    
    # Step 3: Calculate the standard deviation and set the threshold as 2 * std
    std_dev = df_s4[neuron].std(skipna=True)
    threshold = 2 * std_dev
    
    # Step 4: Initialize state tracking
    is_above_threshold = False
    onset_event_count = 0
    
    # Step 5: Iterate over each frame (row) in the neuron's data
    for value in df_s4[neuron]:
        if np.isnan(value):
            continue  # Ignore NaN values
        
        if value > threshold:
            # If value crosses the threshold and we are not already in an event
            if not is_above_threshold:
                onset_event_count += 1
                is_above_threshold = True  # Mark the neuron as active (above threshold)
        else:
            # If the value falls below the threshold, reset for the next event
            is_above_threshold = False
    
    # Append the neuron, NaN count, and onset count to the results lists
    neurons.append(neuron)
    nan_counts.append(nan_count)
    onset_counts.append(onset_event_count)

# Step 6: Create the final DataFrame
df_summary = pd.DataFrame({
    'Neuron': neurons,
    'NaN Count': nan_counts,
    'Onset Count': onset_counts
})

# Print the final DataFrame
print(df_summary)

# Access index 3 from the df_summary and print it
print("\nRow at index 3:")
print(df_summary.iloc[3])




# %%
import pandas as pd
import numpy as np

# Initialize lists to store results for all sessions
all_neurons = []
all_nan_counts = []
all_onset_counts_std1 = []
all_onset_counts_std2 = []
all_onset_counts_std3 = []
all_sessions = []

# Function to count onset events given a threshold
def count_onsets(data, threshold):
    is_above_threshold = False
    onset_event_count = 0
    
    # Iterate over each frame (row) in the neuron's data
    for value in data:
        if np.isnan(value):
            continue  # Ignore NaN values
        
        if value > threshold:
            # If value crosses the threshold and we are not already in an event
            if not is_above_threshold:
                onset_event_count += 1
                is_above_threshold = True  # Mark the neuron as active (above threshold)
        else:
            # If the value falls below the threshold, reset for the next event
            is_above_threshold = False
    
    return onset_event_count

# Step 1: Traverse each session in c_raw_all_sessions_zscore
for session, df_session in c_raw_all_sessions_zscore.items():
    # Step 2: Traverse each neuron (column) in the session
    for neuron in df_session.columns:
        # Step 2.1: Check if the column contains only NaN values
        nan_count = df_session[neuron].isna().sum()
        if nan_count == len(df_session):
            # If the neuron is entirely NaN, we discard it (skip further processing)
            continue
        
        # Step 3: Calculate the standard deviation
        std_dev = df_session[neuron].std(skipna=True)
        
        # Step 4: Count the number of onset events for different thresholds
        onset_std1 = count_onsets(df_session[neuron], std_dev)
        onset_std2 = count_onsets(df_session[neuron], 2 * std_dev)
        onset_std3 = count_onsets(df_session[neuron], 3 * std_dev)
        
        # Append the results to the lists
        all_sessions.append(session)
        all_neurons.append(neuron)
        all_nan_counts.append(nan_count)
        all_onset_counts_std1.append(onset_std1)
        all_onset_counts_std2.append(onset_std2)
        all_onset_counts_std3.append(onset_std3)

# Step 5: Create the final DataFrame with the session column at the beginning
df_summary = pd.DataFrame({
    'Session': all_sessions,
    'Neuron': all_neurons,
    'NaN Count': all_nan_counts,
    'Onset_Count_std1': all_onset_counts_std1,
    'Onset_Count_std2': all_onset_counts_std2,
    'Onset_Count_std3': all_onset_counts_std3
})

# Step 6: Save the DataFrame to a CSV file
csv_filename = 'I_Onset_activity_of_neurons.csv'
df_summary.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' has been created successfully.")

# %%
""" Traverse all subfolders """

import os
import h5py
import pandas as pd
import numpy as np
import re

"""Function to count onset events given a threshold (T)
crossing T +1
no event while other is avtive
event duration: from T-crossing until T-falling""" 
def count_onsets(data, threshold):
    is_above_threshold = False
    onset_event_count = 0
    
    # Iterate over each frame (row)
    for value in data:
        if np.isnan(value):
            continue  # Ignore NaN values
        
        if value > threshold:
            # If value crosses T and not already in an "event"
            if not is_above_threshold:
                onset_event_count += 1
                is_above_threshold = True  
                # Mark the neuron as active one (above T)
        else:
            # If the value falls below T again  
            # reset for the next event
            is_above_threshold = False
    
    return onset_event_count

""" Function to process each Data_Miniscope_PP.mat file of batch B """
def process_mat_file(mat_file_path, number):
    # 1. Load the .mat file
    with h5py.File(mat_file_path, 'r') as h5_file:
        # Extract the 'C_raw_all' data and start frame session information
        data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
        start_frame_session = h5_file['Data']['Start_Frame_Session'][:]
    
    # Convert data to pandas DataFrames
    df_c_raw_all = pd.DataFrame(data_c_raw_all)
    df_start_frame_session = pd.DataFrame(start_frame_session)
    
    # Dictionary to store session-specific data
    c_raw_all_sessions = {}
    rows = df_c_raw_all.shape[0]
    
    # Generate session windows based on start_frame_session data
    for i in range(len(df_start_frame_session.columns)):
        start = int(df_start_frame_session.iloc[0, i])
        if i < len(df_start_frame_session.columns) - 1:
            end = int(df_start_frame_session.iloc[0, i + 1]) - 1
        else:
            end = rows - 1
        key = f"s{i+1}_{start}-{end}"
        c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

    # Apply Z-score normalization
    def zscore_normalization(df):
        return df.apply(lambda col: (col - col.mean(skipna=True)) / col.std(skipna=True), axis=0)

    c_raw_all_sessions_zscore = {key: zscore_normalization(df) for key, df in c_raw_all_sessions.items()}

    # Initialize lists to store results 
    all_neurons = []
    all_nan_counts = []
    all_onset_counts_std1 = []
    all_onset_counts_std2 = []
    all_onset_counts_std3 = []
    all_sessions = []

    # 2: Traverse each session in c_raw_all_sessions_zscore
    for session, df_session in c_raw_all_sessions_zscore.items():
        #Traverse each neuron (column) in the session
        for neuron in df_session.columns:
            nan_count = df_session[neuron].isna().sum()
            if nan_count == len(df_session):
                continue  # Skip neurons with only  NaN values
            
            # Calculate the std, standard deviation
            std = df_session[neuron].std(skipna=True)
            
            # Count the onset events for 
            # thresholds 1*std, 2*std, and 3*std
            onset_std1 = count_onsets(df_session[neuron], std)
            onset_std2 = count_onsets(df_session[neuron], 2 * std)
            onset_std3 = count_onsets(df_session[neuron], 3 * std)
            
            # Append 
            all_sessions.append(session)
            all_neurons.append(neuron)
            all_nan_counts.append(nan_count)
            all_onset_counts_std1.append(onset_std1)
            all_onset_counts_std2.append(onset_std2)
            all_onset_counts_std3.append(onset_std3)
    
    # 3. final df for this file
    df_summary = pd.DataFrame({
        'Session': all_sessions,
        'Neuron': all_neurons,
        'NaN Count': all_nan_counts,
        'Onset_Count_std1': all_onset_counts_std1,
        'Onset_Count_std2': all_onset_counts_std2,
        'Onset_Count_std3': all_onset_counts_std3
    })

    # Save as csv
    csv_filename = f"{number}_Onset_activity_of_neurons.csv"
    df_summary.to_csv(csv_filename, index=False)
    print(f"CSV file '{csv_filename}' has been created successfully.")

# Traverse the 'data/Batch_B' directory
batch_b_directory = os.path.join(os.getcwd(), 'data', 'Batch_B')
batch_a_directory = os.path.join(os.getcwd(), 'data', 'Batch_A')

""" Perform onset counting for batch B"""

# folder_pattern = r"Batch_B_2022_(\d+)_CFC_GPIO"
# pattern to match the subfolder names 
# (regular expression \d+ means 1 or more digits)

# for root, dirs, files in os.walk(batch_b_directory):
#     for dir_name in dirs:
#         match = re.match(folder_pattern, dir_name)
#         if match:
#             # Extract the number from the folder name
#             number = match.group(1)
#             # Construct the path to the Data_Miniscope_PP.mat file
#             mat_file_path = os.path.join(root, dir_name, 'Data_Miniscope_PP.mat')
#             if os.path.exists(mat_file_path):
#                 # Process the .mat file
#                 process_mat_file(mat_file_path, number)
#             else:
#                 print(f"Data_Miniscope_PP.mat not found in {dir_name}")



""" Perform onset counting for batch A"""
folder_pattern = r"Batch_A_2022_(\d+)_CFC_GPIO"

for root, dirs, files in os.walk(batch_a_directory):
    for dir_name in dirs:
        match = re.match(folder_pattern, dir_name)
        if match:
            # Extract the number from the folder name
            number = match.group(1)
            # Construct the path to the Data_Miniscope_PP.mat file
            mat_file_path = os.path.join(root, dir_name, 'Data_Miniscope_PP.mat')
            if os.path.exists(mat_file_path):
                # Process the .mat file
                process_mat_file(mat_file_path, number)
            else:
                print(f"Data_Miniscope_PP.mat not found in {dir_name}")


# %%


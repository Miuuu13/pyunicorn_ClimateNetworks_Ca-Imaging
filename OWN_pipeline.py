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
#rows, columns = df_start_frame_session.shape
# Number of rows: 309203
# Number of columns: 271
# Start Frame Sessions: 
#      0        1        2        3         4         5         6         7           8         9    
# 0  1.0  22361.0  44917.0  91943.0  124994.0  158591.0  191578.0  225081.0    243945.0  290516.0   


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

# s1_0-22360: (22361, 271) -h1 - 22361 rows
# s2_22361-44916: (22556, 271) -h2 - 22556 rows
# s3_44917-91942: (47026, 271) -cfc - 47026 rows
# s4_91943-124993: (33051, 271) - ex day 1 - 33051 rows
# s5_124994-158590: (33597, 271) - ex day 2 - 33597 rows
# s6_158591-191577: (32987, 271) - ex day 3 - 32987 rows
# s7_191578-225080: (33503, 271) - ex day 4  - 33503 rows
# s8_225081-243944: (18864, 271)- ex retrieval 1  - 18864 rows
# s9_243945-290515: (46571, 271) - renewal - 46571 rows
# s10_290516-309202: (18687, 271)- ex retrieval 2 - 18687 rows


# | Session | Start  | End    | Rows  | Columns | Experimental_Session | Time(s) | Time(min) |
# |---------|--------|--------|-------|---------|----------------------|---------|-----------|
# | s1      | 0      | 22360  | 22361 | 271     | h1                   | 1118.05 | 18.63     |
# | s2      | 22361  | 44916  | 22556 | 271     | h2                   | 1127.80 | 18.80     |
# | s3      | 44917  | 91942  | 47026 | 271     | cfc                  | 2351.30 | 39.19     |
# | s4      | 91943  | 124993 | 33051 | 271     | ex day 1             | 1652.55 | 27.54     |
# | s5      | 124994 | 158590 | 33597 | 271     | ex day 2             | 1679.85 | 28.00     |
# | s6      | 158591 | 191577 | 32987 | 271     | ex day 3             | 1649.35 | 27.49     |
# | s7      | 191578 | 225080 | 33503 | 271     | ex day 4             | 1675.15 | 27.92     |
# | s8      | 225081 | 243944 | 18864 | 271     | ex retrieval 1        | 943.20  | 15.72     |
# | s9      | 243945 | 290515 | 46571 | 271     | renewal              | 2328.55 | 38.81     |
# | s10     | 290516 | 309202 | 18687 | 271     | ex retrieval 2        | 934.35  | 15.57     |


print(c_raw_all_sessions)

#%%
"""Check for normalization"""


# Function to check if ΔF/F normalization or z-scoring was performed
def check_dff_and_zscoring(df):
    # Check if the data values are indicative of ΔF/F
    # ΔF/F values are usually close to zero, mostly positive with occasional small negative values
    dff_check = df.mean().min() >= -1 and df.mean().mean() > 0  
    # General def. of ΔF/F

    # Check if the data is z-scored (mean ~ 0 and std ~ 1 for each neuron)
    zscore_check = (df.mean().abs() < 0.1).all() and (df.std().between(0.9, 1.1)).all()

    if dff_check:
        print("The data appears to be ΔF/F normalized.")
    else:
        print("The data does not appear to be ΔF/F normalized.")
    
    if zscore_check:
        print("The data appears to be z-scored.")
    else:
        print("The data does not appear to be z-scored.")

# Apply the check on the raw data
check_dff_and_zscoring(df_c_raw_all)

# Compute the mean and standard deviation per neuron (per column)
neuron_means = df_c_raw_all.mean(axis=0)
neuron_stds = df_c_raw_all.std(axis=0)

# Display the results
mean_std_per_neuron = pd.DataFrame({
    'Neuron': df_c_raw_all.columns,
    'Mean': neuron_means,
    'Standard Deviation': neuron_stds
})


print("Mean and Standard Deviation per Neuron:")
print(mean_std_per_neuron)

#%%
def zscore_normalization(df):
    """Applies Z-score normalization to each column (neuron) of the DataFrame, handling NaN values properly."""
    
    # Define a lambda function to apply Z-score normalization column-wise, handling NaN values
    return df.apply(lambda col: (col - col.mean(skipna=True)) / col.std(skipna=True), axis=0)

# Dictionary to store Z-score normalized sessions
c_raw_all_sessions_zscore = {}

# Iterate over each session and apply Z-score normalization
for key, df in c_raw_all_sessions.items():
    print(f"Applying Z-score normalization to {key}")
    c_raw_all_sessions_zscore[key] = zscore_normalization(df)

# Check the content of the normalized data
for key, df in c_raw_all_sessions_zscore.items():
    print(f"{key}: {df.shape}")



#%% [3]
""" Apply Z-score normalization to C_raw_all per neuron and per session """

def zscore_normalization(df):
    """Applies Z-score normalization to each column (neuron) of the DataFrame, handling NaN values properly."""
    normalized_df = pd.DataFrame()  # Create an empty DataFrame to store normalized values
    
    # Iterate over each neuron (column)
    for col in df.columns:
        neuron_data = df[col]
        
        # Calculate mean and std, excluding NaN values
        mean = neuron_data.mean(skipna=True)
        std = neuron_data.std(skipna=True)
        
        # Apply Z-score normalization
        normalized_col = (neuron_data - mean) / std
        
        # Append the normalized column to the new DataFrame
        normalized_df[col] = normalized_col
    
    return normalized_df

# Dictionary to store Z-score normalized sessions
c_raw_all_sessions_zscore = {}

# Iterate over each session, apply Z-score normalization
for key, df in c_raw_all_sessions.items():
    print(f"Applying Z-score normalization to {key}")
    c_raw_all_sessions_zscore[key] = zscore_normalization(df)

# Check the content of the normalized data
for key, df in c_raw_all_sessions_zscore.items():
    print(f"{key}: {df.shape}")

#%%

#check_dff_and_zscoring(c_raw_all_sessions_zscore)

#%% [3]
""" Heatmap Plotting for all neurons
 Plot the heatmap for the entire df_c_raw_all of one id """


# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True) # plasma, inferno, magma
plt.title('Heatmap of df_c_raw_all')
plt.show()
#%%
#%% [4] # c_raw_all_sessions_zscore
""" Plot heatmap for the entire z-score normalized dataset """

# Concatenate all sessions into one large DataFrame
df_c_raw_all_zscore = pd.concat(c_raw_all_sessions_zscore.values(), axis=0)

# Plot heatmap for the entire z-score normalized dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all_zscore.T, cmap='viridis', cbar=True)  # Try different color maps like 'plasma', 'inferno', or 'magma'
plt.title('Heatmap of Z-score Normalized df_c_raw_all')
plt.xlabel('Time (Frames)')
plt.ylabel('Neurons')
plt.show()
#%% [5]
""" Plot heatmaps for each session z-score """

for key, df in c_raw_all_sessions_zscore.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of Z-score Normalized {key}')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Neurons')
    plt.show()


#%%
""" Heatmap Plotting for all neurons - tried differend color modes
 Plot the heatmap for the entire df_c_raw_all of one id """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='magma', cbar=True) # plasma, inferno, magma
plt.title('Heatmap of df_c_raw_all')
plt.show()

#%% [4]
""" Plot the heatmaps for each session window of one id """

for key, df in c_raw_all_sessions.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of {key}')
    plt.show()

#%% 
""" Make a cs plus dict """
""" Initialize and Calculate cs_plus values for each session """

c_raw_sessions_with_cs_plus = {}

# Only sessions that we are interested in (for example s4, s5, s6, s7)
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']

# Loop through selected sessions and calculate cs_plus values
for session_key in sessions_to_process:
    session_range = session_key.split('_')[1]
    start, end = map(int, session_range.split('-'))
    
    # Calculate cs_plus values
    cs_plus_1 = start + 3600 #180s
    cs_plus_2 = cs_plus_1 + 600 + 1200 #30s +60s
    cs_plus_3 = cs_plus_1 + 2 * (600 + 1200)
    cs_plus_4 = cs_plus_1 + 3 * (600 + 1200)
    
    # Store the values in the dictionary
    c_raw_sessions_with_cs_plus[session_key] = {
        'start': start,
        'end': end,
        'cs_plus_1': cs_plus_1,
        'cs_plus_2': cs_plus_2,
        'cs_plus_3': cs_plus_3,
        'cs_plus_4': cs_plus_4,
        'data': c_raw_all_sessions[session_key]
    }

# Output to verify
for session, values in c_raw_sessions_with_cs_plus.items():
    print(f"{session}: {values}")
#%% [5]

""" Heatmap Plotting for cs_plus Events

​"""

""" Plot heatmaps around each cs_plus tone for one session using C_Raw_all data """

# Define the session to process (e.g., s4_91943-124993)
session_key = 's4_91943-124993'  # You can change this to s5, s6, s7, etc.
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Get the range of start and end frames for the session from df_c_raw_all
# df_c_raw_all contains the actual data (time = rows, neurons = columns)
session_start, session_end = map(int, session_key.split('_')[1].split('-'))

# Loop through each cs_plus value to plot the heatmaps
for cs_plus_key in cs_plus_keys:
    cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]
    
    # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
    start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
    end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range
    
    # Extract the data for this window from df_c_raw_all
    partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)
    
    # Check if the partial data is non-empty
    if partial_data.empty:
        print(f"Warning: Empty data for {session_key} {cs_plus_key} (start: {start_idx}, end: {end_idx})")
        continue  # Skip this plot if there's no data

    # Plot the heatmap for the current cs_plus event
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap with both x (frames) and y (neurons) axis labels enabled
    sns.heatmap(partial_data.T, cmap='viridis', cbar=True, xticklabels=10, yticklabels=10)
    
    # Set the tick labels for frames and neurons
    plt.xticks(ticks=np.arange(0, partial_data.shape[0], step=100), labels=np.arange(start_idx, end_idx + 1, step=100))
    plt.yticks(ticks=np.arange(0, partial_data.shape[1], step=10), labels=np.arange(0, partial_data.shape[1], step=10))
    
    plt.title(f"Heatmap for {session_key} {cs_plus_key}: {start_idx}-{end_idx}")
    plt.xlabel('Frames (time)')
    plt.ylabel('Neurons')
    plt.show()


#%% [5]
""" Plot heatmaps around each cs_plus tone for s4, s5, s6, s7 """

""" Plot heatmaps around each cs_plus tone for all sessions using C_Raw_all data """

# Define the sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Loop through each session in sessions_to_process
for session_key in sessions_to_process:
    print(f"Processing {session_key}")
    
    # Get the session data from df_c_raw_all
    session_start, session_end = map(int, session_key.split('_')[1].split('-'))

    # Loop through each cs_plus value to plot the heatmaps
    for cs_plus_key in cs_plus_keys:
        cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]

        # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
        start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
        end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range

        # Extract the data for this window from df_c_raw_all
        partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)

        # Check if the partial data is non-empty
        if partial_data.empty:
            print(f"Warning: Empty data for {session_key} {cs_plus_key} (start: {start_idx}, end: {end_idx})")
            continue  # Skip this plot if there's no data

        # Plot the heatmap for the current cs_plus event
        plt.figure(figsize=(10, 8))

        # Plot the heatmap with both x (frames) and y (neurons) axis labels enabled
        sns.heatmap(partial_data.T, cmap='viridis', cbar=True, xticklabels=10, yticklabels=10)

        # Set the tick labels for frames and neurons
        plt.xticks(ticks=np.arange(0, partial_data.shape[0], step=100), labels=np.arange(start_idx, end_idx + 1, step=100))
        plt.yticks(ticks=np.arange(0, partial_data.shape[1], step=10), labels=np.arange(0, partial_data.shape[1], step=10))

        plt.title(f"Heatmap for {session_key} {cs_plus_key}: {start_idx}-{end_idx}")
        plt.xlabel('Frames (time)')
        plt.ylabel('Neurons')
        plt.show()

#%%
""" Count active frames """
active_frame_counts_per_session = {}

# Loop through each session in sessions_to_process
for session_key in sessions_to_process:
    print(f"Processing {session_key}")
    
    # Extract start and end frames from the session key
    session_start, session_end = map(int, session_key.split('_')[1].split('-'))
    
    # Extract the data for the current session from df_c_raw_all
    session_data = df_c_raw_all.iloc[session_start:session_end + 1, :]
    
    # Count the number of non-NaN frames for each neuron (each column in session_data)
    active_frame_counts = session_data.notna().sum(axis=0)
    
    # Store the results in the dictionary
    active_frame_counts_per_session[session_key] = active_frame_counts

    # Print the number of active frames per neuron for the current session
    print(f"Active frames per neuron in {session_key}:")
    print(active_frame_counts)

# Convert the dictionary to a DataFrame for easier analysis and display
active_frame_counts_df = pd.DataFrame(active_frame_counts_per_session)

# Display the resulting DataFrame
print("Active frame counts per neuron for sessions 4, 5, 6, and 7:")
print(active_frame_counts_df)

# Save the results to a CSV file for further analysis if needed
active_frame_counts_df.to_csv('Active_Frame_Counts_Per_Neuron_Sessions_4_5_6_7.csv', index_label='Neuron')
active_frame_counts_df.to_csv('Active_neurons_ex_days.csv', index_label='Neuron')


print("Results saved to 'Active_neurons_ex_days.csv'.")


#%% [5]
""" Plot heatmaps around each cs_plus tone for all sessions using C_Raw_all data 
    and add a red bar at the cs_plus frame """

# Define the sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Loop through each session in sessions_to_process
for session_key in sessions_to_process:
    print(f"Processing {session_key}")
    
    # Get the session data from df_c_raw_all
    session_start, session_end = map(int, session_key.split('_')[1].split('-'))

    # Loop through each cs_plus value to plot the heatmaps
    for cs_plus_key in cs_plus_keys:
        cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]

        # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
        start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
        end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range

        # Extract the data for this window from df_c_raw_all
        partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)

        # Check if the partial data is non-empty
        if partial_data.empty:
            print(f"Warning: Empty data for {session_key} {cs_plus_key} (start: {start_idx}, end: {end_idx})")
            continue  # Skip this plot if there's no data

        # Plot the heatmap for the current cs_plus event
        plt.figure(figsize=(100, 80))

        # Plot the heatmap with both x (frames) and y (neurons) axis labels enabled
        sns.heatmap(partial_data.T, cmap='viridis', cbar=True, xticklabels=10, yticklabels=10)

        # Add a red vertical line where the cs_plus event occurs
        cs_plus_relative = cs_plus_value - start_idx  # Calculate the relative position of cs_plus in the window
        plt.axvline(x=cs_plus_relative, color='red', linestyle='--', linewidth=2)  # Add a red vertical line at cs_plus

        # Set the tick labels for frames and neurons
        plt.xticks(ticks=np.arange(0, partial_data.shape[0], step=100), labels=np.arange(start_idx, end_idx + 1, step=100))
        plt.yticks(ticks=np.arange(0, partial_data.shape[1], step=10), labels=np.arange(0, partial_data.shape[1], step=10))

        plt.title(f"Heatmap for {session_key} {cs_plus_key}: {start_idx}-{end_idx}")
        plt.xlabel('Frames (time)')
        plt.ylabel('Neurons')
        plt.show()


#%%

"""Heamap without NaN"""
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Loop through each session in sessions_to_process
for session_key in sessions_to_process:
    print(f"Processing {session_key}")
    
    # Get the session data from df_c_raw_all
    session_start, session_end = map(int, session_key.split('_')[1].split('-'))

    # Loop through each cs_plus value to plot the heatmaps
    for cs_plus_key in cs_plus_keys:
        cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]

        # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
        start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
        end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range

        # Extract the data for this window from df_c_raw_all
        partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)

        # Remove rows where all values are NaN
        partial_data = partial_data.dropna(how='all')

        # Check if the partial data is non-empty after removing NaN rows
        if partial_data.empty:
            print(f"Warning: Empty data for {session_key} {cs_plus_key} (start: {start_idx}, end: {end_idx})")
            continue  # Skip this plot if there's no data

        # Plot the heatmap for the current cs_plus event
        plt.figure(figsize=(12, 8))

        # Plot the heatmap with both x (frames) and y (neurons) axis labels enabled
        sns.heatmap(partial_data.T, cmap='viridis', cbar=True, xticklabels=10, yticklabels=10)

        # Add a red vertical line where the cs_plus event occurs
        cs_plus_relative = cs_plus_value - start_idx  # Calculate the relative position of cs_plus in the window
        plt.axvline(x=cs_plus_relative, color='red', linestyle='--', linewidth=2)  # Add a red vertical line at cs_plus

        # Set the tick labels for frames and neurons
        plt.xticks(ticks=np.arange(0, partial_data.shape[0], step=100), labels=np.arange(start_idx, end_idx + 1, step=100))
        plt.yticks(ticks=np.arange(0, partial_data.shape[1], step=10), labels=np.arange(0, partial_data.shape[1], step=10))

        plt.title(f"Heatmap for {session_key} {cs_plus_key}: {start_idx}-{end_idx}")
        plt.xlabel('Frames (time)')
        plt.ylabel('Neurons')
        plt.show()

#%%

"""ACTIVITY"""

#Activity as not NaN

# sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
# cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# # Loop through each session in sessions_to_process
# for session_key in sessions_to_process:
#     print(f"Processing {session_key}")
    
#     # Get the session start and end frames from the session key
#     session_start, session_end = map(int, session_key.split('_')[1].split('-'))
    
#     # Extract the session data from df_c_raw_all for the given frames
#     session_data = df_c_raw_all.iloc[session_start:session_end + 1, :]

#     # Count active neurons (non-NaN values) per frame
#     active_neurons_per_frame = session_data.notna().sum(axis=1)  # Count non-NaN neurons per frame

#     # Get the cs_plus values for this session
#     cs_plus_values = [c_raw_sessions_with_cs_plus[session_key][cs_plus_key] for cs_plus_key in cs_plus_keys]

#     # Create the x-axis with the actual frame numbers from the session
#     frames = np.arange(session_start, session_end + 1)

#     # Plot the number of active neurons at each frame
#     plt.figure(figsize=(10, 6))
#     plt.plot(frames, active_neurons_per_frame, label="Active Neurons", color='blue')

#     # Mark the cs_plus events with red vertical lines
#     for cs_plus_value in cs_plus_values:
#         plt.axvline(x=cs_plus_value, color='red', linestyle='--', linewidth=2, label='cs_plus' if cs_plus_value == cs_plus_values[0] else "")

#     # Labeling the plot
#     plt.title(f"Number of Active Neurons per Frame for {session_key}")
#     plt.xlabel('Frames (Time)')
#     plt.ylabel('Number of Active Neurons')
#     plt.legend()
#     plt.tight_layout()

    # Show the plot
    plt.show()

#%% [5]
""" Plot subplots for the same cs_plus event across all sessions with red bars and showing every 10th frame number """

# Define the sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Loop through each cs_plus event
#%% [5]
#%% [5]
""" Plot subplots for the same cs_plus event across all sessions with red bars and less dense x-axis labels, vertically stacked """

# Define the sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Loop through each cs_plus event
for cs_plus_key in cs_plus_keys:
    
    # Create a figure with subplots (4 rows, 1 column) for each session's cs_plus heatmap
    fig, axes = plt.subplots(4, 1, figsize=(20, 24))  # 4 rows, 1 column for 4 sessions (increase width for a longer x-axis)

    # Loop through each session in sessions_to_process
    for idx, session_key in enumerate(sessions_to_process):
        print(f"Processing {session_key} for {cs_plus_key}")
        
        # Get the session data from df_c_raw_all
        session_start, session_end = map(int, session_key.split('_')[1].split('-'))

        # Get the cs_plus value for the current session
        cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]

        # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
        start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
        end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range

        # Extract the data for this window from df_c_raw_all
        partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)

        # Check if the partial data is non-empty
        if partial_data.empty:
            print(f"Warning: Empty data for {session_key} {cs_plus_key} (start: {start_idx}, end: {end_idx})")
            continue  # Skip this plot if there's no data

        # Plot the heatmap for the current session's cs_plus event in the corresponding subplot
        ax = axes[idx]  # Select the subplot for the current session
        sns.heatmap(partial_data.T, cmap='viridis', cbar=True, ax=ax, xticklabels=10, yticklabels=10)

        # Add a red vertical line where the cs_plus event occurs
        cs_plus_relative = cs_plus_value - start_idx  # Calculate the relative position of cs_plus in the window
        ax.axvline(x=cs_plus_relative, color='red', linestyle='--', linewidth=2)  # Add red vertical line at cs_plus

        # Set the tick labels for frames (displaying every 50th frame for less clutter)
        xticks = np.arange(0, partial_data.shape[0], step=50)  # Every 50th frame
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.arange(start_idx, end_idx + 1, step=50))  # Set labels for every 50th frame

        # Set the title and labels for the subplot
        ax.set_title(f"{session_key} {cs_plus_key}")
        ax.set_xlabel('Frames (time)')
        ax.set_ylabel('Neurons')

    # Adjust layout and show the plot for this cs_plus event across all sessions
    plt.tight_layout()
    plt.show()


#%%

""" subplots without nans"""

#%% [5]
""" Plot subplots for the same cs_plus event across all sessions with red bars, less dense x-axis labels, and removing NaN rows """

# Define the sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']


# Loop through each cs_plus event
for cs_plus_key in cs_plus_keys:
    
    # Create a figure with subplots (4 rows, 1 column) for each session's cs_plus heatmap
    fig, axes = plt.subplots(4, 1, figsize=(20, 24))  # 4 rows, 1 column for 4 sessions (increase width for a longer x-axis)

    # Loop through each session in sessions_to_process
    for idx, session_key in enumerate(sessions_to_process):
        print(f"Processing {session_key} for {cs_plus_key}")
        
        # Get the session data from df_c_raw_all
        session_start, session_end = map(int, session_key.split('_')[1].split('-'))

        # Get the cs_plus value for the current session
        cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]

        # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
        start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
        end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range

        # Extract the data for this window from df_c_raw_all
        partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)

        # Convert invalid values to NaN (in case they're not properly recognized as NaN)
        partial_data = partial_data.apply(pd.to_numeric, errors='coerce')

        # Remove rows where all values (neuron activity) are NaN
        partial_data_clean = partial_data.dropna(how='all')

        # Check if the cleaned partial data is non-empty
        if partial_data_clean.empty:
            print(f"Warning: Empty data for {session_key} {cs_plus_key} after removing NaNs (start: {start_idx}, end: {end_idx})")
            continue  # Skip this plot if there's no data

        # Plot the heatmap for the current session's cs_plus event in the corresponding subplot
        ax = axes[idx]  # Select the subplot for the current session
        sns.heatmap(partial_data_clean.T, cmap='viridis', cbar=True, ax=ax, xticklabels=10, yticklabels=10)

        # Add a red vertical line where the cs_plus event occurs (adjust for any dropped rows)
        cs_plus_relative = cs_plus_value - start_idx - (partial_data.shape[0] - partial_data_clean.shape[0])  # Adjust cs_plus position
        ax.axvline(x=cs_plus_relative, color='red', linestyle='--', linewidth=2)  # Add red vertical line at cs_plus

        # Set the tick labels for frames (displaying every 50th frame for less clutter)
        xticks = np.arange(0, partial_data_clean.shape[0], step=50)  # Every 50th frame
        ax.set_xticks(xticks)
        ax.set_xticklabels(np.arange(start_idx, start_idx + partial_data_clean.shape[0], step=50))  # Set labels for every 50th frame

        # Set the title and labels for the subplot
        ax.set_title(f"{session_key} {cs_plus_key}")
        ax.set_xlabel('Frames (time)')
        ax.set_ylabel('Neurons')

    # Adjust layout and show the plot for this cs_plus event across all sessions
    plt.tight_layout()
    plt.show()



###########
#%%
# Define the sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']
cs_plus_keys = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']

# Loop through each cs_plus event
for cs_plus_key in cs_plus_keys:
    
    # Create a figure with subplots (1 row, 4 columns) for each session's cs_plus heatmap
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # 1 row, 4 columns for 4 sessions

    # Loop through each session in sessions_to_process
    for idx, session_key in enumerate(sessions_to_process):
        print(f"Processing {session_key} for {cs_plus_key}")
        
        # Get the session data from df_c_raw_all
        session_start, session_end = map(int, session_key.split('_')[1].split('-'))

        # Get the cs_plus value for the current session
        cs_plus_value = c_raw_sessions_with_cs_plus[session_key][cs_plus_key]

        # Define the start and end window for this cs_plus (500 frames before, 600 frames after)
        start_idx = max(cs_plus_value - 500, 0)  # Ensure start is not less than 0
        end_idx = min(cs_plus_value + 600, df_c_raw_all.shape[0] - 1)  # Ensure end is within data range

        # Extract the data for this window from df_c_raw_all

        """ dropping nans not working """
        partial_data = df_c_raw_all.iloc[start_idx:end_idx + 1, :]  # Rows (frames) and all columns (neurons)

        # Check if the partial data is non-empty
        if partial_data.empty:
            print(f"Warning: Empty data for {session_key} {cs_plus_key} (start: {start_idx}, end: {end_idx})")
            continue  # Skip this plot if there's no data

        # Plot the heatmap for the current session's -> cs_plus  in the corresponding subplot
        ax = axes[idx]  # Select the subplot for the current session
        sns.heatmap(partial_data.T, cmap='inferno', cbar=True, ax=ax, xticklabels=10, yticklabels=10)

        # Add red vertical line where the cs_plus tone occurs
        cs_plus_relative = cs_plus_value - start_idx  # Calculate the relative position of cs_plus in the window
        ax.axvline(x=cs_plus_relative, color='red', linestyle='--', linewidth=2)  # Add red vertical line at cs_plus

        # Set the title and labels for the subplot
        ax.set_title(f"{session_key} {cs_plus_key}")
        ax.set_xlabel('Frames (time)')
        ax.set_ylabel('Neurons')

    # Adjust layout and show the plot for this cs_plus event across all sessions
    plt.tight_layout()
    plt.show()

############## END OF HEATMAP PLOTS ####################

#%%

""" COUNT NEURONS"""

# 1. Split the data into sessions and count NaNs and non-NaNs

# Generate sessions to process based on start_frame_session
sessions_to_process = []
for i in range(1, len(df_start_frame_session.columns)):  # Start from session 1
    start = int(df_start_frame_session.iloc[0, i-1])
    end = int(df_start_frame_session.iloc[0, i]) - 1
    session_key = f"s{i}_{start}-{end}"
    sessions_to_process.append((session_key, start, end))

# Add the final session (it goes until the last row of df_c_raw_all)
final_start = int(df_start_frame_session.iloc[0, -1])
final_end = len(df_c_raw_all) - 1
final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
sessions_to_process.append((final_session_key, final_start, final_end))

# Function to count NaN and non-NaN values in each session
def count_nan_non_nan(session_data):
    nan_count = session_data.isna().sum().sum()  # Total NaN values in the session
    non_nan_count = session_data.notna().sum().sum()  # Total non-NaN values in the session
    return nan_count, non_nan_count

# Loop through each session and count NaN and non-NaN values
for session_key, start, end in sessions_to_process:
    session_data = df_c_raw_all.iloc[start:end + 1, :]  # Extract session data based on frames
    nan_count, non_nan_count = count_nan_non_nan(session_data)
    
    # Print the results for each session
    print(f"Session {session_key}:")
    print(f"NaN count: {nan_count}, Non-NaN count: {non_nan_count}")
    print('-' * 50)

# 2. Calculate max, mean, std for each neuron and save to CSV

# Calculate statistics for each neuron (column)
neuron_stats_df = pd.DataFrame({
    'max': df_c_raw_all.max(),
    'mean': df_c_raw_all.mean(),
    'std': df_c_raw_all.std()
})

# Calculate the threshold as 80% of the max value for each neuron
neuron_stats_df['threshold'] = neuron_stats_df['max'] * 0.8

# Save the results to a CSV file
neuron_stats_df.to_csv('neuron_statistics.csv', index_label='Neuron')

# Output the first few rows of the statistics DataFrame to verify
print(neuron_stats_df.head())

#          max      mean       std  threshold
# 0  30.431952  0.994543  2.575916  24.345562
# 1  30.265529  0.860786  2.565490  24.212423
# 2  33.518725  0.662329  2.173652  26.814980
# 3  20.315921  0.856329  1.939148  16.252737
# 4  11.285561  0.173334  0.808306   9.028449

# # Function to count the number of NaN and non-NaN values in a session
# """ Count neurons 
# Before starting NW analysis"""
# # Function to count the number of NaN and non-NaN values for each neuron (column)
# def count_nan_non_nan(session_df):
#     # Count the total number of NaN values per neuron (column-wise)
#     nan_count = session_df.isna().sum().sum()  # Total NaNs in the session
    
#     # Count the total number of non-NaN values per neuron
#     non_nan_count = session_df.notna().sum().sum()  # Total non-NaNs in the session
    
#     print(f"Number of NaN values: {nan_count}")
#     print(f"Number of non-NaN values: {non_nan_count}")
    
#     return nan_count, non_nan_count

# # Function to check if a neuron crosses the threshold for at least 10 consecutive frames, at least 30 times
# def check_threshold_crossings(session_df):
#     neuron_count = 0  # Count of neurons that meet the criteria
    
#     # Loop through each neuron (column-wise) and calculate the threshold based on activity across frames (rows)
#     for neuron in session_df.columns:
#         # Calculate the threshold for the neuron (80% of the max value across all frames)
#         neuron_activity = session_df[neuron]
#         max_value = neuron_activity.max()
#         threshold = max_value * 0.8

#         # Identify where the threshold is crossed (True where crossed)
#         crossing_mask = neuron_activity > threshold

#         # Count consecutive crossings using a rolling window of 10 frames
#         consecutive_crossings = crossing_mask.rolling(window=10).sum() == 10  # Rolling sum = 10 means 10 consecutive frames
#         count_crossings = consecutive_crossings.sum()  # Count how many times the neuron crosses for at least 10 frames

#         # If threshold is crossed at least 30 times, count this neuron
#         if count_crossings >= 30:
#             neuron_count += 1

#     return neuron_count

# # Main Loop Through All Sessions
# for session_key, session_df in c_raw_all_sessions.items():
#     print(f"Processing session: {session_key}")
    
#     # Call the functions to count NaN and non-NaN values
#     nan_count, non_nan_count = count_nan_non_nan(session_df)

#     # Check how many neurons cross the threshold at least 30 times for 10 consecutive frames
#     neurons_meeting_criteria = check_threshold_crossings(session_df)
    
#     # Print results for the session
#     print(f"Session {session_key}:")
#     print(f"NaN count: {nan_count}, Non-NaN count: {non_nan_count}")
#     print(f"Neurons where threshold was crossed at least 30 times for 10 consecutive frames: {neurons_meeting_criteria}")
#     print('-' * 50)

#%%
"""the max, mean, std for eaach session for all neurons"""


# 1. Generate sessions to process based on start_frame_session
sessions_to_process = []
for i in range(1, len(df_start_frame_session.columns)):  # Start from session 1
    start = int(df_start_frame_session.iloc[0, i-1])
    end = int(df_start_frame_session.iloc[0, i]) - 1
    session_key = f"s{i}_{start}-{end}"
    sessions_to_process.append((session_key, start, end))

# Add the final session (it goes until the last row of df_c_raw_all)
final_start = int(df_start_frame_session.iloc[0, -1])
final_end = len(df_c_raw_all) - 1
final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
sessions_to_process.append((final_session_key, final_start, final_end))

# 2. Calculate max, mean, std for each neuron in each session and store results

# DataFrame to store session-based statistics (max, mean, std for all neurons)
session_stats_df = pd.DataFrame()

# Loop through each session to process
for session_key, start, end in sessions_to_process:
    session_data = df_c_raw_all.iloc[start:end + 1, :]  # Extract session data based on frames
    
    # Calculate the max, mean, std for each neuron (column) in this session
    session_max = session_data.max()
    session_mean = session_data.mean()
    session_std = session_data.std()
    
    # Create a DataFrame to store this session's statistics
    session_df = pd.DataFrame({
        'max': session_max,
        'mean': session_mean,
        'std': session_std
    })
    
    # Add the session key as a prefix to the column names
    session_df.columns = [f"{session_key}_{col}" for col in session_df.columns]
    
    # Concatenate the session's statistics to the overall DataFrame
    session_stats_df = pd.concat([session_stats_df, session_df], axis=1)

# 3. Save the results to a CSV file
session_stats_df.to_csv('session_neuron_statistics.csv', index_label='Neuron')

# Output the first few rows of the statistics DataFrame to verify
print(session_stats_df.head())

#%%
""" Add NaN, non-NaN info"""


# DataFrame to store session-based statistics (max, mean, std for all neurons)
session_stats_df = pd.DataFrame()

# DataFrame to store NaN and active neuron information
neuron_activity_stats = pd.DataFrame(columns=['session_key', 'neurons_all_nan', 'neurons_active'])

# Loop through each session to process
for session_key, start, end in sessions_to_process:
    session_data = df_c_raw_all.iloc[start:end + 1, :]  # Extract session data based on frames
    
    # Calculate the max, mean, std for each neuron (column) in this session
    session_max = session_data.max()
    session_mean = session_data.mean()
    session_std = session_data.std()

    # Count how many neurons have all NaN values
    neurons_all_nan = session_data.isna().all(axis=0).sum()
    
    # Count how many neurons are active at least once (have at least one non-NaN value)
    neurons_active = session_data.notna().any(axis=0).sum()

    # Store the neuron activity stats for this session
    new_row = pd.DataFrame({
        'session_key': [session_key],
        'neurons_all_nan': [neurons_all_nan],
        'neurons_active': [neurons_active]
    })
    
    # Concatenate the new row to the neuron_activity_stats DataFrame
    neuron_activity_stats = pd.concat([neuron_activity_stats, new_row], ignore_index=True)
    
    # Create a DataFrame to store this session's statistics
    session_df = pd.DataFrame({
        'max': session_max,
        'mean': session_mean,
        'std': session_std
    })
    
    # Add the session key as a prefix to the column names
    session_df.columns = [f"{session_key}_{col}" for col in session_df.columns]
    
    # Concatenate the session's statistics to the overall DataFrame
    session_stats_df = pd.concat([session_stats_df, session_df], axis=1)

# 3. Save the results to a CSV file
session_stats_df.to_csv('session_neuron_statistics.csv', index_label='Neuron')

# Save the neuron activity stats to a CSV file
neuron_activity_stats.to_csv('neuron_activity_statistics.csv', index=False)

# Output the first few rows of the statistics DataFrame to verify
print("Session Neuron Statistics")
print(session_stats_df.head())

# Output the neuron activity statistics DataFrame
print("\nNeuron Activity Statistics")
print(neuron_activity_stats)


#%%
""" Plot top active neurons around cs_plus"""

"""discard for now"""
# # Sessions we are interested in
# sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']

# # Define the number of frames before and after the cs_plus event
# frames_before = 500
# frames_after = 500
# time_window = frames_before + frames_after

# # Number of top neurons to plot
# top_neurons_count = 10

# # Load the Session Neuron Statistics (from the previous code where the threshold was calculated)
# session_stats_df = pd.read_csv('session_neuron_statistics.csv', index_col='Neuron')

# # Loop through each session
# for session_key in sessions_to_process:
#     print(f"Processing {session_key}")
    
#     # Get session data from c_raw_sessions_with_cs_plus
#     session_info = c_raw_sessions_with_cs_plus[session_key]
#     session_data = session_info['data']
    
#     # Extract relevant cs_plus times (cs_plus_1, cs_plus_2, etc.)
#     cs_plus_events = ['cs_plus_1', 'cs_plus_2', 'cs_plus_3', 'cs_plus_4']
    
#     # Extract the threshold for each neuron in this session
#     threshold_col = f'{session_key}_threshold'
#     neuron_thresholds = session_stats_df[threshold_col]

#     for cs_plus_key in cs_plus_events:
#         cs_plus_value = session_info[cs_plus_key]
        
#         # Define the start and end frame for the window around cs_plus
#         start_idx = max(cs_plus_value - frames_before, 0)
#         end_idx = min(cs_plus_value + frames_after, session_data.shape[0] - 1)

#         # Extract the data for this time window
#         window_data = session_data.iloc[start_idx:end_idx + 1, :]

#         # Count how many times each neuron crosses its threshold in this window
#         threshold_crossings = (window_data > neuron_thresholds).sum()

#         # Identify the top neurons that cross the threshold the most
#         top_neurons = threshold_crossings.nlargest(top_neurons_count).index

#         # Plot the activity of the top neurons across the time window
#         plt.figure(figsize=(10, 6))
#         for neuron in top_neurons:
#             plt.plot(window_data.index - cs_plus_value, window_data[neuron], label=f'Neuron {neuron}')
        
#         # Customize the plot
#         plt.axvline(x=0, color='red', linestyle='--', label='cs_plus event')
#         plt.title(f'{cs_plus_key} Activity of Top {top_neurons_count} Neurons Crossing Threshold - {session_key}')
#         plt.xlabel('Frames (relative to cs_plus)')
#         plt.ylabel('Neuronal Activity')
#         plt.legend()
#         plt.show()


#%%

""" Check mean, std """
# sessions
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']

# DataFrame to store the mean and std for each neuron
neuron_stats_summary = pd.DataFrame(columns=['session', 'neuron', 'mean', 'std'])

# Loop through each session
for session_key in sessions_to_process:
    print(f"Session in process: {session_key}")
    
    # Get session info from c_raw_sessions_with_cs_plus
    session_info = c_raw_sessions_with_cs_plus[session_key]
    session_data = session_info['data']
    
    # Calculate the mean and std for each neuron (column)
    session_means = session_data.mean()
    session_stds = session_data.std()

    # Create a DataFrame for the current session
    new_data = pd.DataFrame({
        'session': session_key,
        'neuron': session_data.columns,
        'mean': session_means.values,
        'std': session_stds.values
    })

    # Concatenate new data into neuron_stats_summary
    neuron_stats_summary = pd.concat([neuron_stats_summary, new_data], ignore_index=True)

# 1. Statistics
print(neuron_stats_summary.groupby('session')[['mean', 'std']].describe())

# 2. Visualization
plt.figure(figsize=(14, 6))
sns.boxplot(x='session', y='mean', data=neuron_stats_summary)
plt.title("Distribution of means across sessions")
plt.ylabel("Mean activity")
plt.xlabel("Session")
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(x='session', y='std', data=neuron_stats_summary)
plt.title("Distribution of std across sessions")
plt.ylabel("Std of activity")
plt.xlabel("Session")
plt.show()


############### END COUNTS #################
#%%

""" NW analysis """

# Step 1: Preprocess sessions to remove completely inactive neurons
def filter_active_neurons(df_session):
    """
    Remove neurons (columns) where all values are NaN (inactive neurons).
    """
    return df_session.dropna(axis=1, how='all')

# Step 2: Build a neuron network
def build_neuron_network(df_session, threshold=5, window_size=10):
    """
    Build a network of neurons where an edge is created if two neurons are co-active (non-NaN)
    for 10 consecutive frames at least 5 times.
    """
    G = nx.Graph()  # Initialize a graph for neurons

    # Get the total number of neurons
    neurons = df_session.columns

    # Loop through every pair of neurons
    for i, neuron_1 in enumerate(neurons):
        for j, neuron_2 in enumerate(neurons):
            if i >= j:
                # Avoid comparing the same neuron or redundant pairs
                continue

            # Get activity data for both neurons
            activity_1 = df_session[neuron_1].notna()  # True where active (non-NaN)
            activity_2 = df_session[neuron_2].notna()  # True where active (non-NaN)

            # Find windows of 10 consecutive frames where both neurons are active
            coactive_windows = (activity_1 & activity_2).rolling(window=window_size).sum() == window_size

            # Count the number of times neurons are co-active in a 10-frame window
            coactive_count = coactive_windows.sum()

            if coactive_count >= threshold:
                # Add an edge if the pair is coactive for at least 'threshold' times
                G.add_edge(neuron_1, neuron_2)

    return G

# Step 3: Perform analysis for each session
def perform_network_analysis(sessions_to_process, session_data_dict):
    """
    Perform network analysis for each session, generating a neuron network
    and outputting basic statistics for each session.
    """
    session_networks = {}

    for session_key in sessions_to_process:
        print(f"Processing session: {session_key}")
        
        # Get the data for this session and filter inactive neurons
        df_session = session_data_dict[session_key]
        df_active_neurons = filter_active_neurons(df_session)
        
        # Build the network for the session
        G = build_neuron_network(df_active_neurons)

        # Store the network for further analysis
        session_networks[session_key] = G
        
        # Output basic network statistics
        print(f"Session {session_key} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return session_networks

# Use the existing c_raw_all_sessions dictionary from your previous code
# Define the sessions to process
session_data_dict = {
    's4_91943-124993': c_raw_all_sessions['s4_91943-124993'],
    's5_124994-158590': c_raw_all_sessions['s5_124994-158590'],
    's6_158591-191577': c_raw_all_sessions['s6_158591-191577'],
    's7_191578-225080': c_raw_all_sessions['s7_191578-225080']
}

# Perform network analysis for each session
session_networks = perform_network_analysis(list(session_data_dict.keys()), session_data_dict)

# Example of further analysis: Degree centrality or other network measures
for session_key, G in session_networks.items():
    print(f"\n--- Network Analysis for {session_key} ---")
    
    # Degree centrality
    degree_centrality = nx.degree_centrality(G)
    top_neurons = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]  # Top 5 neurons
    
    print(f"Top 5 neurons by degree centrality in {session_key}:")
    for neuron, centrality in top_neurons:
        print(f"Neuron {neuron}: Degree centrality = {centrality:.4f}")
    
    # You can also analyze other metrics like clustering coefficient, etc.



#%%


# Function to plot the network for a given session
def plot_neuron_network(G, session_key):
    """
    Plot the neuron network using a spring layout.
    """
    plt.figure(figsize=(10, 10))  # Set the figure size
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Add labels (optional: this can clutter the plot if there are many neurons)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Set plot title
    plt.title(f"Neuron Network for {session_key}", fontsize=15)
    
    # Show the plot
    plt.show()

# Plot the network for each session
for session_key, G in session_networks.items():
    print(f"Plotting network for session: {session_key}")
    plot_neuron_network(G, session_key)



#%%

""" cs plus seperated session 4"""

# Function to build the neuron network for a given frame window
def build_neuron_network_for_window(df_session, window_start, window_end, threshold=10, window_size=10):
    """
    Build a network of neurons active between the given window (from window_start to window_end).
    Only neurons that are active (non-NaN) in the window are included.
    """
    # Extract the relevant window data
    df_window = df_session.iloc[window_start:window_end + 1, :]

    # Remove neurons that are all NaN in this window
    df_window = df_window.dropna(axis=1, how='all')

    G = nx.Graph()  # Initialize a graph for neurons

    # Get the total number of neurons
    neurons = df_window.columns

    # Loop through every pair of neurons
    for i, neuron_1 in enumerate(neurons):
        for j, neuron_2 in enumerate(neurons):
            if i >= j:
                # Avoid comparing the same neuron or redundant pairs
                continue

            # Get activity data for both neurons
            activity_1 = df_window[neuron_1].notna()  # True where active (non-NaN)
            activity_2 = df_window[neuron_2].notna()  # True where active (non-NaN)

            # Find windows of 10 consecutive frames where both neurons are active
            coactive_windows = (activity_1 & activity_2).rolling(window=window_size).sum() == window_size

            # Count the number of different windows where both neurons are co-active
            coactive_count = coactive_windows.sum()

            if coactive_count >= threshold:
                # Add an edge if the pair is coactive for at least 'threshold' windows
                G.add_edge(neuron_1, neuron_2)

    # Filter the network to only include nodes that have at least 1 connection (degree > 0)
    nodes_to_remove = [node for node in G if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)

    return G

# Function to plot the network
def plot_neuron_network(G, session_key, window_desc):
    """
    Plot the neuron network using a spring layout.
    """
    plt.figure(figsize=(10, 10))  # Set the figure size
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Add labels (optional: this can clutter the plot if there are many neurons)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Set plot title
    plt.title(f"Neuron Network for {session_key} - {window_desc}", fontsize=15)
    
    # Show the plot
    plt.show()

# Process only session 4 data
session_key = 's4_91943-124993'
session_data = c_raw_all_sessions[session_key]  # Extract data from the session

# Extract cs_plus information for this session from the dictionary
cs_plus_info = c_raw_sessions_with_cs_plus[session_key]

# Define the start and end points for each network
window_intervals = [
    (cs_plus_info['start'], cs_plus_info['cs_plus_1']),  # From start to cs_plus_1
    (cs_plus_info['cs_plus_1'], cs_plus_info['cs_plus_2']),  # From cs_plus_1 to cs_plus_2
    (cs_plus_info['cs_plus_2'], cs_plus_info['cs_plus_3']),  # From cs_plus_2 to cs_plus_3
    (cs_plus_info['cs_plus_3'], cs_plus_info['cs_plus_4']),  # From cs_plus_3 to cs_plus_4
    (cs_plus_info['cs_plus_4'], cs_plus_info['end'])  # From cs_plus_4 to the end
]

# Build and plot the networks for each window
for i, (window_start, window_end) in enumerate(window_intervals):
    print(f"Processing window {i+1}: Frames {window_start} to {window_end}")
    
    # Build the network for this window
    G = build_neuron_network_for_window(session_data, window_start, window_end)
    
    # Plot the network
    plot_neuron_network(G, session_key, f"Window {i+1}: {window_start} to {window_end}")



















































###############################################################
###############################################################
#Old version: wrong connection definiton:
##############################################################

""" 1. Network analysis per session - 
one session: which neurons are active simultaneously?

DENSE NW"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to perform network analysis and plot the network for a given session
def network_analysis_by_activity(session_df, activity_threshold=None, simultaneous_threshold=5):
    """
    Build a network of neurons based on simultaneous activity.
    
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows.
    - activity_threshold: The threshold for a neuron to be considered "active". If None, use mean activity.
    - simultaneous_threshold: Minimum number of frames where two neurons need to be simultaneously active to create an edge.
    """
    # If no activity threshold is provided, use the mean activity level of each neuron
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean()  # Average activity across all neurons
    
    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)
    
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # If the number of simultaneous active frames exceeds the threshold, add an edge
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Plot the network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Layout for positioning the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

    plt.title("Neural Network Based on Simultaneous Activity")
    plt.show()

    return G  # Return the graph object for further analysis if needed

#%%

import matplotlib.pyplot as plt
import networkx as nx

# Step 2: Build a denser neuron network based on the activity condition
def build_dense_neuron_network(df_session, threshold=5, window_size=10):
    """
    Build a dense network of neurons where an edge is created if two neurons are co-active (non-NaN)
    for 10 consecutive frames at least 5 times. Only include neurons that satisfy this condition.
    """
    G = nx.Graph()  # Initialize a graph for neurons

    # Get the total number of neurons
    neurons = df_session.columns

    # Loop through every pair of neurons
    for i, neuron_1 in enumerate(neurons):
        for j, neuron_2 in enumerate(neurons):
            if i >= j:
                # Avoid comparing the same neuron or redundant pairs
                continue

            # Get activity data for both neurons
            activity_1 = df_session[neuron_1].notna()  # True where active (non-NaN)
            activity_2 = df_session[neuron_2].notna()  # True where active (non-NaN)

            # Find windows of 10 consecutive frames where both neurons are active
            coactive_windows = (activity_1 & activity_2).rolling(window=window_size).sum() == window_size

            # Count the number of times neurons are co-active in a 10-frame window
            coactive_count = coactive_windows.sum()

            if coactive_count >= threshold:
                # Add an edge if the pair is coactive for at least 'threshold' times
                G.add_edge(neuron_1, neuron_2)

    # Filter the network to only include nodes that have at least 1 connection (degree > 0)
    nodes_to_remove = [node for node in G if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    return G

# Step 4: Plot the network
def plot_neuron_network(G, session_key):
    """
    Plot the neuron network using a spring layout.
    """
    plt.figure(figsize=(10, 10))  # Set the figure size
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Add labels (optional: this can clutter the plot if there are many neurons)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Set plot title
    plt.title(f"Neuron Network for {session_key} (Filtered)", fontsize=15)
    
    # Show the plot
    plt.show()

# Step 5: Perform network analysis for each session
def perform_dense_network_analysis(sessions_to_process, session_data_dict):
    session_networks = {}

    for session_key in sessions_to_process:
        print(f"Processing session: {session_key}")
        
        # Get the data for this session and filter inactive neurons
        df_session = session_data_dict[session_key]
        df_active_neurons = filter_active_neurons(df_session)
        
        # Build the denser network for the session
        G = build_dense_neuron_network(df_active_neurons)

        # Store the network for further analysis
        session_networks[session_key] = G
        
        # Output basic network statistics
        print(f"Session {session_key} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return session_networks

# Example of sessions to process and their data
session_data_dict = {
    's4_91943-124993': c_raw_all_sessions['s4_91943-124993'],
    's5_124994-158590': c_raw_all_sessions['s5_124994-158590'],
    's6_158591-191577': c_raw_all_sessions['s6_158591-191577'],
    's7_191578-225080': c_raw_all_sessions['s7_191578-225080']
}

# Perform dense network analysis for each session
dense_session_networks = perform_dense_network_analysis(list(session_data_dict.keys()), session_data_dict)

# Plot each network
for session_key, G in dense_session_networks.items():
    print(f"Plotting network for session: {session_key}")
    plot_neuron_network(G, session_key)


#%%

""" 10 times in 10 frame window """


# Step 2: Build a denser neuron network based on the new condition (10 different windows of 10 frames)
def build_dense_neuron_network(df_session, threshold=10, window_size=10):
    """
    Build a dense network of neurons where an edge is created if two neurons are co-active (non-NaN)
    for 10 consecutive frames at least 10 times (new condition). Only include neurons that satisfy this condition.
    """
    G = nx.Graph()  # Initialize a graph for neurons

    # Get the total number of neurons
    neurons = df_session.columns

    # Loop through every pair of neurons
    for i, neuron_1 in enumerate(neurons):
        for j, neuron_2 in enumerate(neurons):
            if i >= j:
                # Avoid comparing the same neuron or redundant pairs
                continue

            # Get activity data for both neurons
            activity_1 = df_session[neuron_1].notna()  # True where active (non-NaN)
            activity_2 = df_session[neuron_2].notna()  # True where active (non-NaN)

            # Find windows of 10 consecutive frames where both neurons are active
            coactive_windows = (activity_1 & activity_2).rolling(window=window_size).sum() == window_size

            # Count the number of times neurons are co-active in a 10-frame window
            coactive_count = coactive_windows.sum()

            if coactive_count >= threshold:
                # Add an edge if the pair is coactive for at least 'threshold' times
                G.add_edge(neuron_1, neuron_2)

    # Filter the network to only include nodes that have at least 1 connection (degree > 0)
    nodes_to_remove = [node for node in G if G.degree(node) == 0]
    G.remove_nodes_from(nodes_to_remove)
    
    return G

# Step 4: Plot the network
def plot_neuron_network(G, session_key):
    """
    Plot the neuron network using a spring layout.
    """
    plt.figure(figsize=(10, 10))  # Set the figure size
    
    # Use spring layout for visualization
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue')
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    # Add labels (optional: this can clutter the plot if there are many neurons)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Set plot title
    plt.title(f"Neuron Network for {session_key} (Filtered)", fontsize=15)
    
    # Show the plot
    plt.show()

# Step 5: Perform dense network analysis for each session
def perform_dense_network_analysis(sessions_to_process, session_data_dict):
    session_networks = {}

    for session_key in sessions_to_process:
        print(f"Processing session: {session_key}")
        
        # Get the data for this session and filter inactive neurons
        df_session = session_data_dict[session_key]
        df_active_neurons = filter_active_neurons(df_session)
        
        # Build the denser network for the session
        G = build_dense_neuron_network(df_active_neurons)

        # Store the network for further analysis
        session_networks[session_key] = G
        
        # Output basic network statistics
        print(f"Session {session_key} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return session_networks

# Example of sessions to process and their data
session_data_dict = {
    's4_91943-124993': c_raw_all_sessions['s4_91943-124993'],
    's5_124994-158590': c_raw_all_sessions['s5_124994-158590'],
    's6_158591-191577': c_raw_all_sessions['s6_158591-191577'],
    's7_191578-225080': c_raw_all_sessions['s7_191578-225080']
}

# Perform dense network analysis for each session
dense_session_networks = perform_dense_network_analysis(list(session_data_dict.keys()), session_data_dict)

# Plot each network
for session_key, G in dense_session_networks.items():
    print(f"Plotting network for session: {session_key}")
    plot_neuron_network(G, session_key)

# Example of sessions to process and the

####################### NETWORK ANALYSIS #############################




# %%
""" Test on one session """
session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]

# Perform network analysis based on neuron activity
G = network_analysis_by_activity(session_df)

# %%
# Loop through all sessions in the dictionary and apply network analysis
for session_key, session_df in c_raw_all_sessions.items():
    print(f"Performing network analysis for {session_key}...")
    G = network_analysis_by_activity(session_df)
# %%
""" 2. Network analysis per session - 
one session: which neurons are active simultaneously?

SPARSE NW"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to perform network analysis and plot the network for a given session
def network_analysis_sparse(session_df, activity_threshold=None, simultaneous_threshold=10):
    """
    Build a sparse network of neurons based on strong simultaneous activity, excluding inactive neurons.
    
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows.
    - activity_threshold: The threshold for a neuron to be considered "active". If None, use a higher threshold based on mean + std.
    - simultaneous_threshold: Minimum number of frames where two neurons need to be simultaneously active to create an edge.
    """
    # Filter out neurons that have all NaN values (inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')

    # If no activity threshold is provided, use a higher threshold: mean activity + 1 standard deviation
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()  # Average activity + standard deviation
    
    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)
    
    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # Only add an edge if the number of simultaneous active frames exceeds a higher threshold
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Plot the sparse network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Layout for positioning the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

    plt.title("Sparse Neural Network Based on Strong Simultaneous Activity (Active Neurons Only)")
    plt.show()

    return G  # Return the graph object for further analysis if needed

#%%
# Apply the sparse network analysis to a single session
session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]

# Perform sparse network analysis based on strong neuron activity
G = network_analysis_sparse(session_df)

#%%
# Loop through all sessions in the dictionary and apply sparse network analysis
for session_key, session_df in c_raw_all_sessions.items():
    print(f"Performing sparse network analysis for {session_key}...")
    G = network_analysis_sparse(session_df)

#%%

""" Display the neurons with the most connections (try sparser nw, add frame time)"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to perform network analysis, plot the network, and display neurons with most connections
def network_analysis_sparse(session_df, activity_threshold=None, simultaneous_threshold=10, top_n=5):
    """

    Build a sparser network of neurons based on strong simultaneous activity, excluding inactive neurons.
    
    Parameters:
    - session_df: 
    DataFrame with neurons as columns and time frames as rows
    - activity_threshold: 
    The threshold for a neuron to be considered "active". 
    If None, use a higher threshold based on mean + std -?

    - simultaneous_threshold: 
    Minimum number of frames where two neurons need to be simultaneously active to create an edge
    Use 10 to start

    - top_n: 
    Number of top neurons to display based on the number of connections (degree)

    """
    # Filter out neurons that have all NaN values (-> discaard completely inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')
    #how='all': discard only if no NaN

    # If no activity threshold is provided, 
    # #use a higher threshold: mean activity + 1 std?
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()  
        # Average activity + std
    
    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)
    #get 1 for true else 0 for false, if T not reached
    
    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # Only add an edge if the number of simultaneous active frames exceeds a higher threshold
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Plot the sparse network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Layout for positioning the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=50)

    # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

    plt.title("Sparse Neural Network Based on Strong Simultaneous Activity (Active Neurons Only)")
    plt.show()

    # Calculate the degree (number of connections) for each neuron
    degrees = dict(G.degree())

    # Sort neurons by degree in descending order
    sorted_neurons_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)

    # Display the top N neurons with the most connections
    print(f"\nTop {top_n} neurons with the most connections (degree):")
    for neuron, degree in sorted_neurons_by_degree[:top_n]:
        print(f"Neuron: {neuron}, Connections: {degree}")

    return G  # Return the graph object for further analysis if needed
#%%
# Apply the sparse network analysis to a single session and display top 5 most connected neurons
session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]

# Perform sparse network analysis and show top 5 neurons with the most connections
G = network_analysis_sparse(session_df, top_n=5)

# %%

# Top 5 neurons with the most connections (degree):
# Neuron: 40, Connections: 116
# Neuron: 5, Connections: 114
# Neuron: 15, Connections: 114
# Neuron: 20, Connections: 114
# Neuron: 80, Connections: 113

#Neuron 40 having 116 connections means that 
# Neuron 40 is simultaneously active with 116 other neurons 
# in at least the number of frames specified by the simultaneous_threshold (e.g., 10 frames).


#%%

""" Reduce number of neurons using degree threshold"""

def network_analysis_sparse(session_df, activity_threshold=None, simultaneous_threshold=10, top_n=5, max_neurons=30):
    """
    Build a sparser network of neurons based on strong simultaneous activity, excluding inactive neurons.
    
    see better version above
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows
    - activity_threshold: Threshold for a neuron to be considered "active". If None, use mean + std.
    - simultaneous_threshold: Minimum number of frames for two neurons to be simultaneously active to create an edge.
    - top_n: Number of top neurons to display based on the number of connections (degree).
    - max_neurons: Maximum number of neurons to include in the network visualization.
    """
    # Filter out neurons that have all NaN values (discard completely inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')

    # If no activity threshold is provided, use mean activity + 1 std
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()

    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # Only add an edge if the number of simultaneous active frames exceeds the threshold
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Calculate the degree (number of connections) for each neuron
    degrees = dict(G.degree())

    # Sort neurons by degree in descending order
    sorted_neurons_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)

    # Limit the network to the top N best-connected neurons
    top_neurons = [neuron for neuron, degree in sorted_neurons_by_degree[:max_neurons]]
    
    # Create a subgraph containing only the top neurons
    subgraph = G.subgraph(top_neurons)

    # Plot the sparse network for the best-connected neurons
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
    edges = subgraph.edges(data=True)
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

    plt.title(f"Sparse Neural Network: Top {max_neurons} Best-Connected Neurons")
    plt.show()

    # Display the top N neurons with the most connections
    print(f"\nTop {top_n} neurons with the most connections (degree):")
    for neuron, degree in sorted_neurons_by_degree[:top_n]:
        print(f"Neuron: {neuron}, Connections: {degree}")

    return subgraph  # Return the subgraph for further analysis if needed

#%%
    # Apply the sparse network analysis to a single session and display top 5 most connected neurons
session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]

# Perform sparse network analysis and show top 5 neurons with the most connections
G = network_analysis_sparse(session_df, top_n=5)

# %%
""" Use another approach: only neurons with at least X connections"""

def network_analysis_sparse(session_df, activity_threshold=None, simultaneous_threshold=10, top_n=5, degree_threshold=5):
    """
    Build a sparser network of neurons based on strong simultaneous activity, excluding inactive neurons.
    
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows
    - activity_threshold: Threshold for a neuron to be considered "active". If None, use mean + std.
    - simultaneous_threshold: Minimum number of frames for two neurons to be simultaneously active to create an edge.
    - top_n: Number of top neurons to display based on the number of connections (degree).
    - degree_threshold: Only include neurons with degree greater than or equal to this threshold.
    """
    # Filter out neurons that have all NaN values (discard completely inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')

    # If no activity threshold is provided, use mean activity + 1 std
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()

    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # Only add an edge if the number of simultaneous active frames exceeds the threshold
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Calculate the degree (number of connections) for each neuron
    degrees = dict(G.degree())

    # Filter neurons by the degree threshold
    top_neurons = [neuron for neuron, degree in degrees.items() if degree >= degree_threshold]

    # Create a subgraph containing only the top neurons based on the degree threshold
    subgraph = G.subgraph(top_neurons)

    # Plot the sparse network for the best-connected neurons
    plt.figure(figsize=(100, 80))
    pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
    edges = subgraph.edges(data=True)
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

    plt.title(f"Sparse Neural Network: Neurons with Degree ≥ {degree_threshold}")
    plt.show()

    # Display the top N neurons with the most connections
    sorted_neurons_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    print(f"\nTop {top_n} neurons with the most connections (degree):")
    for neuron, degree in sorted_neurons_by_degree[:top_n]:
        print(f"Neuron: {neuron}, Connections: {degree}")

    return subgraph  # Return the subgraph for further analysis if needed

# %%

session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]

# Perform sparse network analysis and show top 100 neurons with the most connections
G = network_analysis_sparse(session_df, top_n=20)
# %%
for session_key, session_df in c_raw_all_sessions.items():
    print(f"Performing sparse network analysis for {session_key}...")
    G = network_analysis_sparse(session_df)

# %%

""" reduce to top neurons"""
# def network_analysis_sparse_top(session_df, activity_threshold=None, simultaneous_threshold=10, top_n=5, degree_threshold=5):
#     """
#     Build a sparser network of neurons based on strong simultaneous activity, excluding inactive neurons.
    
#     Parameters:
#     - session_df: DataFrame with neurons as columns and time frames as rows
#     - activity_threshold: Threshold for a neuron to be considered "active". If None, use mean + std.
#     - simultaneous_threshold: Minimum number of frames for two neurons to be simultaneously active to create an edge.
#     - top_n: Number of top neurons to display based on the number of connections (degree).
#     - degree_threshold: Only include neurons with degree greater than or equal to this threshold.
#     """
#     # Filter out neurons that have all NaN values (discard completely inactive neurons)
#     session_df = session_df.dropna(axis=1, how='all')

#     # If no activity threshold is provided, use mean activity + 1 std
#     if activity_threshold is None:
#         activity_threshold = session_df.mean().mean() + session_df.std().mean()

#     # Create a binary matrix indicating whether each neuron is "active" in each frame
#     binary_activity_matrix = (session_df > activity_threshold).astype(int)

#     # Create the NetworkX graph
#     G = nx.Graph()

#     # Add nodes (one for each neuron, represented by a column in the DataFrame)
#     for neuron in session_df.columns:
#         G.add_node(neuron)

#     # Add edges based on simultaneous activity
#     num_neurons = session_df.shape[1]
    
#     for i in range(num_neurons):
#         for j in range(i + 1, num_neurons):
#             # Count how many frames both neurons are active simultaneously
#             simultaneous_active_frames = np.sum(
#                 (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
#             )
            
#             # Only add an edge if the number of simultaneous active frames exceeds the threshold
#             if simultaneous_active_frames >= simultaneous_threshold:
#                 G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

#     # Calculate the degree (number of connections) for each neuron
#     degrees = dict(G.degree())

#     # Sort neurons by degree and get the top N neurons
#     sorted_neurons_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
#     top_neurons = [neuron for neuron, degree in sorted_neurons_by_degree[:top_n]]

#     # Create a subgraph containing only the top N neurons
#     subgraph = G.subgraph(top_neurons)

#     # Plot the sparse network for the top N neurons
#     plt.figure(figsize=(100, 80))
#     pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes
#     nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

#     # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
#     edges = subgraph.edges(data=True)
#     nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

#     plt.title(f"Sparse Neural Network: Top {top_n} Neurons")
#     plt.show()

#     # Display the top N neurons with the most connections
#     print(f"\nTop {top_n} neurons with the most connections (degree):")
#     for neuron, degree in sorted_neurons_by_degree[:top_n]:
#         print(f"Neuron: {neuron}, Connections: {degree}")

#     return subgraph  # Return the subgraph for further analysis if needed

# # Perform sparse network analysis for a specific session and focus on the top 20 neurons
# session_key = 's1_0-22485'
# session_df = c_raw_all_sessions[session_key]
# G = network_analysis_sparse_top(session_df, top_n=20)

# %%
def network_analysis_sparse_top(session_df, activity_threshold=None, simultaneous_threshold=10, top_n=20, degree_threshold=5):
     # Filter out neurons that have all NaN values (discard completely inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')

    # If no activity threshold is provided, use mean activity + 1 std
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()

    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int) 

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # Only add an edge if the number of simultaneous active frames exceeds the threshold
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Calculate the degree (number of connections) for each neuron
    degrees = dict(G.degree())

    # Sort neurons by degree and get the top N neurons
    sorted_neurons_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    top_neurons = [neuron for neuron, degree in sorted_neurons_by_degree[:top_n]]

    # Create a subgraph containing only the top N neurons
    subgraph = G.subgraph(top_neurons)

    # Plot the sparse network for the top N neurons
    plt.figure(figsize=(12, 12))  # Adjusted figure size for better visualization
    pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes

    # Increase the node size and reduce the edge thickness
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', 
            node_size=1500, font_size=12)  # Increased node size

    # Draw edges with varying thickness 
    edges = subgraph.edges(data=True)
    #nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=[d['weight'] * 0.02 for (u, v, d) in edges])  # Reduced edge thickness
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=[d['weight'] * 0.02 for (u, v, d) in edges])

    plt.title(f"Sparse Neural Network: Top {top_n} Neurons")
    plt.show()

    # Display the top N neurons
    print(f"\nTop {top_n} neurons with the most connections (degree):")
    for neuron, degree in sorted_neurons_by_degree[:top_n]:
        print(f"Neuron: {neuron}, Connections: {degree}")

    return subgraph  # Return the subgraph for further analysis if needed

#
session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]
G = network_analysis_sparse_top(session_df, top_n=20)

# %%
for session_key, session_df in c_raw_all_sessions.items():
    print(f"Performing sparse network analysis for {session_key}...")
    G = network_analysis_sparse_top(session_df)
# %%
# %%
""" Use another approach: only neurons with at least X connections"""
# %% [1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# %% [2]
# %% [1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def network_analysis_sparse(session_df, activity_threshold=None, simultaneous_threshold=10, top_n=20, degree_threshold=5):
    """
    Build a sparser network of neurons based on strong simultaneous activity, excluding inactive neurons.
    
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows
    - activity_threshold: Threshold for a neuron to be considered "active". If None, use mean + std.
    - simultaneous_threshold: Minimum number of frames for two neurons to be simultaneously active to create an edge.
    - top_n: Number of top neurons to display based on the number of connections (degree).
    - degree_threshold: Only include neurons with degree greater than or equal to this threshold.
    """
    # Filter out neurons that have all NaN values (discard completely inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')

    # If no activity threshold is provided, use mean activity + 1 std
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()

    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on simultaneous activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Count how many frames both neurons are active simultaneously
            simultaneous_active_frames = np.sum(
                (binary_activity_matrix.iloc[:, i] == 1) & (binary_activity_matrix.iloc[:, j] == 1)
            )
            
            # Only add an edge if the number of simultaneous active frames exceeds the threshold
            if simultaneous_active_frames >= simultaneous_threshold:
                G.add_edge(session_df.columns[i], session_df.columns[j], weight=simultaneous_active_frames)

    # Calculate the degree (number of connections) for each neuron
    degrees = dict(G.degree())

    # Filter neurons by the degree threshold
    top_neurons = [neuron for neuron, degree in degrees.items() if degree >= degree_threshold]

    # Create a subgraph containing only the top neurons based on the degree threshold
    subgraph = G.subgraph(top_neurons)

    # Plot the sparse network for the best-connected neurons
    plt.figure(figsize=(12, 12))  # Adjust figure size if needed
    pos = nx.spring_layout(subgraph)  # Layout for positioning the nodes
    nx.draw(subgraph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Draw edges with varying thickness based on the number of simultaneous activations (edge weight)
    edges = subgraph.edges(data=True)
    nx.draw_networkx_edges(subgraph, pos, edgelist=edges, width=[d['weight'] * 0.1 for (u, v, d) in edges])

    plt.title(f"Sparse Neural Network: Neurons with Degree ≥ {degree_threshold}")
    plt.show()

    # Display the top N neurons with the most connections
    sorted_neurons_by_degree = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    print(f"\nTop {top_n} neurons with the most connections (degree):")
    for neuron, degree in sorted_neurons_by_degree[:top_n]:
        print(f"Neuron: {neuron}, Connections: {degree}")

    return subgraph  # Return the subgraph for further analysis if needed


# Loop through all the sessions and build networks
for session_key, session_df in c_raw_all_sessions_zscore.items():
    print(f"Performing sparse network analysis for {session_key}...")
    G = network_analysis_sparse(session_df)



# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# %% [1]
def consecutive_active_frames(binary_series, consecutive_count=10, min_occurrences=4):
    """
    Check how many times a neuron is active for at least 'consecutive_count' consecutive frames.
    
    Parameters:
    - binary_series: Binary series (0 or 1) indicating whether a neuron is active at each frame.
    - consecutive_count: The minimum number of consecutive frames for the neuron to be considered "active".
    - min_occurrences: The minimum number of times the neuron needs to be active for the given consecutive frames.
    
    Returns:
    - Boolean indicating whether the neuron was active for at least 'consecutive_count' frames at least 'min_occurrences' times.
    """
    # Find runs of consecutive active frames
    active_runs = (binary_series != 0).astype(int).groupby(binary_series.eq(0).cumsum()).cumsum()
    
    # Count how many times the neuron is active for at least 'consecutive_count' consecutive frames
    consecutive_active_occurrences = (active_runs >= consecutive_count).sum()
    
    # Return True if the number of occurrences meets or exceeds the minimum occurrences
    return consecutive_active_occurrences >= min_occurrences

# %% [2]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# %% [1]
def consecutive_active_frames(binary_series, consecutive_count=10, min_occurrences=4):
    """
    Check how many times a neuron is active for at least 'consecutive_count' consecutive frames.
    
    Parameters:
    - binary_series: Binary series (0 or 1) indicating whether a neuron is active at each frame.
    - consecutive_count: The minimum number of consecutive frames for the neuron to be considered "active".
    - min_occurrences: The minimum number of times the neuron needs to be active for the given consecutive frames.
    
    Returns:
    - Boolean indicating whether the neuron was active for at least 'consecutive_count' frames at least 'min_occurrences' times.
    """
    # Find runs of consecutive active frames
    active_runs = (binary_series != 0).astype(int).groupby(binary_series.eq(0).cumsum()).cumsum()

    # Count how many times the neuron is active for at least 'consecutive_count' consecutive frames
    consecutive_active_occurrences = (active_runs >= consecutive_count).sum()

    # Return True if the number of occurrences meets or exceeds the minimum occurrences
    return consecutive_active_occurrences >= min_occurrences

# %% [2]
def network_analysis_consecutive_activity(session_df, activity_threshold=None, consecutive_count=10, min_occurrences=4):
    """
    Build a network based on neurons being active for at least `consecutive_count` frames, at least `min_occurrences` times.
    
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows.
    - activity_threshold: Threshold for a neuron to be considered "active". If None, use mean + std.
    - consecutive_count: Minimum number of consecutive frames a neuron must be active for.
    - min_occurrences: Minimum number of times the neuron must meet the consecutive active frame requirement.
    """
    # Filter out neurons that have all NaN values (discard completely inactive neurons)
    session_df = session_df.dropna(axis=1, how='all')

    # If no activity threshold is provided, use mean activity + 1 std
    if activity_threshold is None:
        activity_threshold = session_df.mean().mean() + session_df.std().mean()

    # Create a binary matrix indicating whether each neuron is "active" in each frame
    binary_activity_matrix = (session_df > activity_threshold).astype(int)

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on consecutive activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Check if both neurons meet the consecutive activity condition
            neuron_i_active = consecutive_active_frames(binary_activity_matrix.iloc[:, i], consecutive_count, min_occurrences)
            neuron_j_active = consecutive_active_frames(binary_activity_matrix.iloc[:, j], consecutive_count, min_occurrences)
            
            # If both neurons meet the condition, add an edge between them
            if neuron_i_active and neuron_j_active:
                G.add_edge(session_df.columns[i], session_df.columns[j])

    # Plot the sparse network for the neurons that meet the condition
    plt.figure(figsize=(12, 12))  # Adjust figure size if needed
    pos = nx.spring_layout(G)  # Layout for positioning the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    plt.title(f"Neural Network: Neurons active for at least {consecutive_count} frames, {min_occurrences} times")
    plt.show()

    # Display the neurons that were connected
    print(f"Neurons connected based on {consecutive_count} consecutive active frames, {min_occurrences} times:")
    for edge in G.edges():
        print(f"Connected neurons: {edge[0]} <--> {edge[1]}")

    return G  # Return the graph for further analysis if needed

for session_key, session_df in df_c_raw_all_zscore.items():
    print(f"Performing network analysis based on consecutive activity for {session_key}...")
    # Perform network analysis for the current session
    G = network_analysis_consecutive_activity(session_df, consecutive_count=10, min_occurrences=4)


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

def consecutive_active_frames(binary_series, consecutive_count=10, min_occurrences=4):
    """
    Check how many times a neuron is active for at least 'consecutive_count' consecutive frames.
    
    Parameters:
    - binary_series: Binary series (0 or 1) indicating whether a neuron is active at each frame.
    - consecutive_count: The minimum number of consecutive frames for the neuron to be considered "active".
    - min_occurrences: The minimum number of times the neuron needs to be active for the given consecutive frames.
    
    Returns:
    - Boolean indicating whether the neuron was active for at least 'consecutive_count' frames at least 'min_occurrences' times.
    """
    # Find runs of consecutive active frames
    active_runs = (binary_series != 0).astype(int).groupby(binary_series.eq(0).cumsum()).cumsum()

    # Count how many times the neuron is active for at least 'consecutive_count' consecutive frames
    consecutive_active_occurrences = (active_runs >= consecutive_count).sum()

    # Return True if the number of occurrences meets or exceeds the minimum occurrences
    return consecutive_active_occurrences >= min_occurrences

# %% [2]
# def network_analysis_consecutive_activity(session_df, activity_threshold=None, consecutive_count=10, min_occurrences=4):
#     """
#     Build a network based on neurons being active for at least `consecutive_count` frames, at least `min_occurrences` times.
    
#     Parameters:
#     - session_df: DataFrame with neurons as columns and time frames as rows.
#     - activity_threshold: Threshold for a neuron to be considered "active". If None, use mean + std.
#     - consecutive_count: Minimum number of consecutive frames a neuron must be active for.
#     - min_occurrences: Minimum number of times the neuron must meet the consecutive active frame requirement.
#     """
#     # Ensure the input is always a DataFrame (even if it contains a single column)
#     session_df = pd.DataFrame(session_df)

#     # Filter out neurons that have all NaN values (discard completely inactive neurons)
#     session_df = session_df.dropna(axis=1, how='all')

#     # If no activity threshold is provided, use mean activity + 1 std
#     if activity_threshold is None:
#         activity_threshold = session_df.mean().mean() + session_df.std().mean()

#     # Create a binary matrix indicating whether each neuron is "active" in each frame
#     binary_activity_matrix = (session_df > activity_threshold).astype(int)

#     # Create the NetworkX graph
#     G = nx.Graph()

#     # Add nodes (one for each neuron, represented by a column in the DataFrame)
#     for neuron in session_df.columns:
#         G.add_node(neuron)

#     # Add edges based on consecutive activity
#     num_neurons = session_df.shape[1]
    
#     for i in range(num_neurons):
#         for j in range(i + 1, num_neurons):
#             # Check if both neurons meet the consecutive activity condition
#             neuron_i_active = consecutive_active_frames(binary_activity_matrix.iloc[:, i], consecutive_count, min_occurrences)
#             neuron_j_active = consecutive_active_frames(binary_activity_matrix.iloc[:, j], consecutive_count, min_occurrences)
            
#             # If both neurons meet the condition, add an edge between them
#             if neuron_i_active and neuron_j_active:
#                 G.add_edge(session_df.columns[i], session_df.columns[j])

#     # Plot the sparse network for the neurons that meet the condition
#     plt.figure(figsize=(12, 12))  # Adjust figure size if needed
#     pos = nx.spring_layout(G)  # Layout for positioning the nodes
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

#     plt.title(f"Neural Network: Neurons active for at least {consecutive_count} frames, {min_occurrences} times")
#     plt.show()

#     # Display the neurons that were connected
#     print(f"Neurons connected based on {consecutive_count} consecutive active frames, {min_occurrences} times:")
#     for edge in G.edges():
#         print(f"Connected neurons: {edge[0]} <--> {edge[1]}")

#     return G  # Return the graph for further analysis if needed

# # Loop through all session keys in df_c_raw_all_zscore and apply network analysis
# for session_key, session_df in df_c_raw_all_zscore.items():
#     print(f"Performing network analysis based on consecutive activity for {session_key}...")
#     # Perform network analysis for the current session
#     G = network_analysis_consecutive_activity(session_df, consecutive_count=10, min_occurrences=4)



# %% [1] Define the helper function to check for consecutive active frames
def consecutive_active_frames(binary_series, consecutive_count=10, min_occurrences=4):
    """
    Check how many times a neuron is active for at least 'consecutive_count' consecutive frames.
    
    Parameters:
    - binary_series: Binary series (0 or 1) indicating whether a neuron is active at each frame.
    - consecutive_count: The minimum number of consecutive frames for the neuron to be considered "active".
    - min_occurrences: The minimum number of times the neuron needs to be active for the given consecutive frames.
    
    Returns:
    - Boolean indicating whether the neuron was active for at least 'consecutive_count' frames at least 'min_occurrences' times.
    """
    # Find runs of consecutive active frames
    active_runs = (binary_series != 0).astype(int).groupby(binary_series.eq(0).cumsum()).cumsum()

    # Count how many times the neuron is active for at least 'consecutive_count' consecutive frames
    consecutive_active_occurrences = (active_runs >= consecutive_count).sum()

    # Return True if the number of occurrences meets or exceeds the minimum occurrences
    return consecutive_active_occurrences >= min_occurrences

# %% [2] Preprocess data for each session
def preprocess_data(session_df, threshold=0.8):
    """
    Preprocess the data by removing rows that are completely NaN and apply a threshold for activity.
    
    Parameters:
    - session_df: DataFrame containing neuron activity (neurons are columns, time points are rows).
    - threshold: The activity threshold to filter the data. Only values above this threshold are considered active.
    
    Returns:
    - Preprocessed DataFrame with NaN rows removed and activity threshold applied.
    """
    # Remove rows that are completely NaN
    session_df = session_df.dropna(how='all')

    # Apply the threshold for neuron activity (values above threshold are active)
    threshold_df = (session_df >= threshold).astype(int)

    return threshold_df

# %% [3] Build a network graph based on neuron activity
def network_analysis_consecutive_activity(session_df, consecutive_count=10, min_occurrences=4):
    """
    Build a network based on neurons being active for at least `consecutive_count` frames, at least `min_occurrences` times.
    
    Parameters:
    - session_df: DataFrame with neurons as columns and time frames as rows (binary activity after threshold).
    - consecutive_count: Minimum number of consecutive frames a neuron must be active for.
    - min_occurrences: Minimum number of times the neuron must meet the consecutive active frame requirement.
    
    Returns:
    - NetworkX graph object representing the neural network.
    """
    # Ensure the input is always a DataFrame (even if it contains a single column)
    session_df = pd.DataFrame(session_df)

    # Create the NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for neuron in session_df.columns:
        G.add_node(neuron)

    # Add edges based on consecutive activity
    num_neurons = session_df.shape[1]
    
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):
            # Check if both neurons meet the consecutive activity condition
            neuron_i_active = consecutive_active_frames(session_df.iloc[:, i], consecutive_count, min_occurrences)
            neuron_j_active = consecutive_active_frames(session_df.iloc[:, j], consecutive_count, min_occurrences)
            
            # If both neurons meet the condition, add an edge between them
            if neuron_i_active and neuron_j_active:
                G.add_edge(session_df.columns[i], session_df.columns[j])

    return G  # Return the graph for further analysis and plotting

# %% [4] Plot the network
def plot_network(G, title="Neural Network: Connections based on consecutive activity"):
    plt.figure(figsize=(12, 12))  # Set figure size
    pos = nx.spring_layout(G)  # Layout for positioning the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Show the plot with the title
    plt.title(title)
    plt.show()

# %% [5] Full analysis and plotting pipeline
def analyze_and_plot(session_df, session_key, consecutive_count=10, min_occurrences=4, threshold=0.8):
    print(f"Performing network analysis for session: {session_key}...")
    
    # Preprocess the data by applying the threshold and removing NaN rows
    preprocessed_df = preprocess_data(session_df, threshold=threshold)
    
    # Perform network analysis
    G = network_analysis_consecutive_activity(preprocessed_df, consecutive_count, min_occurrences)

    # Plot the resulting network
    plot_network(G, title=f"Neural Network for {session_key}: Consecutive Activity Connections")

# %% [6] Perform analysis for all sessions in `c_raw_all_sessions`
print("Analyzing `c_raw_all_sessions`...\n")
for session_key, session_df in c_raw_all_sessions.items():
    analyze_and_plot(session_df, session_key, consecutive_count=10, min_occurrences=4, threshold=0.8)

# %% [7] Perform analysis for all sessions in `c_raw_all_sessions_zscore`
print("Analyzing `c_raw_all_sessions_zscore`...\n")
for session_key, session_df in c_raw_all_sessions_zscore.items():
    analyze_and_plot(session_df, session_key, consecutive_count=10, min_occurrences=4, threshold=0.8)

# %%
def print_node_degrees(G):
    """
    Print the degrees (number of connections) for each node in the graph.
    """
    degrees = G.degree()
    for node, degree in degrees:
        print(f"Neuron {node} has {degree} connections.")

# Example usage after creating the graph:
G = network_analysis_consecutive_activity(preprocessed_df, consecutive_count, min_occurrences)
print_node_degrees(G)
# %%
pos = nx.circular_layout(G)  # Alternative layout
# %%

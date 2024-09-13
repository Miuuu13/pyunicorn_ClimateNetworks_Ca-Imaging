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

#%% [1]
""" 1. Data Access and Preprocessing """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis

# Define the path to the .mat or HDF5 file 
# file format trivially, stick to .mat as this is the original

path_1012 = "/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/Batch_B/Batch_B_2022_1012_CFC_GPIO/Batch_B_2022_1012_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_1012, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# Convert the data  for 'C_Raw_all' and start frames to a pandas DataFrame
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

# Initialize a list to store the session keys
sessions_to_process = []

# Loop through the start_frame_session to generate session keys
for i in range(1, len(df_start_frame_session.columns)):  # Start from session 1 (skip the first one)
    start = int(df_start_frame_session.iloc[0, i-1])
    end = int(df_start_frame_session.iloc[0, i]) - 1
    
    # Create a session key like 's4_91943-124993'
    session_key = f"s{i}_{start}-{end}"
    sessions_to_process.append(session_key)

# Add the final session (it goes until the last row of df_c_raw_all)
final_start = int(df_start_frame_session.iloc[0, -1])
final_end = len(df_c_raw_all) - 1
final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
sessions_to_process.append(final_session_key)

# Output to verify the generated sessions
print("Generated sessions_to_process:")
print(sessions_to_process)

# The output should look something like this:
# ['s1_0-22360', 's2_22361-44916', ..., 's7_191578-225080']

#%% [2]
""" Create windows and store them in a dictionary """

# Initialize the dictionary to store the windows
c_raw_all_sessions = {}

# Loop through each start frame session to create the windows
for i in range(len(df_start_frame_session.columns)):
    if i == 0:
        # First window starts at index 0
        start = 0
    else:
        # Subsequent windows start at the current session index
        start = int(df_start_frame_session.iloc[0, i])

    # If not the last index, take the next start and subtract 1
    if i < len(df_start_frame_session.columns) - 1:
        end = int(df_start_frame_session.iloc[0, i+1]) - 1
    else:
        # Last window ends at the last row of df_c_raw_all
        end = rows - 1
    
    # Create a key like 's1_0-22485', 's2_22486-44647', etc.
    key = f"s{i+1}_{start}-{end}"
    
    # Store the corresponding rows in the dictionary
    c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

# Output to verify the dictionary content
for key, df in c_raw_all_sessions.items():
    print(f"{key}: {df.shape}")

# s1_0-22360: (22361, 271) -h1
# s2_22361-44916: (22556, 271) -h2
# s3_44917-91942: (47026, 271) -cfc
# s4_91943-124993: (33051, 271) - ex day 1
# s5_124994-158590: (33597, 271) - ex day 2
# s6_158591-191577: (32987, 271) - ex day 3
# s7_191578-225080: (33503, 271) - ex day 4 
# s8_225081-243944: (18864, 271)- ex retrieval 1  
# s9_243945-290515: (46571, 271) - renewal
# s10_290516-309202: (18687, 271)- ex retrieval 2

#%% [3]
""" Heatmap Plotting for all neurons
 Plot the heatmap for the entire df_c_raw_all of one id """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True) # plasma, inferno, magma
plt.title('Heatmap of df_c_raw_all')
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
    cs_plus_1 = start + 5400
    cs_plus_2 = cs_plus_1 + 900 + 1800
    cs_plus_3 = cs_plus_1 + 2 * (900 + 1800)
    cs_plus_4 = cs_plus_1 + 3 * (900 + 1800)
    
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

take Generated_sessions_to_process and make dict, containing frames for cs_plus

HEATMAPS of frames before and after tone is played
# s4_91943-124993: (33051, 271) - ex day 1
# s5_124994-158590: (33597, 271) - ex day 2
# s6_158591-191577: (32987, 271) - ex day 3
# s7_191578-225080: (33503, 271) - ex day 4 
For all sessions: H(180 s) + 4*90 s(CS+, pause)+ 12*90 s(CS-, pause)
In total 1620 s (32.400 frames for 20 fps)

180s = 5400f habituation
-> then 4 * (30 s CS+, 60 s pause)

5400 f + 4* (900 f, then 1800f pause), now not interested in frames with CS- 

for ex4: 91943f + 600f until next 1000f

​"""




#%% [5]
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
        plt.figure(figsize=(10, 8))

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

############### END COUNTS #################


""" NW analysis """

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

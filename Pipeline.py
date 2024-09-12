#%% 
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% [1]
""" Data Access and Preprocessing """

path_1012 = "/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/Batch_B/Batch_B_2022_1012_CFC_GPIO/Batch_B_2022_1012_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_1012, 'r') as h5_file:
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

df_c_raw_all = pd.DataFrame(data_c_raw_all)
df_start_frame_session = pd.DataFrame(start_frame_session)

# Verify data structure
print(f"C_Raw_all: \n{df_c_raw_all.head()}")
print(f"\nNumber of rows: {df_c_raw_all.shape[0]}")
print(f"Number of columns: {df_c_raw_all.shape[1]}")
print(f"Start Frame Sessions: \n{df_start_frame_session.head()}")

#%% [2]
""" Session Key Generation """

sessions_to_process = []

# Loop through the start_frame_session to generate session keys
for i in range(1, len(df_start_frame_session.columns)):  
    start = int(df_start_frame_session.iloc[0, i-1])
    end = int(df_start_frame_session.iloc[0, i]) - 1
    session_key = f"s{i}_{start}-{end}"
    sessions_to_process.append(session_key)

final_start = int(df_start_frame_session.iloc[0, -1])
final_end = len(df_c_raw_all) - 1
final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
sessions_to_process.append(final_session_key)

print("Generated sessions_to_process:")
print(sessions_to_process)

#%% [3]
""" Create windows and store them in a dictionary """

c_raw_all_sessions = {}
for i in range(len(df_start_frame_session.columns)):
    if i == 0:
        start = 0
    else:
        start = int(df_start_frame_session.iloc[0, i])

    if i < len(df_start_frame_session.columns) - 1:
        end = int(df_start_frame_session.iloc[0, i+1]) - 1
    else:
        end = df_c_raw_all.shape[0] - 1
    
    key = f"s{i+1}_{start}-{end}"
    c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

#%% [4]
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
    cs_plus_22 = cs_plus_1 + 900 + 1800
    cs_plus_3 = cs_plus_1 + 2 * (900 + 1800)
    cs_plus_4 = cs_plus_1 + 3 * (900 + 1800)
    
    # Store the values in the dictionary
    c_raw_sessions_with_cs_plus[session_key] = {
        'start': start,
        'end': end,
        'cs_plus_1': cs_plus_1,
        'cs_plus_22': cs_plus_22,
        'cs_plus_3': cs_plus_3,
        'cs_plus_4': cs_plus_4,
        'data': c_raw_all_sessions[session_key]
    }

# Output to verify
for session, values in c_raw_sessions_with_cs_plus.items():
    print(f"{session}: {values}")

#%% [5]
""" Heatmap Plotting for cs_plus Events """

for session_key in sessions_to_process:
    session_info = c_raw_sessions_with_cs_plus[session_key]
    session_df = session_info['data']
    
    # Extract cs_plus values and plot frames
    cs_plus_1 = session_info['cs_plus_1']
    cs_plus_22 = session_info['cs_plus_22']
    cs_plus_3 = session_info['cs_plus_3']
    cs_plus_4 = session_info['cs_plus_4']
    
    data_cs_plus_1 = session_df.iloc[cs_plus_1-100:cs_plus_22]
    data_cs_plus_22 = session_df.iloc[cs_plus_22-100:cs_plus_3]
    data_cs_plus_3 = session_df.iloc[cs_plus_3-100:cs_plus_4]
    data_cs_plus_4 = session_df.iloc[cs_plus_4-100:]
    
    combined_data = pd.concat([data_cs_plus_1, data_cs_plus_22, data_cs_plus_3, data_cs_plus_4])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_data, cmap="viridis")
    plt.title(f"Heatmap for {session_key}")
    plt.xlabel("Neurons")
    plt.ylabel("Frames")
    plt.show()




########################




#%%
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% [1]
""" Access data """

# Define the path to the .mat or HDF5 file 
path_1012 = "/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/Batch_B/Batch_B_2022_1012_CFC_GPIO/Batch_B_2022_1012_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_1012, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# Convert the data for 'C_Raw_all' and start frames to a pandas DataFrame
df_c_raw_all = pd.DataFrame(data_c_raw_all)
df_start_frame_session = pd.DataFrame(start_frame_session)

# Header, number of rows/columns of the DataFrame
rows, columns = df_c_raw_all.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
print(f"Start Frame Sessions: \n{df_start_frame_session.head()}")

#%% [2]
""" Generate sessions_to_process list automatically """

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

#%% [3]
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

#%% [4]
""" Add cs_plus calculations to each session and plot heatmaps for s4, s5, s6, s7 """

# Initialize a dictionary to store cs_plus calculations and DataFrame
c_raw_sessions_with_cs_plus = {}

# Define the sessions to process (s4, s5, s6, s7)
sessions_to_plot = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']

# Loop through the selected sessions
for session_key in sessions_to_plot:
    # Access the DataFrame and session key
    session_df = c_raw_all_sessions[session_key]
    
    # Parse the start and end from the session key
    session_range = session_key.split('_')[1]
    start, end = map(int, session_range.split('-'))
    
    # Calculate cs_plus values based on the given rules
    cs_plus_1 = start + 5400
    cs_plus_22 = cs_plus_1 + 900 + 1800
    cs_plus_3 = cs_plus_1 + 2 * (900 + 1800)
    cs_plus_4 = cs_plus_1 + 3 * (900 + 1800)
    
    # Store the values in the dictionary for this session
    c_raw_sessions_with_cs_plus[session_key] = {
        'start': start,
        'end': end,
        'cs_plus_1': cs_plus_1,
        'cs_plus_22': cs_plus_22,
        'cs_plus_3': cs_plus_3,
        'cs_plus_4': cs_plus_4,
        'data': session_df  # Store the original session data (DataFrame)
    }

    # Extract data around cs_plus events and plot heatmap
    data_cs_plus_1 = session_df.iloc[cs_plus_1-100:cs_plus_22]
    data_cs_plus_22 = session_df.iloc[cs_plus_22-100:cs_plus_3]
    data_cs_plus_3 = session_df.iloc[cs_plus_3-100:cs_plus_4]
    data_cs_plus_4 = session_df.iloc[cs_plus_4-100:]
    
    # Concatenate all data for the heatmap
    combined_data = pd.concat([data_cs_plus_1, data_cs_plus_22, data_cs_plus_3, data_cs_plus_4])
    
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_data.T, cmap="viridis")
    plt.title(f"Heatmap for {session_key}")
    plt.xlabel("Neurons")
    plt.ylabel("Frames")
    plt.show()

# Output the cs_plus calculations for verification
for session, values in c_raw_sessions_with_cs_plus.items():
    print(f"{session}: start={values['start']}, end={values['end']}, "
          f"cs_plus_1={values['cs_plus_1']}, cs_plus_22={values['cs_plus_22']}, "
          f"cs_plus_3={values['cs_plus_3']}, cs_plus_4={values['cs_plus_4']}")


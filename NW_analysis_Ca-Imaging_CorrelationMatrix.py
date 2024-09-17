"""

Please try this to build your network:
Forget about the events, and use the following code to construct a network:

# The following uses the variable activityis, which is a 2D numpy array that  contains the luminance profile, first index is the neuron number, second index is the frame. You can use the luminance profile that is z-score normalized over the session, but it should give the same result with the original data, so for a first try you can directly use the original data.

# What is done in the following: Get a correlation matrix, turn correlation matrix into connectivity map
import numpy as np

# first calculate correlation coefficient between all pre-selected neurons (first use all neurons, then calculate some network measures, see how they develop over extinction days, then select subset and see how this changes)

corr_mat = np.corrcoef(activity[selectedNeurons, :])    #replace selectedNeurons by : to analyse all neurons. This is heavy computing.

# ignore self-connections by zeroing diagonal elements (since self-correlation is always 1)
np.fill_diagonal(corr_mat, 0)

# set threshold for correlation, so we don't take into account weak connections
corr_th = 0.7       # play around with this a bit to see how strongly the correlation threshold changes the number of neurons in the network, but 0.7 sounds reasonable, work with that for the beginning
# zero all connections below threshold
corr_mat[corr_mat < corr_th] = 0

# create graph from correlation matrix
G = nx.from_numpy_array(corr_mat)   # G is a network where still all neurons are part of the network, only that some neurons are not connected as we used a thresholding on the connection strength, i.e. the correlation strength. These unconnected neurons change the network measures, but keep them for the beginning.


# Display using simple drawing. Fix seed to have the same result every time we run the code
nx.draw(G, pos=nx.spring_layout(G, seed=12345678), node_size=10, width=0.3)

"""

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

path_1012 = r"C:\Users\manue\Desktop\NW_Analysis_Ca-Imaging\data\Batch_B\Batch_B_2022_1012_CFC_GPIO\Batch_B_2022_1012_CFC_GPIO\Data_Miniscope_PP.mat"

# Open the .mat file (load mat does not work for this .mat version) 
# and extract C_Raw_all data (matrix)
with h5py.File(path_1012, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    
# Convert the data  for 'C_Raw_all' and start frames to pandas DataFrames
df_c_raw_all = pd.DataFrame(data_c_raw_all)

#%%
    
with h5py.File(path_1012, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# Convert the data  for 'C_Raw_all' and start frames to pandas DataFrames
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
    
#%%
    
""" Use a correlation matrix for getting network """

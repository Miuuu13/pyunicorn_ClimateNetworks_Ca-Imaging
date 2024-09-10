#%%
# import os
import h5py
import pandas as pd


#Dynamic Network: 
# Created by splitting data into time windows 
# and computing correlations for each window.
#Neuron Activation Sequence: 
# By monitoring degree centrality over time, 
# you can track when neurons become "activated."
#Visualization: 
# Networks at each time window show evolving 
# connections between neurons, 
# and degree centrality plots highlight when specific neurons activate

# Define the path to the HDF5 file 
path_936 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_A/Batch_A_2022_936_CFC_GPIO/Data_Miniscope_PP.h5"

# Open the HDF5 file and extract the required data
with h5py.File(path_936, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]

# Convert the data  for 'C_Raw_all' to a pandas DataFrame
df_c_raw_all = pd.DataFrame(data_c_raw_all)

# header, nr of rows/columns of the DataFrame
print(df_c_raw_all.head())
rows, columns = df_c_raw_all.shape
print(f"\nNumber of rows: {rows}")
print(f"Number of columns: {columns}")

#%%

""" Set Window for frames """
#30 fps - 1800frames are 60s, try later different windows
# Define the window size (e.g., 1800 frames)
window_size = 1800

# Number of windows to analyze
num_windows = df_c_raw_all.shape[0] // window_size

# Create a list of DataFrames, 
# where each one is corresponding to a time window
time_windows = [df_c_raw_all.iloc[i * window_size : (i + 1) * window_size] for i in range(num_windows)]

#%%

adjacency_matrices = []

# Iterate over each time window and compute correlation matrix
for window in time_windows:
    # Fill missing values (for example, with zero)
    window_filled = window.fillna(0)
    
    # Compute correlation matrix for the window
    correlation_matrix = window_filled.corr()
    
    # Apply a threshold to create an adjacency matrix (e.g., correlations > 0.5)
    threshold = 0.7 #0.5 set at beginning, was recommended
    #THRESHOLD
    #lower threshold would create a denser nw, weaker connections

    # higher threshold would create a sparse nw, stronger correlations, 
    #but could exclude connections that are weak but meaningful

    adjacency_matrix = (correlation_matrix > threshold).astype(int)
    
    adjacency_matrices.append(adjacency_matrix)

#%%

import networkx as nx
import matplotlib.pyplot as plt

# Create dynamic networks for each window
for idx, adjacency_matrix in enumerate(adjacency_matrices):
    G = nx.from_numpy_array(adjacency_matrix.values)
    mapping = {i: col for i, col in enumerate(df_c_raw_all.columns)}
    G = nx.relabel_nodes(G, mapping)
    
    # Plot the network for each window (see window size at beginning)
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=False, node_size=20, node_color='blue', edge_color='gray', alpha=0.6)
    plt.title(f"Neuronal Network at Window {idx + 1}")
    plt.show()

#%%


""" Does not work: """
# Track degree centrality over time for each neuron
degree_centralities = pd.DataFrame(index=df_c_raw_all.columns)

for idx, adjacency_matrix in enumerate(adjacency_matrices):
    G = nx.from_numpy_array(adjacency_matrix.values)
    mapping = {i: col for i, col in enumerate(df_c_raw_all.columns)}
    G = nx.relabel_nodes(G, mapping)
    
    # Calculate degree centrality for the current window see set frames at beginning
    degree_centrality = nx.degree_centrality(G)
    
    # Add the degree centrality for this window to the DataFrame
    degree_centralities[f"Window_{idx + 1}"] = pd.Series(degree_centrality)

# Display degree centralities over time (frames, window)
import ace_tools as tools; tools.display_dataframe_to_user(name="Degree Centralities Over Time", dataframe=degree_centralities)

#%%
plt.figure(figsize=(10, 6))

# Plot degree centrality for a few example neurons
for neuron in df_c_raw_all.columns[:5]:  
    # Plot first 5 neurons as an example

    plt.plot(degree_centralities.index, degree_centralities.loc[neuron], label=f'Neuron {neuron}')

plt.xlabel('Time Window')
plt.ylabel('Degree Centrality')
plt.title('Neuronal Activation Over Time')
plt.legend()
plt.show()


#%%

""" Set window according to start frame session"""
import h5py
import pandas as pd

# Path to the HDF5 file
path_936 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_A/Batch_A_2022_936_CFC_GPIO/Data_Miniscope_PP.h5"

# Open the HDF5 file
with h5py.File(path_936, 'r') as h5_file:
    # Extract the 'Start_Frame_Session' data (assuming it's in the 'Data' group)
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

    # Extract the 'C_raw_all' data (neuron activity data)
    c_raw_all_data = h5_file['Data']['C_Raw_all'][:]

# Convert the data to a DataFrame
df_c_raw_all = pd.DataFrame(c_raw_all_data)

# Convert 'Start_Frame_Session' to a list of frame indices
start_frame_session = list(start_frame_session)

#%%


# You now have a list `sessions` where each element is a DataFrame corresponding to a session.


#%%


""" Set windows acc. to session starts"""
#not working


#%%
import h5py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

path_936 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_A/Batch_A_2022_936_CFC_GPIO/Data_Miniscope_PP.h5"


with h5py.File(path_936, 'r') as h5_file:
    # Extract the 'Start_Frame_Session' data (in the 'Data' group)
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

    # Extract the 'C_raw_all' data (neuron activity)
    c_raw_all_data = h5_file['Data']['C_Raw_all'][:]

# Convert the data to a DataFrame
df_c_raw_all = pd.DataFrame(c_raw_all_data)

# Convert 'Start_Frame_Session' to a list of frame indices
start_frame_session = list(start_frame_session)

#%%

# Set window according to session starts
sessions = []

# Split the data into sessions based on Start_Frame_Session
for i in range(len(start_frame_session) - 1):
    start_frame = start_frame_session[i]
    end_frame = start_frame_session[i + 1] - 1
    
    # Slice the data for the current session
    session_data = df_c_raw_all.iloc[start_frame:end_frame]
    
    # Append the session data to the list of sessions
    sessions.append(session_data)

# You now have a list `sessions` where each element is a DataFrame corresponding to a session.

#%%

adjacency_matrices = []

# Iterate over each session and compute correlation matrix
for session_data in sessions:
    # Fill missing values (for example, with zero)
    session_filled = session_data.fillna(0)
    
    # Compute correlation matrix for the session
    correlation_matrix = session_filled.corr()
    
    # Apply a threshold to create an adjacency matrix (e.g., correlations > 0.5)
    threshold = 0.7  # Set to 0.7, but you can adjust it
    # Lower threshold creates a denser network (weaker connections included)
    # Higher threshold creates a sparse network (stronger connections included)

    adjacency_matrix = (correlation_matrix > threshold).astype(int)
    
    adjacency_matrices.append(adjacency_matrix)

#%%

# Create dynamic networks for each window
for idx, adjacency_matrix in enumerate(adjacency_matrices):
    G = nx.from_numpy_array(adjacency_matrix.values)
    mapping = {i: col for i, col in enumerate(df_c_raw_all.columns)}
    G = nx.relabel_nodes(G, mapping)
    
    # Plot the network for each window (see window size at beginning)
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=False, node_size=20, node_color='blue', edge_color='gray', alpha=0.6)
    plt.title(f"Neuronal Network at Window {idx + 1}")
    plt.show()

# %%

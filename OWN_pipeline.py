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
#      0        1        2        3         4         5         6         7  \
# 0  1.0  22361.0  44917.0  91943.0  124994.0  158591.0  191578.0  225081.0   

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
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True)
plt.title('Heatmap of df_c_raw_all')
plt.show()

#%% [4]
""" Plot the heatmaps for each session window of one id """

for key, df in c_raw_all_sessions.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of {key}')
    plt.show()

#%% [5]

""" Heatmap Plotting for cs_plus Events

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

​
  """
import seaborn as sns
import matplotlib.pyplot as plt

# Define the sessions to process (s4, s5, s6, s7)
sessions_to_plot = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']

# Loop through the selected sessions
for session_key in sessions_to_plot:
    # Access the DataFrame and cs_plus values from the original dictionary
    session_info = c_raw_sessions_with_cs_plus[session_key]
    session_df = session_info['data']
    
    # Extract cs_plus values and use them to slice the DataFrame
    cs_plus_1 = session_info['cs_plus_1']
    cs_plus_22 = session_info['cs_plus_22']
    cs_plus_3 = session_info['cs_plus_3']
    cs_plus_4 = session_info['cs_plus_4']
    
    # Extract frames from 100 before cs_plus_1 to cs_plus_22
    data_cs_plus_1 = session_df.iloc[cs_plus_1-100:cs_plus_22]
    
    # Extract frames from 100 before cs_plus_22 to cs_plus_3
    data_cs_plus_22 = session_df.iloc[cs_plus_22-100:cs_plus_3]
    
    # Extract frames from 100 before cs_plus_3 to cs_plus_4
    data_cs_plus_3 = session_df.iloc[cs_plus_3-100:cs_plus_4]
    
    # Extract frames from 100 before cs_plus_4 to the end of the session
    data_cs_plus_4 = session_df.iloc[cs_plus_4-100:]
    
    # Concatenate all data for the heatmap
    combined_data = pd.concat([data_cs_plus_1, data_cs_plus_22, data_cs_plus_3, data_cs_plus_4])
    
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(combined_data, cmap="viridis")
    plt.title(f"Heatmap for {session_key}")
    plt.xlabel("Neurons")
    plt.ylabel("Frames")
    plt.show()


# Initialize a dictionary to store the new sessions with calculated values
c_raw_sessions_with_cs_plus = {}

# List of the specific sessions to process
sessions_to_process = ['s4_91943-124993', 's5_124994-158590', 's6_158591-191577', 's7_191578-225080']

# Loop through the selected sessions
for session_key in sessions_to_process:
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
        'cs_plus_4': cs_plus_4
    }

# Output the dictionary to verify the result
for session, values in c_raw_sessions_with_cs_plus.items():
    print(f"{session}: {values}")







#################change this according to the meeting with Janina:###########





""" plot head of sessions"""

# Set the number of rows to plot from the head of each session
num_rows_to_plot = 5

# Loop through each session window and plot the first few rows
for key, df in c_raw_all_sessions.items():
    # Get the first 'num_rows_to_plot' rows of the session
    df_head = df.head(num_rows_to_plot)
    
    # Plot the heatmap of the head of each session
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_head.T, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of Head (First {num_rows_to_plot} Rows) of {key}')
    plt.show()
# %% [6]

""" Count neurons 
Before starting NW analysis"""
# Function to count the number of NaN and non-NaN values in a session
def count_nan_non_nan(session_df):
    # Count the total number of NaN values
    nan_count = session_df.isna().sum().sum()
    
    # Count the total number of non-NaN values
    non_nan_count = session_df.notna().sum().sum()
    
    # Print the counts
    print(f"Number of NaN values: {nan_count}")
    print(f"Number of non-NaN values: {non_nan_count}")
    
    return nan_count, non_nan_count

# Function to find values greater than 80% of the maximum value in the session
def filter_by_threshold(session_df):
    # Find the maximum value in the session
    max_value = session_df.max().max()
    
    # Define the threshold as 80% of the max value (max - 20%)
    threshold = max_value * 0.8
    
    # Filter the DataFrame to get the values greater than the threshold
    above_threshold_df = session_df[session_df > threshold]
    
    # Count the number of values above the threshold
    count_above_threshold = above_threshold_df.notna().sum().sum()
    
    # Print the threshold and count of values above the threshold
    print(f"Threshold (80% of max value {max_value}): {threshold}")
    print(f"Number of values above threshold: {count_above_threshold}")
    
    return count_above_threshold

# Example usage with a session from c_raw_all_sessions:
# Let's assume you want to check 's1_0-22485' from c_raw_all_sessions
session_key = 's1_0-22485'  # Replace with the session key you want to test
session_df = c_raw_all_sessions[session_key]

# Call the functions
nan_count, non_nan_count = count_nan_non_nan(session_df)
count_above_threshold = filter_by_threshold(session_df)

#%%

# import pandas as pd

# # Function to count the number of NaN and non-NaN values in a session
# def count_nan_non_nan(session_df):
#     # Count the total number of NaN values
#     nan_count = session_df.isna().sum().sum()
    
#     # Count the total number of non-NaN values
#     non_nan_count = session_df.notna().sum().sum()
    
#     return nan_count, non_nan_count

# # Function to find values greater than 80% of the maximum value in the session
# def filter_by_threshold(session_df):
#     # Find the maximum value in the session
#     max_value = session_df.max().max()
    
#     # Define the threshold as 80% of the max value (max - 20%)
#     threshold = max_value * 0.8
    
#     # Filter the DataFrame to get the values greater than the threshold
#     above_threshold_df = session_df[session_df > threshold]
    
#     # Count the number of values above the threshold
#     count_above_threshold = above_threshold_df.notna().sum().sum()
    
#     return count_above_threshold, threshold

# # Initialize a list to store the result for each session
# results = []

# # Loop through all sessions in the dictionary
# for session_key, session_df in c_raw_all_sessions.items():
#     # Get NaN and non-NaN counts
#     nan_count, non_nan_count = count_nan_non_nan(session_df)
    
#     # Get count of values above the threshold and the threshold value
#     count_above_threshold, threshold = filter_by_threshold(session_df)
    
#     # Append the results as a row
#     results.append({
#         "Session": session_key,
#         "NaN Count": nan_count,
#         "Non-NaN Count": non_nan_count,
#         "Threshold (Max - 20%)": threshold,
#         "Count Above Threshold": count_above_threshold
#     })

# # Convert the results into a pandas DataFrame
# results_df = pd.DataFrame(results)

# # Display the DataFrame using pandas
# print(results_df)

# # Optionally, save to a CSV file if needed
# results_df.to_csv('session_data_summary.csv', index=False)

# # Optionally, save to an Excel file if needed
# # results_df.to_excel('session_data_summary.xlsx', index=False)

# %%
""" Count active neurons (not-NaN) and above Threshold (max-20%)"""

# Function to count the number of NaN and non-NaN values in a session
def count_nan_non_nan(session_df):
    # Count the total number of NaN values
    nan_count = session_df.isna().sum().sum()
    
    # Count the total number of non-NaN values
    non_nan_count = session_df.notna().sum().sum()
    
    return nan_count, non_nan_count

# Function to find values greater than 80% of the maximum value in the session
def filter_by_threshold(session_df):
    # Find the maximum value in the session
    max_value = session_df.max().max()
    
    # Define the threshold as 80% of the max value (max - 20%)
    threshold = max_value * 0.8
    
    # Filter the DataFrame to get the values greater than the threshold
    above_threshold_df = session_df[session_df > threshold]
    
    # Count the number of values above the threshold
    count_above_threshold = above_threshold_df.notna().sum().sum()
    
    return count_above_threshold, threshold

# Initialize a list to store the result for each session
results = []

# Loop through all sessions in the dictionary
for session_key, session_df in c_raw_all_sessions.items():
    # Get NaN and non-NaN counts
    nan_count, non_nan_count = count_nan_non_nan(session_df)
    
    # Get count of values above the threshold and the threshold value
    count_above_threshold, threshold = filter_by_threshold(session_df)
    
    # Append the results as a row
    results.append({
        "Session": session_key,
        "NaN Count": nan_count,
        "Non-NaN Count": non_nan_count,
        "Threshold (Max - 20%)": threshold,
        "Count Above Threshold": count_above_threshold
    })

# Convert the results into a pandas DataFrame
results_df = pd.DataFrame(results)

# Display the DataFrame using pandas
print(results_df)

# Optionally, save to a CSV file if needed
results_df.to_csv('session_data_summary.csv', index=False)

# Optionally, save to an Excel file if needed
# results_df.to_excel('session_data_summary.xlsx', index=False)

# %%
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt

# # Function to perform network analysis and plot the network for a given session
# def network_analysis(session_df, correlation_threshold=0.5):
#     # Compute the Pearson correlation between all columns (neurons)
#     correlation_matrix = session_df.corr()

#     # Create a NetworkX graph
#     G = nx.Graph()

#     # Add nodes (one for each neuron, represented by a column in the DataFrame)
#     for col in session_df.columns:
#         G.add_node(col)

#     # Add edges based on the correlation matrix
#     for i in range(len(correlation_matrix.columns)):
#         for j in range(i + 1, len(correlation_matrix.columns)):
#             correlation = correlation_matrix.iloc[i, j]
#             if np.abs(correlation) > correlation_threshold:  # Use the absolute value to consider both positive and negative correlations
#                 # Add an edge between the two neurons if the correlation is above the threshold
#                 G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=correlation)

#     # Plot the network
#     plt.figure(figsize=(10, 8))
#     pos = nx.spring_layout(G)  # Layout for positioning the nodes
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

#     # Draw edges with varying thickness based on weight (correlation strength)
#     edges = G.edges(data=True)
#     nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] * 2 for (u, v, d) in edges])

#     plt.title("Neural Network based on Correlation")
#     plt.show()

#     return G  # Return the graph object for further analysis if needed

# # %%
# # Loop through all sessions in the dictionary and apply network analysis
# for session_key, session_df in c_raw_all_sessions.items():
#     print(f"Performing network analysis for {session_key}...")
#     G = network_analysis(session_df)

# %%


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

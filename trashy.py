# %%
import os
import h5py
import pandas as pd

# %%
""" Acess data """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis


# Define the path to the .mat or HDF5 file 
# (no matter what file format is used, stick to .mat as this is the original)
path_934 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_B/Batch_B_2022_990_CFC_GPIO/Batch_B_2022_990_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_934, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# Convert the data  for 'C_Raw_all' and start frames to a pandas DataFrame
df_c_raw_all = pd.DataFrame(data_c_raw_all)
df_start_frame_session = pd.DataFrame(start_frame_session)


# header, nr of rows/columns of the DataFrame
print(f"C_Raw_all: \n{df_c_raw_all.head()}")

print(f"\nNumber of rows: {rows}")
print(f"Number of columns: {columns}")
print(f"Start Frame Sessions: \n{df_start_frame_session.head()}")
rows, columns = df_start_frame_session.shape
# Number of rows: 311447
# Number of columns: 435

# %%
""" Network analysis for one animal id (Batch A or B)"""
#%% #imports
import os
import h5py
import pandas as pd

# %%
""" Acess data """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis


# Define the path to the .mat or HDF5 file 
# (no matter what file format is used, stick to .mat as this is the original)
path_934 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_B/Batch_B_2022_990_CFC_GPIO/Batch_B_2022_990_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_934, 'r') as h5_file:
    # Access the "Data" group and extract the "C_raw_all" dataset
    data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# Convert the data  for 'C_Raw_all' and start frames to a pandas DataFrame
df_c_raw_all = pd.DataFrame(data_c_raw_all)
df_start_frame_session = pd.DataFrame(start_frame_session)


# header, nr of rows/columns of the DataFrame
print(f"C_Raw_all: \n{df_c_raw_all.head()}")
rows, columns = df_c_eaw_all.shape
print(f"\nNumber of rows: {rows}")
print(f"Number of columns: {columns}")
print(f"Start Frame Sessions: \n{df_start_frame_session.head()}")
rows, columns = df_start_frame_session.shape
# Number of rows: 311447
# Number of columns: 435

# %%
""" Network analysis for one animal id (Batch A or B)"""
#%% #imports
import os
import h5py
import pandas as pd

# %%
""" Acess data """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis


# Define the path to the .mat or HDF5 file 
# (no matter what file format is used, stick to .mat as this is the original)
path_934 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_B/Batch_B_2022_990_CFC_GPIO/Batch_B_2022_990_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_934, 'r') as h5_file:
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
rows, columns = df_start_frame_session.shape
# Number of rows: 311447
# Number of columns: 435

# %%
""" Create windows and store them in a dictionary """

# Initialize the dictionary to store the windows
c_raw_all_sessions = {}

# Loop through each start frame session to create the windows
for i in range(len(df_start_frame_session.columns)):
    start = int(df_start_frame_session.iloc[0, i])
    
    # If not the last index, take the next start and subtract 1
    if i < len(df_start_frame_session.columns) - 1:
        end = int(df_start_frame_session.iloc[0, i+1]) - 1
    else:
        # Last window ends at the last row of df_c_raw_all
        end = rows - 1
    
    # Create a key like 's1_0-22486', 's2_22486-44648', etc.
    key = f"s{i+1}_{start}-{end}"
    
    # Store the corresponding rows in the dictionary
    c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

# Output to verify the dictionary content
for key, df in c_raw_all_sessions.items():
    print(f"{key}: {df.shape}")

# %%
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

# %%
""" Create windows and store them in a dictionary """

# Initialize the dictionary to store the windows
c_raw_all_sessions = {}

# Loop through each start frame session to create the windows
for i in range(len(df_start_frame_session.columns)-1):
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

# %%
""" Create windows and store them in a dictionary """

# Initialize the dictionary to store the windows
c_raw_all_sessions = {}

# Loop through each start frame session to create the windows
for i in range(len(df_start_frame_session.columns)):
    start = int(df_start_frame_session.iloc[0, i])
    
    # If not the last index, take the next start and subtract 1
    if i < len(df_start_frame_session.columns) - 1:
        end = int(df_start_frame_session.iloc[0, i+1]) - 1
    else:
        # Last window ends at the last row of df_c_raw_all
        end = rows - 1
    
    # Create a key like 's1_0-22486', 's2_22486-44648', etc.
    key = f"s{i+1}_{start}-{end}"
    
    # Store the corresponding rows in the dictionary
    c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

# Output to verify the dictionary content
for key, df in c_raw_all_sessions.items():
    print(f"{key}: {df.shape}")

# %%
""" Network analysis for one animal id (Batch A or B)"""
#%% #imports
import os
import h5py
import pandas as pd

# %%
""" Acess data """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis


# Define the path to the .mat or HDF5 file 
# (no matter what file format is used, stick to .mat as this is the original)
path_934 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_B/Batch_B_2022_990_CFC_GPIO/Batch_B_2022_990_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_934, 'r') as h5_file:
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
# Number of rows: 311447
# Number of columns: 435

# %%
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

# %%
""" Plot the heatmap for the entire df_c_raw_all """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True)
plt.title('Heatmap of df_c_raw_all')
plt.show()

# %%
""" Plot the heatmap for the entire df_c_raw_all """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True)
plt.title('Heatmap of df_c_raw_all')
plt.show()

# %%
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
""" Plot the heatmap for the entire df_c_raw_all """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True)
plt.title('Heatmap of df_c_raw_all')
plt.show()

# %%
""" Plot the heatmap for the entire df_c_raw_all """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True)
plt.title('Heatmap of df_c_raw_all')
plt.show()

# %%
""" Plot the heatmaps for each session window """

for key, df in c_raw_all_sessions.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of {key}')
    plt.show()

# %%
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

# %%
""" Network analysis for one animal id (Batch A or B)"""
#%% #imports
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
""" Acess data """

# choose based on heatmap animal that has many active neurons at beginning to 
# have a good starting point for the analysis


# Define the path to the .mat or HDF5 file 
# (no matter what file format is used, stick to .mat as this is the original)
path_934 = "/home/manuela/Documents/PROJECT_Pyunicorn_ClimateNetworks_SEP24/data/Batch_B/Batch_B_2022_990_CFC_GPIO/Batch_B_2022_990_CFC_GPIO/Data_Miniscope_PP.mat"

# Open the HDF5 file and extract the required data
with h5py.File(path_934, 'r') as h5_file:
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
# Number of rows: 311447
# Number of columns: 435

# %%
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

# s1_0-22485: (22486, 435)
# s2_22486-44647: (22162, 435)
# s3_44648-91938: (47291, 435)
# s4_91939-126214: (34276, 435)
# s5_126215-159820: (33606, 435)
# s6_159821-193798: (33978, 435)
# s7_193799-226676: (32878, 435)
# s8_226677-245159: (18483, 435)
# s9_245160-292745: (47586, 435)
# s10_292746-311446: (18701, 435)

# %%
""" Plot the heatmap for the entire df_c_raw_all """

# Plot heatmap for the entire dataset
plt.figure(figsize=(10, 8))
sns.heatmap(df_c_raw_all.T, cmap='viridis', cbar=True)
plt.title('Heatmap of df_c_raw_all')
plt.show()

# %%
""" Plot the heatmaps for each session window """

for key, df in c_raw_all_sessions.items():
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.T, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of {key}')
    plt.show()

# %%
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

# %%
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

# %%
import pandas as pd

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

# Display the DataFrame in a nice table format
import ace_tools as tools; tools.display_dataframe_to_user(name="Session Data Summary", dataframe=results_df)

# %%
import pandas as pd

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
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to perform network analysis and plot the network for a given session
def network_analysis(session_df, correlation_threshold=0.5):
    # Compute the Pearson correlation between all columns (neurons)
    correlation_matrix = session_df.corr()

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes (one for each neuron, represented by a column in the DataFrame)
    for col in session_df.columns:
        G.add_node(col)

    # Add edges based on the correlation matrix
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            correlation = correlation_matrix.iloc[i, j]
            if np.abs(correlation) > correlation_threshold:  # Use the absolute value to consider both positive and negative correlations
                # Add an edge between the two neurons if the correlation is above the threshold
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=correlation)

    # Plot the network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Layout for positioning the nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)

    # Draw edges with varying thickness based on weight (correlation strength)
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight'] * 2 for (u, v, d) in edges])

    plt.title("Neural Network based on Correlation")
    plt.show()

    return G  # Return the graph object for further analysis if needed

# %%
# Loop through all sessions in the dictionary and apply network analysis
for session_key, session_df in c_raw_all_sessions.items():
    print(f"Performing network analysis for {session_key}...")
    G = network_analysis(session_df)

# %%
""" Network analysis per session - 
one session: which neurons are active simultaneously?"""

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
# Test on one session
session_key = 's1_0-22485'
session_df = c_raw_all_sessions[session_key]

# Perform network analysis based on neuron activity
G = network_analysis_by_activity(session_df)

# %%
# Loop through all sessions in the dictionary and apply network analysis
for session_key, session_df in c_raw_all_sessions.items():
    print(f"Performing network analysis for {session_key}...")
    G = network_analysis_by_activity(session_df)



#%% [0]

# Imports
import h5py
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

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

#%%
# check'split_sessions_dict'
split_sessions_dict

#check keys: are all sessions processed?
for key in split_sessions_dict:
    print(key)
for key, session_data in split_sessions_dict.items():
    rows, cols = session_data.shape
    print(f"Session {key}: Rows/Frames: {rows}; Columns/Neurons: {cols} ")

#%%

""" III.) Run analysis for each key in split_sessions_dict """
# idea: transform each matrix of the key into df and feed function for nw analysis

""" without saving"""

# Define the function for plotting and network analysis 
# def plot_and_analyse_nw(activity_df, corr_th=0.2, seed=12345678):
#     """ Function for network plot and analysis"""
#     # Step 1: Drop columns with NaN values - all!
#     activity_df_non_nan = activity_df.dropna(how='any', axis=1)

#     # Check if there is enough data to compute the correlation matrix
#     if activity_df_non_nan.shape[1] < 2:  # Fewer than 2 neurons (columns) left after dropping NaN
#         print(f"Skipping this session: Not enough neurons after dropping NaN values.")
#         return None  # Skip this session

#     # Step 2: Compute correlation matrix
#     corr_mat = np.corrcoef(activity_df_non_nan.T)  # Transpose so that neurons are along columns

#     # Step 3: Zero diagonal elements to ignore self-connections
#     np.fill_diagonal(corr_mat, 0)

#     # Step 4: Zero out connections below the threshold
#     corr_mat[corr_mat < corr_th] = 0

#     # Step 5: Create a graph from the correlation matrix
#     G = nx.from_numpy_array(corr_mat)

#     # Step 6: Visualize the graph using a spring layout
#     plt.figure(figsize=(10, 10))
#     nx.draw(G, pos=nx.spring_layout(G, seed=seed), node_size=10, width=0.3)
#     plt.title(f"Neuronal Network (Threshold {corr_th})")

#     plt.show()

#     # Analyze graph metrics
#     num_edges = G.number_of_edges()
#     num_nodes = G.number_of_nodes()
    
#     # Output graph metrics
#     print(f"Number of edges: {num_edges}")
#     print(f"Number of nodes: {num_nodes}")
    
#     # Step 7: Compute mean degree of the whole network
#     degrees = [degree for node, degree in G.degree()]
#     mean_degree = np.mean(degrees)
#     print(f"Mean degree of the whole network: {mean_degree}")
    
#     # Step 8: Calculate the number of sub-networks (connected components)
#     components = list(nx.connected_components(G))
#     num_sub_networks = len(components)
#     print(f"Number of sub-networks: {num_sub_networks}")
    
#     assortativity = nx.degree_assortativity_coefficient(G)
#     print(f"Assortativity (degree correlation): {assortativity}")

    
#     return G, num_edges, num_nodes, mean_degree, assortativity

#%%
""" Try for one session """
# Extract the first session from split_sessions_dict and save it as df
# first_session_key = next(iter(split_sessions_dict))
# first_session_data = split_sessions_dict[first_session_key]

# # Convert the first session to a df
# df_first_session = pd.DataFrame(first_session_data)

# # Run the function on the extracted first session
# plot_and_analyse_nw(df_first_session)

#%%

""" ADD SAVING PLOT """
import os
from datetime import datetime

# Updated function to save the plot with a unique name and timestamp
def plot_and_analyse_nw(activity_df, key_name, corr_th=0.2, seed=12345678):
    """ Function for network plot and analysis with plot saving feature"""
    
    # Step 1: Drop columns with NaN values - all!
    activity_df_non_nan = activity_df.dropna(how='any', axis=1)

    # Check if there is enough data to compute the correlation matrix
    if activity_df_non_nan.shape[1] < 2:  # Fewer than 2 neurons (columns) left after dropping NaN
        print(f"Skipping this session: Not enough neurons after dropping NaN values.")
        return None  # Skip this session

    # Step 2: Compute correlation matrix
    corr_mat = np.corrcoef(activity_df_non_nan.T)  # Transpose so that neurons are along columns

    # Step 3: Zero diagonal elements to ignore self-connections
    np.fill_diagonal(corr_mat, 0)

    # Step 4: Zero out connections below the threshold
    corr_mat[corr_mat < corr_th] = 0

    # Step 5: Create a graph from the correlation matrix
    G = nx.from_numpy_array(corr_mat)

    # Create folder 'nw_plots' if it doesn't exist
    plot_folder = os.path.join(os.getcwd(), 'nw_plots')
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    # Generate unique file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{key_name}_{timestamp}.png"
    plot_path = os.path.join(plot_folder, plot_filename)

    # Step 6: Visualize the graph using a spring layout and save the plot
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=nx.spring_layout(G, seed=seed), node_size=10, width=0.3)
    plt.title(f"Neuronal Network (Threshold {corr_th})")
    plt.savefig(plot_path)
    plt.close()  # Close the figure to avoid displaying it here

    print(f"Plot saved to {plot_path}")

    # Analyze graph metrics
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    
    # Output graph metrics
    print(f"Number of edges: {num_edges}")
    print(f"Number of nodes: {num_nodes}")
    
    # Step 7: Compute mean degree of the whole network
    degrees = [degree for node, degree in G.degree()]
    mean_degree = np.mean(degrees)
    print(f"Mean degree of the whole network: {mean_degree}")
    
    # Step 8: Calculate the number of sub-networks (connected components)
    components = list(nx.connected_components(G))
    num_sub_networks = len(components)
    print(f"Number of sub-networks: {num_sub_networks}")
    
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"Assortativity (degree correlation): {assortativity}")

    
    return G, num_edges, num_nodes, mean_degree, assortativity

# Run the function on the first session and save the plot
#plot_and_analyse_nw(df_first_session, first_session_key)


#%%
# Loop over each session in split_sessions_dict
network_analysis_results = {}

for key_name, session_data in split_sessions_dict.items():
    # Convert session data to DataFrame
    session_df = pd.DataFrame(session_data)
    
    # Run the analysis and plot function
    G, num_edges, num_nodes, mean_degree, assortativity = plot_and_analyse_nw(session_df, key_name)
    
    # Store the results in the dictionary with the key_name as the header
    network_analysis_results[key_name] = {
        'num_edges': num_edges,
        'num_nodes': num_nodes,
        'mean_degree': mean_degree,
        'assortativity': assortativity
    }

# Convert the results dictionary to a DataFrame for better visualization
network_analysis_df = pd.DataFrame.from_dict(network_analysis_results, orient='index')

# Display the DataFrame
print(network_analysis_df)

#%%#
""" Add in total 5 more columns to the df """

"""Add R+ and R- info """
#R_plus and R_minus lists

R_plus = [1037, 988, 1002, 936, 970]
R_minus = [1022, 994, 982, 990, 951, 935, 1012, 991, 955]

# Create new columns for animal_id, batch, session, R_plus, and R_minus
def extract_info_from_key(key):
    """Extract animal_id, batch, session from the key."""
    match = re.search(r"(\d+)_([A-F])_s(\d+)", key)
    if match:
        animal_id = int(match.group(1))  # Extract numeric part before the first underscore
        batch = match.group(2)           # Extract the letter (batch)
        session = int(match.group(3))    # Extract the session number
        return animal_id, batch, session
    return None, None, None

# Apply the extraction and check for R_plus and R_minus
network_analysis_df['animal_id'] = network_analysis_df.index.map(lambda key: extract_info_from_key(key)[0])
network_analysis_df['batch'] = network_analysis_df.index.map(lambda key: extract_info_from_key(key)[1])
network_analysis_df['session'] = network_analysis_df.index.map(lambda key: extract_info_from_key(key)[2])

# Add the R_plus and R_minus columns
network_analysis_df['R_plus'] = network_analysis_df['animal_id'].apply(lambda x: 1 if x in R_plus else 0)
network_analysis_df['R_minus'] = network_analysis_df['animal_id'].apply(lambda x: 1 if x in R_minus else 0)

print(network_analysis_df)

#%%

""" Save as csv """
# Save the updated DataFrame as a CSV file in the current working directory
csv_filename = "network_analysis_results.csv"
network_analysis_df.to_csv(csv_filename, index=True)

#print(f"The DataFrame has been saved as {csv_filename} in the current working directory.")
# Save the dfas a CSV file with a timestamp in the filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename_with_timestamp = f"network_analysis_results_{timestamp}.csv"
network_analysis_df.to_csv(csv_filename_with_timestamp, index=True)

print(f"The DataFrame for network analysis has been saved as {csv_filename_with_timestamp} in the current working directory.")

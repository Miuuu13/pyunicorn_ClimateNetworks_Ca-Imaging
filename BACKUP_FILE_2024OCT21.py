##### MERGE BACKUPS for coding at home

#%% [0]

# Imports
import os
import re     
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

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

#%% [4]
# check'split_sessions_dict'
split_sessions_dict

#check keys: are all sessions processed?
for key in split_sessions_dict:
    print(key)
for key, session_data in split_sessions_dict.items():
    rows, cols = session_data.shape
    print(f"Session {key}: Rows/Frames: {rows}; Columns/Neurons: {cols} ")

#%% [5]

""" III.) Run analysis for each key in split_sessions_dict """
# idea: transform each matrix of the key into df and feed function for nw analysis

""" Try for one session """
# Extract the first session from split_sessions_dict and save it as df
# first_session_key = next(iter(split_sessions_dict))
# first_session_data = split_sessions_dict[first_session_key]

# # Convert the first session to a df
# df_first_session = pd.DataFrame(first_session_data)

# # Run the function on the extracted first session
# plot_and_analyse_nw(df_first_session)


""" ADD SAVING PLOT """


# Updated function to save the plot with a unique name and timestamp
# def plot_and_analyse_nw(activity_df, key_name, corr_th=0.2, seed=12345678):
#     """ Function for network plot and analysis with plot saving feature"""
    
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

#     # Create folder 'nw_plots' if it doesn't exist
#     plot_folder = os.path.join(os.getcwd(), 'nw_plots')
#     if not os.path.exists(plot_folder):
#         os.makedirs(plot_folder)

#     # Generate unique file name with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plot_filename = f"{key_name}_{timestamp}.png"
#     plot_path = os.path.join(plot_folder, plot_filename)

#     # Step 6: Visualize the graph using a spring layout and save the plot
#     plt.figure(figsize=(10, 10))
#     nx.draw(G, pos=nx.spring_layout(G, seed=seed), node_size=10, width=0.3)
#     plt.title(f"Neuronal Network (Threshold {corr_th})")
#     plt.savefig(plot_path)
#     plt.close()  # Close the figure to avoid displaying it here

#     print(f"Plot saved to {plot_path}")

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

# Run the function on the first session and save the plot
#plot_and_analyse_nw(df_first_session, first_session_key)


""" Add new Measurements"""

# number_of_edges(G)
# density(G)
# diameter(G)    # for this one please check whether this is slowing down the calculation
# transitivity(G)   # Clustering Coefficient
# average_clustering(G)
# wiener_index(G)
# average_degree_connectivity(G
# degree_assortativity_coefficient(G)
# degree_pearson_correlation_coefficient(G)
# number_strongly_connected_components(G)
# nx.average_shortest_path_length(G)

""" Just use functions from networkX docu; think about meaning later !"""
# function to save the plot with a unique name and timestamp

def plot_and_analyse_nw(activity_df, key_name, corr_th=0.2, seed=12345678):
    """ Function for network plot and analysis with plot saving feature - added new metrics 20SEP,2pm"""
    
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
    
    # Step 9: Graph Density
    density = nx.density(G)
    print(f"Graph density: {density}")
    
    # Step 10: Graph Diameter 
    try:
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            print(f"Graph diameter: {diameter}")
        else:
            diameter = None
            print(f"Graph is disconnected; diameter not defeined.")
    except Exception as e:
        diameter = None
        print(f"Error calculating diameter: {e}")
    
    # Step 11: Transitivity 
    transitivity = nx.transitivity(G)
    print(f"Transitivity: {transitivity}")
    
    # Step 12: Average Clustering Coefficient
    avg_clustering = nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg_clustering}")
    
    # Step 13: Wiener Index  #(?)
    try:
        if nx.is_connected(G):
            wiener_index = nx.wiener_index(G)
            print(f"Wiener Index: {wiener_index}")
        else:
            wiener_index = None
            print(f"Graph is disconnected, the Wiener Index is not defined. ")
    except Exception as e:
        wiener_index = None
        print(f"Error calculating Wiener Index: {e}")
    
    # Step 14: Average Degree Connectivity
    avg_deg_connectivity = nx.average_degree_connectivity(G)
    print(f"Average degree connectivity: {avg_deg_connectivity}")
    
    # Step 15: Degree Assortativity Coefficient
    assortativity_coeff = nx.degree_assortativity_coefficient(G)
    print(f"Degree assortativity coefficient: {assortativity_coeff}")
    
    # Step 16: Degree of the Pearson Correlation Coefficient
    try:
        pearson_corr = nx.degree_pearson_correlation_coefficient(G)
        print(f"Degree Pearson correlation coefficient: {pearson_corr}")
    except Exception as e:
        pearson_corr = None
        print(f"Error: calculating degree_pearson_correlation_coefficient: {e}")
    
    # Step 17: Average Shortest Path Length (Check for performance)
    try:
        if nx.is_connected(G):
            avg_shortest_path_len = nx.average_shortest_path_length(G)
            print(f"Average shortest path length: {avg_shortest_path_len}")
        else:
            avg_shortest_path_len = None
            print(f"Graph is disconnected, average shortest path length is not defined.") 
    except Exception as e:
        avg_shortest_path_len = None
        print(f"Error: Calculating avg_shortest_path_len: {e}")
    
    return {
        'graph': G,
        'num_edges': num_edges,
        'num_nodes': num_nodes,
        'mean_degree': mean_degree,
        'assortativity': assortativity,
        'density': density,
        'diameter': diameter,
        'transitivity': transitivity,
        'avg_clustering': avg_clustering,
        'wiener_index': wiener_index,
        'avg_deg_connectivity': avg_deg_connectivity,
        'assortativity_coeff': assortativity_coeff,
        'pearson_corr': pearson_corr,
        'avg_shortest_path_len': avg_shortest_path_len
    }


# Run the function on the first session and save the plot
#plot_and_analyse_nw(df_first_session, first_session_key)


#TODO if metrics added: also add the metrics in the following loop -> changed loop; todo cleared
#should now be able to handle various metrics 
#%% [6]
# Loop over each session in split_sessions_dict
network_analysis_results = {}

for key_name, session_data in split_sessions_dict.items():
    # Convert session data to DataFrame
    session_df = pd.DataFrame(session_data)
    
    # Run the analysis and plot function
    analysis_results = plot_and_analyse_nw(session_df, key_name)  # get the dictionary

    # Check if results are None (for skipped sessions)
    if analysis_results is None:
        continue
    
    # Store the results in the dictionary with the key_name as the header
    network_analysis_results[key_name] = analysis_results

# Convert the results dictionary to a DataFrame for better visualization
network_analysis_df = pd.DataFrame.from_dict(network_analysis_results, orient='index')

# Display the DataFrame
print(network_analysis_df)

# for key_name, session_data in split_sessions_dict.items():
#     # Convert session data to DataFrame
#     session_df = pd.DataFrame(session_data)
    
#     # Run the analysis and plot function
#     G, num_edges, num_nodes, mean_degree, assortativity = plot_and_analyse_nw(session_df, key_name)
    
#     # Store the results in the dictionary with the key_name as the header
#     network_analysis_results[key_name] = {
#         'graph': G,
#         'num_edges': num_edges,
#         'num_nodes': num_nodes,
#         'mean_degree': mean_degree,
#         'assortativity': assortativity,
#         'density': density,
#         'diameter': diameter,
#         'transitivity': transitivity,
#         'avg_clustering': avg_clustering,
#         'wiener_index': wiener_index,
#         'avg_deg_connectivity': avg_deg_connectivity,
#         'assortativity_coeff': assortativity_coeff,
#         'pearson_corr': pearson_corr,
#         'avg_shortest_path_len': avg_shortest_path_len
#     }

# # Convert the results dictionary to a DataFrame for better visualization
# network_analysis_df = pd.DataFrame.from_dict(network_analysis_results, orient='index')

# Display the DataFrame
print(network_analysis_df)

#%%# [7]
""" Add in total 5 more columns to the df """
#rename first column (index) to key_name
#network_analysis_df = network_analysis_df.reset_index().rename(columns={'index': 'key_name'})

""" Add 3 columns for id, batch, session (is currently only together stored in key name)"""

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

# Apply the extraction function "extract_info_from_key"; use lambda expression and map (like in Scala)
network_analysis_df['animal_id'] = network_analysis_df.index.map(lambda key: extract_info_from_key(key)[0])
network_analysis_df['batch'] = network_analysis_df.index.map(lambda key: extract_info_from_key(key)[1])
network_analysis_df['session'] = network_analysis_df.index.map(lambda key: extract_info_from_key(key)[2])


""" Add 2 columns R_plus, R_minus with R+ and R- info """
#R_plus and R_minus lists
R_plus = [1037, 988, 1002, 936, 970]
R_minus = [1022, 994, 982, 990, 951, 935, 1012, 991, 955]

# Add the R_plus and R_minus columns
#if in a list 1 (TRUE) else 0 (FALSE)
network_analysis_df['R_plus'] = network_analysis_df['animal_id'].apply(lambda x: 1 if x in R_plus else 0)
network_analysis_df['R_minus'] = network_analysis_df['animal_id'].apply(lambda x: 1 if x in R_minus else 0)

print(network_analysis_df)

#%% [8]

""" Save as csv """

csv_filename = "network_analysis_results.csv"
network_analysis_df.to_csv(csv_filename, index=True)

# Save the df as a CSV file with a timestamp inside filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename_with_timestamp = f"network_analysis_results_{timestamp}.csv"
network_analysis_df.to_csv(csv_filename_with_timestamp, index=True)

print(f"DataFrame for network analysis has been saved as {csv_filename_with_timestamp} in the current working directory.")


#%%

""" Save df to json """
#network_analysis_df.to_json(csv_filename_with_timestamp, index=True)
# Save the df as a JSON file with a timestamp inside filename
# json_filename_with_timestamp = f"network_analysis_results_{timestamp}.json"
# network_analysis_df.to_json(json_filename_with_timestamp, orient='index')

# print(f"DataFrame for network analysis has been saved as {json_filename_with_timestamp} in the current working directory.")



############plots

#%%

""" Plotting for extinction days """




# %%

#TODO: Adjust plots to handle more metrices

# data = {
#     'key_name': ['1022_B_s1', '1022_B_s2', '1022_B_s3', '1022_B_s4', '1022_B_s5', '1022_B_s6', '1022_B_s7', '1022_B_s8', '1022_B_s9', '1022_B_s10',
#                  '1002_B_s1', '1002_B_s2', '1002_B_s3', '1002_B_s4', '1002_B_s5', '1002_B_s6', '1002_B_s7', '1002_B_s8', '1002_B_s9', '1002_B_s10'],
#     'num_edges': [179, 219, 62, 173, 220, 291, 208, 194, 90, 469, 22, 22, 8, 37, 49, 55, 18, 15, 32, 150],
#     'num_nodes': [184, 183, 151, 190, 196, 185, 176, 155, 145, 225, 34, 31, 27, 46, 53, 58, 36, 28, 28, 84],
#     'mean_degree': [1.94, 2.39, 0.82, 1.82, 2.24, 3.14, 2.36, 2.5, 1.24, 4.17, 1.29, 1.42, 0.59, 1.60, 1.85, 1.89, 1.0, 1.07, 2.29, 3.57],
#     'assortativity': [0.29, 0.34, 0.59, 0.21, 0.10, 0.17, 0.15, 0.49, 0.18, 0.23, 0.13, 0.23, -0.03, 0.38, 0.25, 0.26, 0.13, 0.18, 0.33, 0.18],
#     'animal_id': ['1022', '1022', '1022', '1022', '1022', '1022', '1022', '1022', '1022', '1022', '1002', '1002', '1002', '1002', '1002', '1002', '1002', '1002', '1002', '1002'],
#     'R_plus': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     'R_minus': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     'session': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# }

# df = pd.DataFrame(data)

df= network_analysis_df

r_plus_df = df[df['R_plus'] == 1]
r_minus_df = df[df['R_minus'] == 1]

# Filter 
r_plus_sessions = r_plus_df[(r_plus_df['session'] >= 4) & (r_plus_df['session'] <= 7)]
r_minus_sessions = r_minus_df[(r_minus_df['session'] >= 4) & (r_minus_df['session'] <= 7)]


r_plus_sums = r_plus_sessions.groupby('session').sum()
r_minus_sums = r_minus_sessions.groupby('session').sum()

# Create bar plots for R_plus and R_minus
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

r_plus_sums[['num_edges', 'num_nodes', 'mean_degree', 'assortativity']].plot(kind='bar', ax=axes[0], title="R_plus Summed over Sessions 4-7", legend=True)
r_minus_sums[['num_edges', 'num_nodes', 'mean_degree', 'assortativity']].plot(kind='bar', ax=axes[1], title="R_minus Summed over Sessions 4-7", legend=True)

plt.tight_layout()
plt.show()

# %%
"""line plots for R_plus/_minus animals across sessions 4-7"""


r_plus_animal_ids = r_plus_sessions['animal_id'].unique()
#metrics from nw analysis in the df columns
metrics = ['num_edges', 'num_nodes', 'mean_degree', 'assortativity']


fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12))

# Loop through each metric and plot the line plots for R_plus
for i, metric in enumerate(metrics):
    for animal_id in r_plus_animal_ids:
        animal_data = r_plus_sessions[r_plus_sessions['animal_id'] == animal_id]
        axes[i].plot(animal_data['session'], animal_data[metric], label=f"Animal {animal_id}")
    axes[i].set_title(f"R_plus: {metric} over Sessions")
    axes[i].set_xlabel('Session')
    axes[i].set_ylabel(metric.capitalize())
    axes[i].legend()

plt.tight_layout()
plt.show()
# %%

"""R-"""
r_minus_animal_ids = r_minus_sessions['animal_id'].unique()

# Initialize subplots for each metric
fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 12))

# Loop through each metric and plot the line plots for R_minus
for i, metric in enumerate(metrics):
    for animal_id in r_minus_animal_ids:
        animal_data = r_minus_sessions[r_minus_sessions['animal_id'] == animal_id]
        axes[i].plot(animal_data['session'], animal_data[metric], label=f"Animal {animal_id}")
    axes[i].set_title(f"R_minus: {metric} over Sessions")
    axes[i].set_xlabel('Session')
    axes[i].set_ylabel(metric.capitalize())
    axes[i].legend()

plt.tight_layout()
plt.show()
# %%


#######################################################################################################

#######################################################################################################
# Overall_activity.py
#######################################################################################################


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









#######################################################################################################

#######################################################################################################
# NAN value handling
#######################################################################################################


""" handle Nan"""
def plot_first_10_neurons_separately_skip_invalid(session_dict, session_key):
    """
    Plots the first 10 neurons' activity (luminescence over frames) in separate plots for a given session,
    with the same y-axis scale (max value of all neurons) and skips neurons with NaN/Inf values.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(100, session_data.shape[1])
    
    # Calculate the maximum value across the first 10 neurons that do not have NaN or Inf values
    valid_data = session_data[:, :num_neurons]
    valid_data = valid_data[~np.isnan(valid_data).any(axis=1)]  # Remove rows with NaN values
    valid_data = valid_data[~np.isinf(valid_data).any(axis=1)]  # Remove rows with Inf values

    if valid_data.size == 0:
        print(f"All neurons in {session_key} contain NaN or Inf values.")
        return
    
    max_value = np.max(valid_data)  # Get the maximum value after excluding NaN/Inf
    
    # Create individual plots for each neuron with the same y-axis scale
    for neuron_id in range(num_neurons):
        neuron_data = session_data[:, neuron_id]
        
        # Check if the neuron contains NaN or Inf values
        if np.isnan(neuron_data).any() or np.isinf(neuron_data).any():
            print(f"Neuron {neuron_id} in session {session_key} contains NaN or Inf values and was skipped.")
            continue  # Skip plotting this neuron

        # Plot only if valid
        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(neuron_data, label=f'Neuron {neuron_id}')
        
        plt.title(f"Neuron {neuron_id} Activity for {session_key}")
        plt.xlabel("Frames")
        plt.ylabel("Luminescence (Neuron Activity)")
        plt.ylim(0, max_value)  # Set y-axis limits based on the maximum valid value
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage:
# Plot the first 10 neurons in separate plots for session '1022_B_s1' and skip invalid neurons
plot_first_10_neurons_separately_skip_invalid(split_sessions_dict, '1022_B_s1')
#%%

""" count nan values per session """

def count_nan_values_in_sessions(session_dict):
    """
    Counts and prints the number of NaN values in each session in the session_dict.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    """
    for session_key, session_data in session_dict.items():
        # Count the total number of NaN values in the session data
        nan_count = np.isnan(session_data).sum()
        
        # Print the result
        if nan_count > 0:
            print(f"Session {session_key} contains {nan_count} NaN values.")
        else:
            print(f"Session {session_key} contains no NaN values.")

# Example usage:
# Count and print the number of NaN values for each session in the split_sessions_dict
count_nan_values_in_sessions(split_sessions_dict)

# %%
""" nans per neuron in one session """

def display_nan_count_per_neuron(session_dict, session_key):
    """
    Displays the number of NaN values for each neuron in a specific session.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to check (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # Loop through each neuron (column) and count the NaN values
    for neuron_id in range(session_data.shape[1]):
        nan_count = np.isnan(session_data[:, neuron_id]).sum()
        
        # Print the neuron ID and its respective NaN count
        if nan_count > 0:
            print(f"Neuron {neuron_id} in session {session_key} contains {nan_count} NaN values.")
        else:
            print(f"Neuron {neuron_id} in session {session_key} contains no NaN values.")

# Example usage:
# Display the number of NaN values per neuron in session '1022_B_s1'
display_nan_count_per_neuron(split_sessions_dict, '1022_B_s5')

#%%

""" total values (incl. nan)"""
def display_total_values_per_neuron(session_dict, session_key):
    """
    Displays the total number of values (frames) for each neuron in a specific session.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to check (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # The total number of values per neuron is the number of rows (frames)
    total_frames = session_data.shape[0]

    # Loop through each neuron (column) and print the total number of values
    for neuron_id in range(session_data.shape[1]):
        print(f"Neuron {neuron_id} in session {session_key} has {total_frames} total values (frames).")

# Example usage:
# Display the total number of values per neuron in session '1022_B_s1'
display_total_values_per_neuron(split_sessions_dict, '1022_B_s1')

# %%

""" Combine count total and count nan values per neuron"""

def display_total_and_nan_values_per_neuron(session_dict, session_key):
    """
    Displays the total number of values (frames) and the number of NaN values for each neuron
    in a specific session.
    
    Parameters:
    - session_dict: dictionary containing the split session data.
    - session_key: key of the session to check (e.g., '1022_B_s1').
    """
    if session_key not in session_dict:
        print(f"Session {session_key} not found.")
        return

    # Get the data for the specified session
    session_data = session_dict[session_key]

    # The total number of values per neuron is the number of rows (frames)
    total_frames = session_data.shape[0]

    # Loop through each neuron (column)
    for neuron_id in range(session_data.shape[1]):
        # Count the NaN values for the current neuron
        nan_count = np.isnan(session_data[:, neuron_id]).sum()

        # Prepare the message based on whether the neuron has NaN values or not
        if nan_count > 0:
            print(f"Neuron {neuron_id} in session {session_key} has {total_frames} total values (frames) and contains {nan_count} NaN values.")
        else:
            print(f"Neuron {neuron_id} in session {session_key} has {total_frames} total values (frames) and contains no NaN values.")

# Example usage:
# Display the total number of values and NaN counts per neuron in session '1022_B_s1'
display_total_and_nan_values_per_neuron(split_sessions_dict, '1022_B_s7')

# %%


#######################CHECK dff and z score normalization ##################################################

def check_deltaf_f_and_zscore(c_raw_all_dict):
    """
    Checks whether deltaF/F and z-score have been applied to the data in c_raw_all_dict.
    
    Parameters:
    - c_raw_all_dict: dictionary containing the raw calcium imaging data.
    
    Returns:
    - Dictionary with session keys and booleans indicating whether deltaF/F or z-score is likely applied.
    """
    results = {}

    for session_key, data in c_raw_all_dict.items():
        c_raw_all = data['C_Raw_all']

        # Check for deltaF/F by inspecting if the data has both positive and negative values?
        #TODO is this already enough???
        has_positive_values = (c_raw_all > 0).any()
        has_negative_values = (c_raw_all < 0).any()
        deltaf_f_applied = has_positive_values and has_negative_values

        # Check for z-score by inspecting if the data has:
        # 1. mean around 0 and 
        # 2. standard deviation around 1
        mean_value = np.mean(c_raw_all)
        std_value = np.std(c_raw_all)
        zscore_applied = np.isclose(mean_value, 0, atol=0.1) and np.isclose(std_value, 1, atol=0.1)

        # Store the results for this session
        results[session_key] = {
            'deltaF/F_applied': deltaf_f_applied,
            'zscore_applied': zscore_applied
        }

        # Print results for the session
        print(f"Session {session_key}:")
        print(f"  - DeltaF/F applied: {'Yes' if deltaf_f_applied else 'No'}")
        print(f"  - Z-score applied: {'Yes' if zscore_applied else 'No'}")
        print(f"  - Mean of data: {mean_value:.4f}, Standard deviation of data: {std_value:.4f}")
    
    return results

# Example usage:
# Check for deltaF/F and z-score in the c_raw_all_dict
check_deltaf_f_and_zscore(c_raw_all_dict)
#%%
""" Excluded global way, does not make sense! """
""" Z score - 3 ways 

# Global Z-Score for c_raw_all_dict:

#     Concatenate all C_Raw_all data across all sessions into a single matrix (all_data), 
#     compute the global mean and standard deviation (mean_all and std_all), 
#     and then apply the z-score normalization across all frames and neurons. 
#     The normalized data is stored in c_raw_all_dict_z.

Z-Score for Each Session in split_sessions_dict:

    For each session, compute the mean and standard deviation of the entire session and 
    apply the z-score normalization. 
    The normalized session-specific data is stored in split_sessions_dict_z.

Z-Score for Each Neuron in Each Session:

    For each neuron (i.e., each column) within each session, 
    compute the mean and standard deviation column-wise and apply the z-score normalization. 
    The result is stored in split_sessions_dict_z_per_neuron.
"""

def apply_zscore_all_modes(c_raw_all_dict, split_sessions_dict):
    """
    Applies z-score normalization in three modes:
    1. On the entire dataset in c_raw_all_dict across all frames and neurons.
    2. On each session in split_sessions_dict.
    3. On each neuron within each session in split_sessions_dict.
    
    Parameters:
    - c_raw_all_dict: dictionary containing the raw calcium imaging data.
    - split_sessions_dict: dictionary containing session-specific data.
    
    Returns:
    - c_raw_all_dict_z: z-scored data for the entire dataset (c_raw_all_dict).
    - split_sessions_dict_z: z-scored data for each session (split_sessions_dict).
    - split_sessions_dict_z_per_neuron: z-scored data for each neuron per session (split_sessions_dict).
    """

    """ Excluded global way, does not make sense! """
    
    # # 1. Z-score on the entire c_raw_all_dict (over all columns and frames)
    # all_data = np.concatenate([data['C_Raw_all'] for data in c_raw_all_dict.values()], axis=0)
    # mean_all = np.mean(all_data)
    # std_all = np.std(all_data)
    
    # c_raw_all_dict_z = {}
    # for session_key, data in c_raw_all_dict.items():
    #     c_raw_all = data['C_Raw_all']
        
    #     # Apply global z-score normalization
    #     c_raw_all_z = (c_raw_all - mean_all) / std_all
        
    #     # Save in the new dictionary
    #     c_raw_all_dict_z[session_key] = {
    #         'C_Raw_all': c_raw_all_z,
    #         'Start_Frame_Session': data['Start_Frame_Session']
    #     }

    # 2. Z-score on each session independently (across all neurons in the session)
    split_sessions_dict_z = {}
    for session_key, session_data in split_sessions_dict.items():
        mean_session = np.mean(session_data)
        std_session = np.std(session_data)
        
        # Apply session-level z-score normalization
        session_data_z = (session_data - mean_session) / std_session
        
        # Save in the new dictionary
        split_sessions_dict_z[session_key] = session_data_z

    # 3. Z-score on each neuron individually within each session
    split_sessions_dict_z_per_neuron = {}
    for session_key, session_data in split_sessions_dict.items():
        # Apply z-score normalization to each neuron (column)
        session_data_z_per_neuron = (session_data - np.mean(session_data, axis=0)) / np.std(session_data, axis=0)
        
        # Save in the new dictionary
        split_sessions_dict_z_per_neuron[session_key] = session_data_z_per_neuron

    return split_sessions_dict_z, split_sessions_dict_z_per_neuron #c_raw_all_dict_z, 

# Example usage:
# Apply z-score normalization in all three modes
#c_raw_all_dict_z, 
split_sessions_dict_z, split_sessions_dict_z_per_neuron = apply_zscore_all_modes(c_raw_all_dict, split_sessions_dict)

#%%

#check with check_deltaf_f_and_zscore

def check_deltaf_f_and_zscore(session_dict):
    """
    Checks whether deltaF/F and z-score have been applied to the data in session_dict.
    
    Parameters:
    - session_dict: dictionary containing session-specific data (as NumPy arrays).
    
    Returns:
    - Dictionary with session keys and booleans indicating whether deltaF/F or z-score is likely applied.
    """
    results = {}

    for session_key, session_data in session_dict.items():
        # Check for deltaF/F by inspecting if the data has both positive and negative values
        has_positive_values = (session_data > 0).any()
        has_negative_values = (session_data < 0).any()
        deltaf_f_applied = has_positive_values and has_negative_values

        # Check for z-score by inspecting if the data has mean ~0 and standard deviation ~1
        mean_value = np.mean(session_data)
        std_value = np.std(session_data)
        zscore_applied = np.isclose(mean_value, 0, atol=0.1) and np.isclose(std_value, 1, atol=0.1)

        # Store the results for this session
        results[session_key] = {
            'deltaF/F_applied': deltaf_f_applied,
            'zscore_applied': zscore_applied
        }

        # Print results for the session
        print(f"Session {session_key}:")
        print(f"  - DeltaF/F applied: {'Yes' if deltaf_f_applied else 'No'}")
        print(f"  - Z-score applied: {'Yes' if zscore_applied else 'No'}")
        print(f"  - Mean of data: {mean_value:.4f}, Standard deviation of data: {std_value:.4f}")
    
    return results

# Example usage:
# Check for deltaF/F and z-score in split_sessions_dict_z and split_sessions_dict_z_per_neuron
check_deltaf_f_and_zscore(split_sessions_dict_z)

#%%
check_deltaf_f_and_zscore(split_sessions_dict_z_per_neuron)

#%%

""" plot z scored neuron traces"""



def plot_first_10_neurons_separately_same_scale_zscore_skip_nan(session_dict_z, session_key):
    """
    Plots the first 10 neurons' activity (z-scored data) in separate plots for a given session,
    skipping NaN values and using the same y-axis scale (based on the non-NaN min and max).
    
    Parameters:
    - session_dict_z: dictionary containing the z-scored data.
    - session_key: key of the session to plot (e.g., '1022_B_s1').
    """
    if session_key not in session_dict_z:
        print(f"Session {session_key} not found.")
        return

    # Get the z-scored data for the specified session
    session_data_z = session_dict_z[session_key]

    # Limit to the first 10 neurons or however many neurons exist in the session
    num_neurons = min(10, session_data_z.shape[1])

    # Calculate the global min and max across the first 10 neurons, ignoring NaNs
    min_value = np.nanmin(session_data_z[:, :num_neurons])
    max_value = np.nanmax(session_data_z[:, :num_neurons])

    # Create individual plots for each neuron with the same y-axis scale
    for neuron_id in range(num_neurons):
        neuron_data = session_data_z[:, neuron_id]
        
        # Check for NaN values and skip them
        if np.isnan(neuron_data).any():
            print(f"Neuron {neuron_id} in session {session_key} contains NaN values and they will be skipped.")
            neuron_data = neuron_data[~np.isnan(neuron_data)]  # Filter out NaN values

        if neuron_data.size == 0:
            print(f"Neuron {neuron_id} has no valid data points after removing NaN values.")
            continue  # Skip plotting if no valid data remains

        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        plt.plot(neuron_data, label=f'Neuron {neuron_id}')
        
        plt.title(f"Neuron {neuron_id} Activity (Z-Scored) for {session_key}")
        plt.xlabel("Frames")
        plt.ylabel("Z-Scored Activity")
        plt.ylim(min_value, max_value)  # Set y-axis limits based on the global min/max of valid data
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage:
# Plot the first 10 neurons in separate plots for z-scored data in session '1022_B_s1'
plot_first_10_neurons_separately_same_scale_zscore_skip_nan(split_sessions_dict_z_per_neuron, '1022_B_s1')

# Example usage:
# Plot the first 10 neurons in separate plots for z-scored data in session '1022_B_s1'
plot_first_10_neurons_separately_same_scale_zscore(split_sessions_dict_z_per_neuron, '1022_B_s1')


#%%
plot_first_10_neurons_separately_same_scale_zscore(split_sessions_dict_z, '1022_B_s1')
#%%
""" rerun analysis for unnormalized and each 3 z score normalization "ways" to compare """

###################CHETA SHEET1111111111111111111



# Let's break down your request into clear functions, each targeting the specific task you've outlined. I'll guide you through implementing the required functions, step by step.
# 1. Function to get the number of NaNs per neuron in each session

# We need to loop through each session, count the number of NaN values per neuron, and store this information.

# python

# 1. i need a function to get the number of nan per neuron in each session 
# 2.  i need a function to get the mean per neuron in each session 
# 3.  i need a function to get the standard deviation per neuron in each session 

# 4. and want this information saved in a df and a in a csv file wih the headers:  session_name, nr_of_NaNs, std, std_2x (calculate 2 times std), std_3x (calculate 3 times std),- as rows the neurons. 

# 5. go through all sessions and neurons in split_session_dict and check if the neuron has no nan (function1.), then keep it, 1 nan, then replace nan with mean function (2.). remove neurons that have more than 1 nan per session. save as split_session_dict_no_nan

# 6. I now ant a function that takes split_session_dict_no_nan and z score normalizes for all neurons per single session. so for each single session zscore normalize the values in this session for each enuron seperately. 

import numpy as np

def count_nans_per_neuron(split_sessions_dict: dict):
    """Count the number of NaNs per neuron in each session."""
    nan_count_dict = {}
    
    for key, session_data in split_sessions_dict.items():
        nan_count_per_neuron = np.isnan(session_data).sum(axis=0)  # Count NaNs per column (neuron)
        nan_count_dict[key] = nan_count_per_neuron
        
    return nan_count_dict

# 2. Function to get the mean per neuron in each session


def mean_per_neuron(split_sessions_dict: dict):
    """Calculate the mean per neuron in each session, ignoring NaNs."""
    mean_dict = {}
    
    for key, session_data in split_sessions_dict.items():
        mean_per_neuron = np.nanmean(session_data, axis=0)  # Calculate mean, ignoring NaNs
        mean_dict[key] = mean_per_neuron
        
    return mean_dict

# 3. Function to get the standard deviation per neuron in each session

# We’ll calculate the standard deviation per neuron, ignoring NaN values.

# python

def std_per_neuron(split_sessions_dict: dict):
    """Calculate the standard deviation per neuron in each session, ignoring NaNs."""
    std_dict = {}
    
    for key, session_data in split_sessions_dict.items():
        std_per_neuron = np.nanstd(session_data, axis=0)  # Calculate std deviation, ignoring NaNs
        std_dict[key] = std_per_neuron
        
    return std_dict

# 4. Save all this information in a DataFrame and then export to CSV



import pandas as pd

def save_session_stats_to_csv(split_sessions_dict: dict, filename: str):
    """Save session statistics (NaNs, mean, std, 2x std, 3x std) in a DataFrame and export to CSV."""
    data_list = []
    
    nan_counts = count_nans_per_neuron(split_sessions_dict)
    means = mean_per_neuron(split_sessions_dict)
    stds = std_per_neuron(split_sessions_dict)
    
    for session, nan_count in nan_counts.items():
        mean_vals = means[session]
        std_vals = stds[session]
        
        for neuron_idx in range(len(nan_count)):
            row = {
                'session_name': session,
                'nr_of_NaNs': nan_count[neuron_idx],
                'mean': mean_vals[neuron_idx],
                'std': std_vals[neuron_idx],
                'std_2x': std_vals[neuron_idx] * 2,
                'std_3x': std_vals[neuron_idx] * 3
            }
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    df.to_csv(filename, index=False)
    return df

# 5. Processing neurons based on the number of NaNs



def filter_and_replace_nans(split_sessions_dict: dict):
    """Filter neurons with more than 1 NaN and replace NaNs with mean if there is only 1 NaN."""
    split_sessions_dict_no_nan = {}
    
    for session, session_data in split_sessions_dict.items():
        nan_count_per_neuron = np.isnan(session_data).sum(axis=0)
        means = np.nanmean(session_data, axis=0)  # Mean of each neuron, ignoring NaNs
        
        # Filter and replace NaNs
        cleaned_data = []
        for neuron_idx in range(session_data.shape[1]):
            if nan_count_per_neuron[neuron_idx] == 0:
                cleaned_data.append(session_data[:, neuron_idx])  # No NaNs, keep as is
            elif nan_count_per_neuron[neuron_idx] == 1:
                # Replace the single NaN with the mean of that neuron
                neuron_data = session_data[:, neuron_idx]
                neuron_data[np.isnan(neuron_data)] = means[neuron_idx]
                cleaned_data.append(neuron_data)
        
        split_sessions_dict_no_nan[session] = np.column_stack(cleaned_data)
    
    return split_sessions_dict_no_nan

#6. Z-score normalization for each neuron in each session


def z_score_normalization(split_sessions_dict_no_nan: dict):
    """Z-score normalize each neuron in each session."""
    z_score_normalized_dict = {}
    
    for session, session_data in split_sessions_dict_no_nan.items():
        means = np.mean(session_data, axis=0)
        stds = np.std(session_data, axis=0)
        
        z_scored_data = (session_data - means) / stds
        z_score_normalized_dict[session] = z_scored_data
    
    return z_score_normalized_dict

#



def main():
    # Data extraction
    c_raw_all_dict = walk_through_data_and_extract_info_from_matlab_file()
    
    # Data preprocessing 
    split_sessions_dict = split_c_raw_all_into_sessions_dict(c_raw_all_dict)
    
    # Filter neurons and handle NaNs
    split_sessions_dict_no_nan = filter_and_replace_nans(split_sessions_dict)
    
    # Save session stats to CSV
    save_session_stats_to_csv(split_sessions_dict, 'session_stats.csv')
    
    # Z-score normalization
    z_scored_sessions = z_score_normalization(split_sessions_dict_no_nan)

if __name__ == "__main__":
    main()

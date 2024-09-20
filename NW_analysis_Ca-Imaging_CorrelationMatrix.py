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

#%% [1]
""" I.) Acess data from Ca-Imaging, stored in .mat files """

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

#%%
#Check dict and keys
print(c_raw_all_dict)
c_raw_all_dict.keys()

#%%

""" II. Generate dictionary, capturing session-specific info about Ca-Imaging data """
# use "c_raw_all_dict" and split info in key "data_c_raw_all according" to key "start_frame_session"

def split_c_raw_all_into_sessions_dict(c_raw_all_dict):
    """ Function to generate new dictionary by
    splitting C_Raw_all (luminescence matrix) into sessions 
    using start info in Start_Frame_sessions (only start of the session given)"""

    # New dict to store "session-splitted" Ca-Imaging data
    c_raw_all_sessions_dict = {}

    # Loop through each key in the original dict
    for key, data in c_raw_all_dict.items():
        # Extract C_Raw_all and Start_Frame_Session info
        c_raw_all = data['C_Raw_all']
        start_frame_session = data['Start_Frame_Session']

        # Convert the arrays to data frames for easier handling
        df_c_raw_all = pd.DataFrame(c_raw_all)
        df_start_frame_session = pd.DataFrame(start_frame_session).transpose()

        # List to store session keys
        sessions_to_process = []

        # Create session keys and process C_Raw_all into windows (sessions)
        for i in range(1, len(df_start_frame_session.columns)):
            start = int(df_start_frame_session.iloc[0, i-1])
            end = int(df_start_frame_session.iloc[0, i]) - 1

            session_key = f"s{i}_{start}-{end}"
            sessions_to_process.append(session_key)

        # Add the final session
        final_start = int(df_start_frame_session.iloc[0, -1])
        final_end = len(df_c_raw_all) - 1
        final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
        sessions_to_process.append(final_session_key)

        # Dictionary to store each session's data
        session_dict = {}

        # Loop through each session and store corresponding C_Raw_all data
        for i, session_key in enumerate(sessions_to_process):
            if i == 0:
                start = 0
            else:
                start = int(df_start_frame_session.iloc[0, i])

            if i < len(df_start_frame_session.columns) - 1:
                end = int(df_start_frame_session.iloc[0, i+1]) - 1
            else:
                end = len(df_c_raw_all) - 1

            # Store the session in the session_dict
            session_dict[session_key] = df_c_raw_all.iloc[start:end+1, :]

        # Store the session splits and original Start_Frame_Session into the new dictionary
        c_raw_all_sessions_dict[key] = {
            'Start_Frame_Session': start_frame_session,
            'Sessions': session_dict
        }

    return c_raw_all_sessions_dict

#%%
c_raw_all_sessions_dict = split_c_raw_all_into_sessions_dict(c_raw_all_dict)
c_raw_all_sessions_dict.keys()
c_raw_all_sessions_dict.
#%%

#now I have the session accessible for future analysis
""" Info about R+/ R- """
R_plus = [1037, 988, 1002, 936, 970] 

R_minus = [1022, 994, 982, 990, 951, 935, 1012, 991, 955]	














#################################################################################################
########### OLD CODE: #################
#######################################

#%%
#TODO relative path! Traverse (sub)folders
# path_1012 = r"/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/Batch_B/Batch_B_2022_1012_CFC_GPIO/Batch_B_2022_1012_CFC_GPIO/Data_Miniscope_PP.mat"

# """ Extract info from C Raw all """
# with h5py.File(path_1012, 'r') as h5_file:
#     # Access the "Data" group and extract the "C_raw_all" dataset
#     data_c_raw_all = h5_file['Data']['C_Raw_all'][:]
    
# # Convert the data  for 'C_Raw_all' and start frames to df
# df_c_raw_all = pd.DataFrame(data_c_raw_all)
# """ Extract info from start Frame Sessions """
# with h5py.File(path_1012, 'r') as h5_file:
#     # Access the "Data" group and extract the "C_raw_all" dataset from it
#     start_frame_session = h5_file['Data']['Start_Frame_Session'][:]

# # Convert the data  for 'C_Raw_all' and start frames to df
# df_start_frame_session = pd.DataFrame(start_frame_session)
# # header, nr of rows/columns of the df

# print(f"C_Raw_all: \n{df_c_raw_all.head()}")
# rows, cols = df_c_raw_all.shape
# print(f"\nNumber of rows: {rows}")
# print(f"Number of columns: {cols}")
# print(f"Start Frame Sessions: \n{df_start_frame_session.head()}")
# %%
# #%% [2]
# """ 2. Session Key Generation

# Generate sessions_to_process list automatically """

# # initialize list 
# sessions_to_process = []

# # Loop through the start_frame_session
# for i in range(1, len(df_start_frame_session.columns)):  # Start from session 1 
#     start = int(df_start_frame_session.iloc[0, i-1])
#     end = int(df_start_frame_session.iloc[0, i]) - 1
    
#     # Create session key 
#     session_key = f"s{i}_{start}-{end}"
#     sessions_to_process.append(session_key)

# # Add the final session (->goes until the last row of df_c_raw_all)
# final_start = int(df_start_frame_session.iloc[0, -1])
# final_end = len(df_c_raw_all) - 1
# final_session_key = f"s{len(df_start_frame_session.columns)}_{final_start}-{final_end}"
# sessions_to_process.append(final_session_key)

# print("Generated sessions_to_process:")
# print(sessions_to_process)

# #%%
# """ 3. Create windows and store them in a dictionary """

# # dictto store the windows
# c_raw_all_sessions = {}

# # Loop through each start frame session; create windows
# for i in range(len(df_start_frame_session.columns)):
#     if i == 0:
#         # First window starts at index 0
#         start = 0
#     else:
#         # Subsequent windows start at the current session index
#         start = int(df_start_frame_session.iloc[0, i])

#     # If not the last index, 
#     #take the next start and subtract 1
#     if i < len(df_start_frame_session.columns) - 1:
#         end = int(df_start_frame_session.iloc[0, i+1]) - 1
#     else:
#         # Last window ends at the last row of df_c_raw_all
#         end = rows - 1
    
#     # Create a key like 's1_0-22485', 's2_22486-44647'...
#     key = f"s{i+1}_{start}-{end}"
    
#     # Store corresponding rows in the dictionary
#     c_raw_all_sessions[key] = df_c_raw_all.iloc[start:end+1, :]

# # check 
# for key, df in c_raw_all_sessions.items():
#     print(f"{key}: {df.shape}")

#%%

""" 4. Use extinction days in  c_raw_all_sessions for network analysis

s4_91943-124993: (33051, 271)
s5_124994-158590: (33597, 271)
s6_158591-191577: (32987, 271)
s7_191578-225080: (33503, 271)"""


key_s4 = 's4_91943-124993'
key_s5 = 's5_124994-158590'
key_s6 = 's6_158591-191577'
key_s7 = 's7_191578-225080'

activity_s4 = c_raw_all_sessions[key_s4]
activity_s5 = c_raw_all_sessions[key_s5]
activity_s6 = c_raw_all_sessions[key_s6]
activity_s7 = c_raw_all_sessions[key_s7]


#%%
#TODO write function for nw generation!

# def plot_and_analyse_nw(activity_df, corr_th=0.2, seed=12345678):

    
#     # Step 1: Drop columns with NaN values - all!
#     activity_df_non_nan = activity_df.dropna(how='any', axis=1)

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

#     # Optional: Analyze graph metrics
#     num_edges = G.number_of_edges()
#     num_nodes = G.number_of_nodes()
    
#     # Output graph metrics
#     print(f"Number of edges: {num_edges}")
#     print(f"Number of nodes: {num_nodes}")
    
#     #return G, num_edges, num_nodes
#     # Step 7: Compute mean degree of the whole network
#     degrees = [degree for node, degree in G.degree()]
#     mean_degree = np.mean(degrees)
#     print(f"Mean degree of the whole network: {mean_degree}")
    
#     # Step 8: Calculate the number of sub-networks (connected components)
#     components = list(nx.connected_components(G))
#     num_sub_networks = len(components)
#     print(f"Number of sub-networks: {num_sub_networks}")
    
#     # Step 9: Compute mean degree of sub-networks (connected components)
#     # sub_network_degrees = []
#     # for component in components:
#     #     subgraph = G.subgraph(component)
#     #     sub_network_degrees.append(np.mean([degree for node, degree in subgraph.degree()]))
    
#     # if sub_network_degrees:
#     #     mean_sub_network_degree = np.mean(sub_network_degrees)
#     #     print(f"Mean degree per sub-network: {mean_sub_network_degree}")
    
#     # Step 10: Compute assortativity (degree correlation)
#     assortativity = nx.degree_assortativity_coefficient(G)
#     print(f"Assortativity (degree correlation): {assortativity}")

#     # Step 11: Compute clustering coefficient
#     #clustering_coefficient = nx.average_clustering(G)
#     #print(f"Average clustering coefficient: {clustering_coefficient}")
    
#     return G, num_edges, num_nodes, mean_degree, assortativity
# #, mean_betweenness, mean_sub_network_degree, clustering_coefficient

#%%
#default corr_th=0.2
plot_and_analyse_nw(activity_s4)

#%%
plot_and_analyse_nw(activity_s5)

#%%

plot_and_analyse_nw(activity_s6)

#%%
plot_and_analyse_nw(activity_s7)

# %%
#corr_th=0.5
plot_nw(activity_s4, corr_th=0.5)
plot_nw(activity_s5, corr_th=0.5)
plot_nw(activity_s6, corr_th=0.5)
plot_nw(activity_s7, corr_th=0.5)
# %%

# %%
def plot_and_analyse_nw(activity_df, corr_th=0.2, seed=12345678):

    
    # Step 1: Drop columns with NaN values - all!
    activity_df_non_nan = activity_df.dropna(how='any', axis=1)

    # Step 2: Compute correlation matrix
    corr_mat = np.corrcoef(activity_df_non_nan.T)  # Transpose so that neurons are along columns

    # Step 3: Zero diagonal elements to ignore self-connections
    np.fill_diagonal(corr_mat, 0)

    # Step 4: Zero out connections below the threshold
    corr_mat[corr_mat < corr_th] = 0

    # Step 5: Create a graph from the correlation matrix
    G = nx.from_numpy_array(corr_mat)

    # Step 6: Visualize the graph using a spring layout
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=nx.spring_layout(G, seed=seed), node_size=10, width=0.3)
    plt.title(f"Neuronal Network (Threshold {corr_th})")

    plt.show()

    # Optional: Analyze graph metrics
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    
    # Output graph metrics
    print(f"Number of edges: {num_edges}")
    print(f"Number of nodes: {num_nodes}")
    
    #return G, num_edges, num_nodes
    # Step 7: Compute mean degree of the whole network
    degrees = [degree for node, degree in G.degree()]
    mean_degree = np.mean(degrees)
    print(f"Mean degree of the whole network: {mean_degree}")
    
    # Step 8: Calculate the number of sub-networks (connected components)
    components = list(nx.connected_components(G))
    num_sub_networks = len(components)
    print(f"Number of sub-networks: {num_sub_networks}")
    
    # Step 9: Compute mean degree of sub-networks (connected components)
    # sub_network_degrees = []
    # for component in components:
    #     subgraph = G.subgraph(component)
    #     sub_network_degrees.append(np.mean([degree for node, degree in subgraph.degree()]))
    
    # if sub_network_degrees:
    #     mean_sub_network_degree = np.mean(sub_network_degrees)
    #     print(f"Mean degree per sub-network: {mean_sub_network_degree}")
    
    # Step 10: Compute assortativity (degree correlation)
    assortativity = nx.degree_assortativity_coefficient(G)
    print(f"Assortativity (degree correlation): {assortativity}")

    # Step 11: Compute clustering coefficient
    #clustering_coefficient = nx.average_clustering(G)
    #print(f"Average clustering coefficient: {clustering_coefficient}")
    
    return G, num_edges, num_nodes, mean_degree, assortativity

#%%
    #default corr_th=0.2

""" NW s4 """
plot_and_analyse_nw(activity_s4)

#%%
plot_and_analyse_nw(activity_s5)
plot_and_analyse_nw(activity_s6)
plot_and_analyse_nw(activity_s7)

# %%

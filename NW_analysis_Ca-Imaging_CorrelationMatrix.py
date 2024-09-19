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
# Display for one key
# for key, data in c_raw_all_sessions_dict.items():
#     print(f"Key: {key}")
#     print(f"Start Frame Session: {data['Start_Frame_Session']}")
    
#     for session_key, session_data in data['Sessions'].items():
#         print(f"Session {session_key}: {session_data.shape}")
    #break  # only for the first key
#%%

c_raw_all_sessions_dict = split_c_raw_all_into_sessions_dict()

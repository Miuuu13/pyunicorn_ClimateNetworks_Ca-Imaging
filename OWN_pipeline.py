

""" Network analysis for one animal id (Batch A or B)"""
#%% #imports
import os
import h5py
import pandas as pd

#%% [1]
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

#%% [2]




#%% [3]

#%% [3]

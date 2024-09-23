

#%%
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

# Define the file path
file_path = "/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/Batch_B/Batch_B_2022_1002_CFC_GPIO/Batch_B_2022_1002_CFC_GPIO/Data_Miniscope_PP.mat"
#%%
# Open the .mat file and extract the C_raw_all data
with h5py.File(file_path, 'r') as f:
    if 'Data' in f and 'C_Raw_all' in f['Data']:
        # Extract the C_raw_all data
        c_raw_all = f['Data']['C_Raw_all'][:]

# Convert the extracted data into a DataFrame
c_raw_all_df = pd.DataFrame(c_raw_all)

# Display the number of columns and rows
rows, columns = c_raw_all_df.shape
print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")

"""z score"""


c_raw_all_df_zscored = c_raw_all_df.apply(zscore)

# Display the first few rows
print(c_raw_all_df_zscored.head())

#%%

"""plot 3 neurons in window"""

frame_start = 100
frame_end = 150
selected_neurons = [1, 2, 41]  # Change these indices if you want different neurons

# Plot the selected neurons over the specified frame range
plt.figure(figsize=(10, 6))
for neuron in selected_neurons:
    plt.plot(c_raw_all_df_zscored.iloc[frame_start:frame_end, neuron], label=f'Neuron {neuron + 1}')

# Add labels and title
plt.xlabel('Frame')
plt.ylabel('Z-score')
plt.title('Neuronal Activity Over Time')
plt.legend()
plt.show()

#%%

"y achsis"

# # Define the file path
# file_path = "/path/to/your/Data_Miniscope_PP.mat"  # Replace with the correct path

# # Open the .mat file and extract the C_raw_all data
# with h5py.File(file_path, 'r') as f:
#     # Check and extract the data correctly
#     if 'Data' in f and 'C_Raw_all' in f['Data']:
#         c_raw_all = np.array(f['Data']['C_Raw_all']).T  # Transpose if needed

# # Convert the extracted data into a DataFrame
# c_raw_all_df = pd.DataFrame(c_raw_all)

# # Apply z-scoring to each column (neuron)
# c_raw_all_df_zscored = c_raw_all_df.apply(zscore)

# # Define the frame range and neurons to plot
# frame_start = 100
# frame_end = 150
# selected_neurons = [0, 1, 2]  # Change these indices if you want different neurons

# # Plot the selected neurons over the specified frame range
# plt.figure(figsize=(10, 6))
# for neuron in selected_neurons:
#     plt.plot(c_raw_all_df_zscored.iloc[frame_start:frame_end, neuron], label=f'Neuron {neuron + 1}')

# # Adjusting y-axis automatically to fit all data
# plt.ylim(c_raw_all_df_zscored.iloc[frame_start:frame_end, selected_neurons].min().min(),
#          c_raw_all_df_zscored.iloc[frame_start:frame_end, selected_neurons].max().max())

# # Add labels and title
# plt.xlabel('Frame')
# plt.ylabel('Z-score')
# plt.title('Neuronal Activity Over Time')
# plt.legend()
# plt.show()










#%%
############

#c_raw_all_zscored_df.to_csv('/home/manuela/Documents/PROJECT_NW_ANALYSIS_Ca-IMAGING_SEP24/data/c_raw_all_zscored.csv', index=False)
#%%
import matplotlib.pyplot as plt
import numpy as np

def find_and_plot_neuron_triplet(dataframe, threshold=2, window_size=10, time_offset=10, correlation_threshold=0.8):
    """
    Find two neurons with similar action potential shapes and a third neuron that is inactive
    during their activation window but activates 10 frames later. Plot them side-by-side.
    
    Parameters:
    - dataframe: DataFrame containing the z-scored neuron data (rows: frames/time points, columns: neurons)
    - threshold: Value above which a neuron is considered to have an action potential
    - window_size: Number of consecutive frames over which neurons should show similar activity
    - time_offset: The number of frames later that the third neuron should activate
    - correlation_threshold: Minimum correlation value to consider two neurons' activity as "similar"
    """
    # Find two neurons with similar action potentials in a 10-frame window
    similar_neuron_pair = None
    num_frames = dataframe.shape[0]
    num_neurons = dataframe.shape[1]
    
    for neuron1_index in range(num_neurons):
        for neuron2_index in range(neuron1_index + 1, num_neurons):
            neuron1_signal = dataframe.iloc[:, neuron1_index]
            neuron2_signal = dataframe.iloc[:, neuron2_index]
            
            # Look for a 10-frame window where the correlation is high
            for start_frame in range(num_frames - window_size):
                end_frame = start_frame + window_size
                if (neuron1_signal[start_frame:end_frame] > threshold).any() and \
                   (neuron2_signal[start_frame:end_frame] > threshold).any():
                    corr = np.corrcoef(neuron1_signal[start_frame:end_frame],
                                       neuron2_signal[start_frame:end_frame])[0, 1]
                    if corr >= correlation_threshold:
                        similar_neuron_pair = (neuron1_index, neuron2_index, start_frame, end_frame)
                        break
            if similar_neuron_pair:
                break
        if similar_neuron_pair:
            break

    if not similar_neuron_pair:
        print("No similar neuron pair found.")
        return
    
    # Extract details of the found neuron pair and the frame window
    neuron1_index, neuron2_index, start_frame, end_frame = similar_neuron_pair

    # Find a third neuron that is inactive during this window but active 10 frames later
    inactive_neuron_index = None
    for neuron_index in range(num_neurons):
        if neuron_index in [neuron1_index, neuron2_index]:
            continue
        neuron_signal = dataframe.iloc[start_frame:end_frame, neuron_index]
        if (neuron_signal.abs() < 0.5).all():
            later_signal = dataframe.iloc[end_frame + time_offset:end_frame + time_offset + window_size, neuron_index]
            if (later_signal > threshold).any():
                inactive_neuron_index = neuron_index
                break

    if inactive_neuron_index is None:
        print("No suitable third neuron found.")
        return
    
    # Collect indices for plotting
    neuron_indices = [neuron1_index, neuron2_index, inactive_neuron_index]

    # Define a common y-axis range
    y_min = dataframe.iloc[start_frame:end_frame + time_offset + window_size, neuron_indices].min().min()
    y_max = dataframe.iloc[start_frame:end_frame + time_offset + window_size, neuron_indices].max().max()

    # Create a figure for subplots arranged side-by-side with a consistent y-axis
    plt.figure(figsize=(20, 5))

    # Plot each neuron's signal in a separate subplot arranged horizontally
    for i, neuron_index in enumerate(neuron_indices):
        if i < 2:
            neuron_signal = dataframe.iloc[start_frame:end_frame + 1, neuron_index]
        else:
            neuron_signal = dataframe.iloc[end_frame + time_offset:end_frame + time_offset + window_size + 1, neuron_index]
        
        plt.subplot(1, 3, i + 1)
        plt.plot(range(len(neuron_signal)), neuron_signal, marker='o')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
        plt.ylim([y_min, y_max])  # Set the same y-axis range for all subplots
        plt.title(f"Neuron {neuron_index + 1} Signal")
        plt.xlabel("Time (Frames)")
        plt.ylabel("Z-Scored Signal Intensity")
        plt.grid(True)
        plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    print(f"Plotted neurons: {neuron1_index + 1}, {neuron2_index + 1}, {inactive_neuron_index + 1}")

# Example usage: run this multiple times to find different triplet plots
find_and_plot_neuron_triplet(c_raw_all_df_zscored)

#%%
import matplotlib.pyplot as plt

# Define the threshold and the time range
threshold = 1.5
time_start = 50
time_end = time_start +30

# Find neurons that meet the criteria
neurons_meeting_criteria = []
for neuron_index in range(c_raw_all_df_zscored.shape[1]):
    neuron_signal = c_raw_all_df_zscored.iloc[time_start:time_end + 1, neuron_index]
    # Check if the neuron crosses the threshold for at least 10 consecutive frames
    above_threshold = (neuron_signal > threshold).astype(int)
    consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
    if consecutive_frames.max() == 10:
        neurons_meeting_criteria.append(neuron_index)
    # Break when we have enough neurons
    if len(neurons_meeting_criteria) == 2:
        break

# Select a neuron that doesn't cross the threshold within the first 50 frames but does after
third_neuron_index = None
for neuron_index in range(c_raw_all_df_zscored.shape[1]):
    neuron_signal = c_raw_all_df_zscored.iloc[:time_end + 10, neuron_index]
    above_threshold = (neuron_signal > threshold).astype(int)
    consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
    if consecutive_frames.max() < 10:
        # Check after the time window if it crosses the threshold for at least 10 frames
        neuron_signal_post = c_raw_all_df_zscored.iloc[time_end + 10:, neuron_index]
        above_threshold_post = (neuron_signal_post > threshold).astype(int)
        consecutive_frames_post = above_threshold_post.rolling(window=10, min_periods=10).sum()
        if consecutive_frames_post.max() == 10:
            third_neuron_index = neuron_index
            break

# Collect the indices for plotting
neuron_indices = neurons_meeting_criteria + [third_neuron_index]

# Define a common y-axis range
y_min = min(c_raw_all_df_zscored.iloc[time_start:time_end + 1, neuron_indices].min().min(), 1.5) - 0.5
y_max = max(c_raw_all_df_zscored.iloc[time_start:time_end + 1, neuron_indices].max().max(), threshold) + 0.5

# Create a figure for subplots arranged side-by-side with a consistent y-axis
plt.figure(figsize=(20, 5))

# Plot each neuron's signal in a separate subplot arranged horizontally
for i, neuron_index in enumerate(neuron_indices):
    # Extract the data for the specified neuron over the given time frame
    neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]
    
    # Create a subplot for each neuron in a horizontal arrangement
    plt.subplot(1, 3, i + 1)
    plt.plot(range(time_start, time_end + 1), neuron_signal, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
    plt.ylim([y_min, y_max])  # Set the same y-axis range for all subplots
    plt.title(f"Signal for Neuron {neuron_index + 1} from Frame {time_start} to {time_end}")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Signal Intensity")
    plt.grid(True)
    plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



##########
#%%


#%%
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
window_size = 10
threshold = 2  # Define a threshold to detect action potential
time_offset = 10  # The third neuron should be active 10 frames later

# Find two neurons with similar action potentials in a 10-frame window
similar_neuron_pair = None
for neuron1_index in range(c_raw_all_zscored_df.shape[1]):
    for neuron2_index in range(neuron1_index + 1, c_raw_all_zscored_df.shape[1]):
        # Extract the signals of the two neurons over the entire frame range
        neuron1_signal = c_raw_all_zscored_df.iloc[:, neuron1_index]
        neuron2_signal = c_raw_all_zscored_df.iloc[:, neuron2_index]
        
        # Look for a 10-frame window where the correlation is high
        for start_frame in range(len(neuron1_signal) - window_size):
            end_frame = start_frame + window_size
            # Check if both neurons exceed the threshold in this window
            if (neuron1_signal[start_frame:end_frame] > threshold).any() and \
               (neuron2_signal[start_frame:end_frame] > threshold).any():
                # Check similarity by correlation
                corr = np.corrcoef(neuron1_signal[start_frame:end_frame],
                                   neuron2_signal[start_frame:end_frame])[0, 1]
                if corr > 0.8:  # You can adjust this threshold for similarity
                    similar_neuron_pair = (neuron1_index, neuron2_index, start_frame, end_frame)
                    break
        if similar_neuron_pair:
            break
    if similar_neuron_pair:
        break

# Extract details of the found neuron pair and the frame window
neuron1_index, neuron2_index, start_frame, end_frame = similar_neuron_pair

# Find a third neuron that is inactive during this window but active 10 frames later
inactive_neuron_index = None
for neuron_index in range(c_raw_all_zscored_df.shape[1]):
    if neuron_index == neuron1_index or neuron_index == neuron2_index:
        continue
    # Check if inactive during the window
    neuron_signal = c_raw_all_zscored_df.iloc[start_frame:end_frame, neuron_index]
    if (neuron_signal.abs() < 0.5).all():  # Adjust threshold to determine inactivity
        # Check if active 10 frames later
        later_signal = c_raw_all_zscored_df.iloc[end_frame + time_offset:end_frame + time_offset + window_size, neuron_index]
        if (later_signal > threshold).any():
            inactive_neuron_index = neuron_index
            break

# Collect indices for plotting
neuron_indices = [neuron1_index, neuron2_index, inactive_neuron_index]

# Define a common y-axis range
y_min = c_raw_all_zscored_df.iloc[start_frame:end_frame + time_offset + window_size, neuron_indices].min().min()
y_max = c_raw_all_zscored_df.iloc[start_frame:end_frame + time_offset + window_size, neuron_indices].max().max()

# Create a figure for subplots arranged side-by-side with a consistent y-axis
plt.figure(figsize=(20, 5))

# Plot each neuron's signal in a separate subplot arranged horizontally
for i, neuron_index in enumerate(neuron_indices):
    if i < 2:
        neuron_signal = c_raw_all_zscored_df.iloc[start_frame:end_frame + 1, neuron_index]
    else:
        neuron_signal = c_raw_all_zscored_df.iloc[end_frame + time_offset:end_frame + time_offset + window_size + 1, neuron_index]
    
    plt.subplot(1, 3, i + 1)
    plt.plot(range(len(neuron_signal)), neuron_signal, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
    plt.ylim([y_min, y_max])  # Set the same y-axis range for all subplots
    plt.title(f"Neuron {neuron_index + 1} Signal")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Z-Scored Signal Intensity")
    plt.grid(True)
    plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


















####################

# %%
import matplotlib.pyplot as plt


neuron_index = 0  #id of the neuron
time_start = 20
time_end = 50


neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]


plt.figure(figsize=(10, 6))
plt.plot(range(time_start, time_end + 1), neuron_signal, marker='o')
plt.title(f"Signal for Neuron {neuron_index + 1} from Frame {time_start} to {time_end}")
plt.xlabel("Time (Frames)")
plt.ylabel("Signal Intensity")
plt.grid(True)
plt.show()
# %%
#3 plots
# Selecting the indices of the three neurons to plot
neuron_indices = [0, 1, 2]  # You can change these indices to plot different neurons
time_start = 10
time_end = 50


plt.figure(figsize=(10, 15))


for i, neuron_index in enumerate(neuron_indices):

    neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]
    

    plt.subplot(3, 1, i + 1)
    plt.plot(range(time_start, time_end + 1), neuron_signal, marker='o')
    plt.title(f"Signal for Neuron {neuron_index + 1} from Frame {time_start} to {time_end}")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Signal Intensity")
    plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

#%%

""" Find to neurons crossing T for 10 frames
third neuron crosses T  10 frames or later and not at same time """

import matplotlib.pyplot as plt

# Define the threshold and the time range
threshold = 2
time_start = 10
time_end = 50

# Find neurons that meet the criteria
neurons_meeting_criteria = []
for neuron_index in range(c_raw_all_df.shape[1]):
    neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]
    # Check if the neuron crosses the threshold for at least 10 consecutive frames
    above_threshold = (neuron_signal > threshold).astype(int)
    consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
    if consecutive_frames.max() == 10:
        neurons_meeting_criteria.append(neuron_index)
    # Break when we have enough neurons
    if len(neurons_meeting_criteria) == 2:
        break

# Select a neuron that doesn't cross the threshold within the first 50 frames but does after
third_neuron_index = None
for neuron_index in range(c_raw_all_df.shape[1]):
    neuron_signal = c_raw_all_df.iloc[:time_end + 10, neuron_index]
    above_threshold = (neuron_signal > threshold).astype(int)
    consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
    if consecutive_frames.max() < 10:
        # Check after the time window if it crosses the threshold for at least 10 frames
        neuron_signal_post = c_raw_all_df.iloc[time_end + 10:, neuron_index]
        above_threshold_post = (neuron_signal_post > threshold).astype(int)
        consecutive_frames_post = above_threshold_post.rolling(window=10, min_periods=10).sum()
        if consecutive_frames_post.max() == 10:
            third_neuron_index = neuron_index
            break

# Collect the indices for plotting
neuron_indices = neurons_meeting_criteria + [third_neuron_index]

# Create a figure for subplots
plt.figure(figsize=(10, 15))

# Plot each neuron's signal in a separate subplot
for i, neuron_index in enumerate(neuron_indices):
    # Extract the data for the specified neuron over the given time frame
    neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]
    
    # Create a subplot for each neuron
    plt.subplot(3, 1, i + 1)
    plt.plot(range(time_start, time_end + 1), neuron_signal, marker='o')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
    plt.title(f"Signal for Neuron {neuron_index + 1} from Frame {time_start} to {time_end}")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Signal Intensity")
    plt.grid(True)
    plt.legend()


plt.tight_layout()
plt.show()

# %%

""" try different values """
""" Find to neurons crossing T for 10 frames
third neuron crosses T  10 frames or later and not at same time """

import matplotlib.pyplot as plt

# T and window def
threshold = 2
time_start = 10
time_end = 50

# Find neurons that meet the criteria
neurons_meeting_criteria = []
for neuron_index in range(c_raw_all_df.shape[1]):
    neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]
    # Check if the neuron crosses the threshold for at l
    # east 10 consecutive frames
    above_threshold = (neuron_signal > threshold).astype(int)
    consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
    if consecutive_frames.max() == 10:
        neurons_meeting_criteria.append(neuron_index)
    # Break have enough neurons
    if len(neurons_meeting_criteria) == 2:
        break

# Select a neuron that doesn't cross the threshold within 
# the first 50 frames but does after
third_neuron_index = None
for neuron_index in range(c_raw_all_df.shape[1]):
    neuron_signal = c_raw_all_df.iloc[:time_end + 10, neuron_index]
    above_threshold = (neuron_signal > threshold).astype(int)
    consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
    if consecutive_frames.max() < 10:
        # Check after the time window if it crosses the threshold for at least 10 frames
        neuron_signal_post = c_raw_all_df.iloc[time_end + 10:, neuron_index]
        above_threshold_post = (neuron_signal_post > threshold).astype(int)
        consecutive_frames_post = above_threshold_post.rolling(window=10, min_periods=10).sum()
        if consecutive_frames_post.max() == 10:
            third_neuron_index = neuron_index
            break

# Collect the indices for plotting
neuron_indices = neurons_meeting_criteria + [third_neuron_index]

# Create a figure for subplots
plt.figure(figsize=(20, 5))


for i, neuron_index in enumerate(neuron_indices):

    neuron_signal = c_raw_all_df.iloc[time_start:time_end + 1, neuron_index]
    

    #plt.subplot(3, 1, i + 1) # above
    plt.subplot(1, 3, i + 1) # bedide
    plt.plot(range(time_start, time_end + 1), neuron_signal, marker='o')
    #would plot T a line
    #plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
    plt.title(f"Signal for Neuron {neuron_index + 1} from Frame {time_start} to {time_end}")
    plt.xlabel("Time (Frames)")
    plt.ylabel("Signal Intensity")
    plt.grid(True)
    plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
# %%
"""
when searching the window where the first and second neuron 
cross T together for 10 frame, 
-> remember the frames. plot 5 frames before, the 10 frames rmembered, 
then 20 frames after it. 
the 3rd neuron should stay at Threshold - 0.5 (so in this case 1.5) 
for the first 15 frames. 
then there should be a clear Threshold crossing 10 frames after 
the remembered window for neuron 1 and 2"""

# import matplotlib.pyplot as plt

# # Define the threshold and time window parameters
# threshold = 2
# threshold_adjusted = threshold - 0.5  # For the third neuron
# pre_frames = 5
# post_frames = 20

# # Find the first window where both neurons cross the threshold together for at least 10 consecutive frames
# crossing_window_start = None
# for frame in range(c_raw_all_df.shape[0] - 10):
#     # Check if neurons 1 and 2 cross the threshold together for 10 consecutive frames
#     neuron_1_signal = c_raw_all_df.iloc[frame:frame + 10, neurons_meeting_criteria[0]]
#     neuron_2_signal = c_raw_all_df.iloc[frame:frame + 10, neurons_meeting_criteria[1]]
#     if all(neuron_1_signal > threshold) and all(neuron_2_signal > threshold):
#         crossing_window_start = frame
#         break

# # Define the full time window for plotting
# start_frame = max(0, crossing_window_start - pre_frames)
# end_frame = crossing_window_start + 10 + post_frames

# # Create a figure for subplots arranged side-by-side
# plt.figure(figsize=(20, 5))

# # Plot the signals for neurons 1 and 2 within the defined window
# for i, neuron_index in enumerate(neurons_meeting_criteria[:2]):
#     neuron_signal = c_raw_all_df.iloc[start_frame:end_frame + 1, neuron_index]
    
#     # Create a subplot for each neuron
#     plt.subplot(1, 3, i + 1)
#     plt.plot(range(start_frame, end_frame + 1), neuron_signal, marker='o')
#     plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
#     plt.title(f"Neuron {neuron_index + 1} Signal from Frame {start_frame} to {end_frame}")
#     plt.xlabel("Time (Frames)")
#     plt.ylabel("Signal Intensity")
#     plt.grid(True)
#     plt.legend()

# # For the third neuron, make it stay around threshold_adjusted and cross the threshold after 10 frames from the remembered window
# neuron_3_signal = [threshold_adjusted] * 15  # Staying below threshold for 15 frames
# cross_frame = crossing_window_start + 10 + 10  # Crossing after 10 frames from the remembered window
# remaining_frames = (end_frame - start_frame + 1) - 15  # Remaining frames for crossing
# neuron_3_crossing_signal = c_raw_all_df.iloc[cross_frame:cross_frame + remaining_frames, third_neuron_index].tolist()

# # Combine the signals for the third neuron
# neuron_3_signal.extend(neuron_3_crossing_signal)

# # Plot the third neuron's signal in the last subplot
# plt.subplot(1, 3, 3)
# plt.plot(range(start_frame, end_frame + 1), neuron_3_signal, marker='o', label=f"Neuron {third_neuron_index + 1} Signal")
# plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
# plt.title(f"Neuron {third_neuron_index + 1} Signal from Frame {start_frame} to {end_frame}")
# plt.xlabel("Time (Frames)")
# plt.ylabel("Signal Intensity")
# plt.grid(True)
# plt.legend()

# # Adjust layout to prevent overlap
# plt.tight_layout()
# plt.show()

# # %%
# """ more combis"""
# import matplotlib.pyplot as plt
# import random

# def find_neuron_windows(dataframe, threshold=2, pre_frames=5, post_frames=20, num_plots=10):
    
#     plots = []
    

#     while len(plots) < num_plots:
#         # Find neurons that meet the criteria for the first two neurons crossing threshold
#         neurons_meeting_criteria = []
#         for neuron_index in range(dataframe.shape[1]):
#             neuron_signal = dataframe.iloc[:, neuron_index]
#             # Check if the neuron crosses the threshold for at least 10 consecutive frames
#             above_threshold = (neuron_signal > threshold).astype(int)
#             consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
#             if consecutive_frames.max() == 10:
#                 neurons_meeting_criteria.append(neuron_index)
#             if len(neurons_meeting_criteria) == 2:
#                 break

#         # Select a neuron that doesn't cross the threshold within the first 50 frames but does after
#         third_neuron_index = None
#         for neuron_index in range(dataframe.shape[1]):
#             if neuron_index in neurons_meeting_criteria:
#                 continue
#             neuron_signal = dataframe.iloc[:, neuron_index]
#             above_threshold = (neuron_signal > threshold).astype(int)
#             consecutive_frames = above_threshold.rolling(window=10, min_periods=10).sum()
#             if consecutive_frames.max() < 10:
#                 # Check after the threshold window
#                 neuron_signal_post = dataframe.iloc[50:, neuron_index]
#                 above_threshold_post = (neuron_signal_post > threshold).astype(int)
#                 consecutive_frames_post = above_threshold_post.rolling(window=10, min_periods=10).sum()
#                 if consecutive_frames_post.max() == 10:
#                     third_neuron_index = neuron_index
#                     break
        
#         # If suitable neurons are found, find the first window where both neurons cross the threshold together for 10 frames
#         crossing_window_start = None
#         for frame in range(dataframe.shape[0] - 10):
#             neuron_1_signal = dataframe.iloc[frame:frame + 10, neurons_meeting_criteria[0]]
#             neuron_2_signal = dataframe.iloc[frame:frame + 10, neurons_meeting_criteria[1]]
#             if all(neuron_1_signal > threshold) and all(neuron_2_signal > threshold):
#                 crossing_window_start = frame
#                 break

#         if crossing_window_start is None or third_neuron_index is None:
#             continue

#         # Define the full time window for plotting
#         start_frame = max(0, crossing_window_start - pre_frames)
#         end_frame = crossing_window_start + 10 + post_frames

#         # Save the successful plot configuration
#         plots.append((neurons_meeting_criteria, third_neuron_index, start_frame, end_frame))
    
#     return plots

# # Example usage of the function
# plots_info = find_neuron_windows(c_raw_all_df, threshold=2)

# def plot_neuron_windows(dataframe, plots_info, threshold=2):
#     for plot_num, (neurons_meeting_criteria, third_neuron_index, start_frame, end_frame) in enumerate(plots_info):
#         plt.figure(figsize=(20, 5))
        
#         # Plot signals for the first two neurons
#         for i, neuron_index in enumerate(neurons_meeting_criteria):
#             neuron_signal = dataframe.iloc[start_frame:end_frame + 1, neuron_index]
#             plt.subplot(1, 3, i + 1)
#             plt.plot(range(start_frame, end_frame + 1), neuron_signal, marker='o')
#             plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
#             plt.title(f"Neuron {neuron_index + 1} Signal from Frame {start_frame} to {end_frame}")
#             plt.xlabel("Time (Frames)")
#             plt.ylabel("Signal Intensity")
#             plt.grid(True)
#             plt.legend()
        
#         # For the third neuron, make it stay around threshold_adjusted and cross the threshold after 10 frames from the remembered window
#         neuron_3_signal = [threshold - 0.5] * 15  # Staying below threshold for 15 frames
#         cross_frame = crossing_window_start + 10 + 10  # Crossing after 10 frames from the remembered window
#         remaining_frames = (end_frame - start_frame + 1) - 15  # Remaining frames for crossing
#         neuron_3_crossing_signal = dataframe.iloc[cross_frame:cross_frame + remaining_frames, third_neuron_index].tolist()

#         # Combine the signals for the third neuron
#         neuron_3_signal.extend(neuron_3_crossing_signal)

#         # Plot the third neuron's signal in the last subplot
#         plt.subplot(1, 3, 3)
#         plt.plot(range(start_frame, end_frame + 1), neuron_3_signal, marker='o', label=f"Neuron {third_neuron_index + 1} Signal")
#         plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold = 2')
#         plt.title(f"Neuron {third_neuron_index + 1} Signal from Frame {start_frame} to {end_frame}")
#         plt.xlabel("Time (Frames)")
#         plt.ylabel("Signal Intensity")
#         plt.grid(True)
#         plt.legend()

#         # Adjust layout to prevent overlap
#         plt.tight_layout()
#         plt.show()

# # Call the plotting function to visualize the 10 plots
# plot_neuron_windows(c_raw_all_df, plots_info)
# %%














""" Luminescence trace for poster """
#%%
import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt

# Define the relative path of the file in your cwd
cwd = os.getcwd()
relative_path = r"data/Batch_B/Batch_B_2022_935_CFC_GPIO/Data_Miniscope_PP.mat"
file_path = os.path.join(cwd, relative_path)

def load_c_raw_all_to_dataframe(file_path):
    """
    Load 'C_raw_all' from the HDF5 .mat file and convert it to a Pandas DataFrame.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'Data' in f and 'C_Raw_all' in f['Data']:
                # Extract the C_raw_all data
                c_raw_all = f['Data']['C_Raw_all'][:]
                
                # Convert to DataFrame
                df = pd.DataFrame(c_raw_all.T)
                
                # Add labels (optional, can be customized)
                num_neurons, num_frames = df.shape
                df.columns = [f"Frame_{i+1}" for i in range(num_frames)]
                df.index = [f"Neuron_{i+1}" for i in range(num_neurons)]
                
                return df
            else:
                print("The dataset 'C_Raw_all' was not found in the file.")
                return None
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def print_shape(df):
    """
    Print the number of rows and columns of the DataFrame.
    """
    if df is not None:
        num_neurons, num_frames = df.shape
        print(f"Number of rows (neurons): {num_neurons}")
        print(f"Number of columns (frames/time): {num_frames}")
    else:
        print("DataFrame is empty or not loaded.")

def plot_neuron_signal(df, neuron_index):
    """
    Plot the signal of a given neuron over all frames.
    """
    if df is not None and 0 <= neuron_index < len(df):
        plt.figure(figsize=(10, 6))
        plt.plot(df.iloc[neuron_index], label=f"Neuron {neuron_index + 1}")
        plt.xlabel("Frame / Time")
        plt.ylabel("Signal Intensity")
        plt.title(f"Neuron {neuron_index + 1} Signal Over Time")
        plt.legend()
        plt.show()
    else:
        print("Invalid neuron index or DataFrame is not loaded.")

def plot_neuron_signal_with_range(df, neuron_index, start_frame, end_frame):
    """
    Plot the signal of a given neuron over a specified frame range.
    """
    if df is not None and 0 <= neuron_index < len(df):
        if 0 <= start_frame < end_frame <= df.shape[1]:
            plt.figure(figsize=(10, 6))
            plt.plot(df.iloc[neuron_index, start_frame:end_frame], label=f"Neuron {neuron_index + 1}")
            plt.xlabel("Frame / Time")
            plt.ylabel("Signal Intensity")
            plt.title(f"Neuron {neuron_index + 1} Signal from Frame {start_frame + 1} to {end_frame}")
            plt.legend()
            plt.show()
        else:
            print("Invalid frame range.")
    else:
        print("Invalid neuron index or DataFrame is not loaded.")

def plot_three_neurons(df, neuron_indices, start_frame, end_frame):
    """
    Plot signals of three neurons over a specified frame range.
    """
    if df is not None:
        if len(neuron_indices) == 3 and all(0 <= idx < len(df) for idx in neuron_indices):
            if 0 <= start_frame < end_frame <= df.shape[1]:
                plt.figure(figsize=(12, 8))
                for idx in neuron_indices:
                    plt.plot(df.iloc[idx, start_frame:end_frame], label=f"Neuron {idx + 1}")
                
                plt.xlabel("Frame / Time")
                plt.ylabel("Signal Intensity")
                plt.title(f"Signals of Neurons {neuron_indices[0] + 1}, {neuron_indices[1] + 1}, and {neuron_indices[2] + 1}")
                plt.legend()
                plt.show()
            else:
                print("Invalid frame range.")
        else:
            print("Invalid neuron indices or DataFrame is not loaded.")
    else:
        print("DataFrame is empty or not loaded.")

# 1. Load the data
df_c_raw_all = load_c_raw_all_to_dataframe(file_path)

# 2. Print number of rows and columns
print_shape(df_c_raw_all)

# 3. Plotting example: Plot a signal of a neuron (neuron index 0)
plot_neuron_signal(df_c_raw_all, neuron_index=0)

# 4. Plotting example with start and end frame: Plot signal of a neuron with frame range (neuron index 0, frames 100 to 200)
plot_neuron_signal_with_range(df_c_raw_all, neuron_index=0, start_frame=100, end_frame=200)

# 5. Plotting example for 3 neurons: Neuron indices [0, 1, 2], frame range 100 to 200
plot_three_neurons(df_c_raw_all, neuron_indices=[0, 1, 2], start_frame=100, end_frame=200)

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_action_potential_signal(length=500, peak_frame=200, peak_value=1.0, width=20):
    """
    Generate a synthetic action potential signal with a given peak position.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: The frame where the peak occurs.
    - peak_value: The value at the peak of the action potential.
    - width: The width of the action potential peak.
    
    Returns:
    - A NumPy array representing the action potential signal.
    """
    signal = np.zeros(length)
    start_frame = max(0, peak_frame - width // 2)
    end_frame = min(length, peak_frame + width // 2)
    
    # Create a simple Gaussian-like peak for the action potential
    signal[start_frame:end_frame] = np.exp(-((np.arange(start_frame, end_frame) - peak_frame) ** 2) / (2 * (width // 4) ** 2)) * peak_value
    return signal

def plot_similar_signals_with_shift(length=500, peak_frame=200, shift=20, width=20):
    """
    Plot two signals with similar action potentials, where the second signal is delayed.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: Frame at which the first signal peaks.
    - shift: The number of frames to shift the action potential in the second signal.
    - width: Width of the action potential.
    """
    # Generate two similar action potentials
    signal_1 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width)
    signal_2 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width)
    
    # Generate the third signal with the shifted peak
    signal_3 = generate_action_potential_signal(length=length, peak_frame=peak_frame + shift, peak_value=1.0, width=width)
    
    # Plot the signals
    plt.figure(figsize=(12, 8))

    # Plot signal 1
    plt.subplot(3, 1, 1)
    plt.plot(signal_1, label='Neuron 1 (Original Signal)', color='blue')
    plt.title('Neuron 1 Signal')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot signal 2 directly below
    plt.subplot(3, 1, 2)
    plt.plot(signal_2, label='Neuron 2 (Similar Action Potential)', color='green')
    plt.title('Neuron 2 Signal')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot the delayed signal 3
    plt.subplot(3, 1, 3)
    plt.plot(signal_3, label='Neuron 3 (Shifted Action Potential)', color='red')
    plt.title('Neuron 3 Signal with Delayed Action Potential')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the signals
plot_similar_signals_with_shift(length=500, peak_frame=200, shift=20, width=20)

# %%
import numpy as np
import matplotlib.pyplot as plt

def generate_action_potential_signal(length=500, peak_frame=200, peak_value=1.0, width=20, noise_level=0.1):
    """
    Generate a synthetic action potential signal with a given peak position and add noise.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: The frame where the peak occurs.
    - peak_value: The value at the peak of the action potential.
    - width: The width of the action potential peak.
    - noise_level: The standard deviation of the Gaussian noise added to the signal.
    
    Returns:
    - A NumPy array representing the action potential signal with added noise.
    """
    signal = np.zeros(length)
    start_frame = max(0, peak_frame - width // 2)
    end_frame = min(length, peak_frame + width // 2)
    
    # Create a Gaussian-like peak for the action potential
    signal[start_frame:end_frame] = np.exp(-((np.arange(start_frame, end_frame) - peak_frame) ** 2) / (2 * (width // 4) ** 2)) * peak_value
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, length)
    signal += noise
    
    return signal

def plot_similar_signals_with_shift(length=500, peak_frame=200, shift=40, width=20, noise_level=0.1, shorter_width=10):
    """
    Plot two signals with similar action potentials, where the second signal is delayed and shorter with added noise.
    
    Parameters:
    - length: Total number of frames (time points).
    - peak_frame: Frame at which the first signal peaks.
    - shift: The number of frames to shift the action potential in the second signal.
    - width: Width of the action potential for the first two signals.
    - noise_level: The noise level to add to the signals.
    - shorter_width: The width of the action potential for the third signal.
    """
    # Generate two similar action potentials with noise
    signal_1 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width, noise_level=noise_level)
    signal_2 = generate_action_potential_signal(length=length, peak_frame=peak_frame, peak_value=1.0, width=width, noise_level=noise_level)
    
    # Generate the third signal with the shifted and shorter action potential
    signal_3 = generate_action_potential_signal(length=length, peak_frame=peak_frame + shift, peak_value=1.0, width=shorter_width, noise_level=noise_level)
    
    # Plot the signals
    plt.figure(figsize=(12, 10))

    # Plot signal 1
    plt.subplot(3, 1, 1)
    plt.plot(signal_1, label='Neuron 1 (Original Signal)', color='blue')
    plt.title('Neuron 1 Signal with Noise')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot signal 2 directly below
    plt.subplot(3, 1, 2)
    plt.plot(signal_2, label='Neuron 2 (Similar Action Potential with Noise)', color='green')
    plt.title('Neuron 2 Signal with Noise')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    # Plot the delayed and shorter signal 3
    plt.subplot(3, 1, 3)
    plt.plot(signal_3, label='Neuron 3 (Delayed and Shorter Action Potential with Noise)', color='red')
    plt.title('Neuron 3 Signal with Delayed & Shorter Action Potential')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Frame / Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot the signals with noise and shifted action potential
plot_similar_signals_with_shift(length=500, peak_frame=200, shift=40, width=20, noise_level=0.1, shorter_width=10)

# %%

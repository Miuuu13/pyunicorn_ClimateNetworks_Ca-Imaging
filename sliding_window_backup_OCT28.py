import numpy as np
from scipy.stats import zscore

# Plan: smooth with sliding window,
# Trend anylysis - is there a trend? if yes, zscore not appropriate
# normalization correction - apply the normalization, if zscore appropriate use this
# 


          
        
#TODO chheck other older scripts


# def zscore_normalization(split_sessions_dict_no_nan: dict):
#     """Z-score normalize each neuron in each session after nans have been hadled."""
#     z_score_normalized_dict = {}
    
#     for session, session_data in split_sessions_dict_no_nan.items():
#         means = np.mean(session_data, axis=0)
#         stds = np.std(session_data, axis=0)
        
#         z_scored_data = (session_data - means) / stds
#         z_score_normalized_dict[session] = z_scored_data
    
#     return z_score_normalized_dict


#TODO smoothing using sliding window

#TODO in main: try 10,20,50 and plot them using alpha value

def smooth_neuron_traces_per_session(split_session_dct_nan_free: dict, sliding_window = 10) -> dict:
    """ Smooths the trace of a neuron per session (splitted dict) using a pre-defined sliding window """
    
    #TODO try 10,25,50 and 
    pass


# SMOOTHING


def handle_start(neuron_trace, window=10):
    """
    Smooth the start of the neuron trace by incrementally increasing the window size
    from 5 up to the full window size.
    """
    smoothed_start = []
    for i in range(5, window + 1):
        start_window = neuron_trace[:i]
        smoothed_start.append(np.mean(start_window))
    return smoothed_start

def handle_end(neuron_trace, window=10):
    """
    Smooth the end of the neuron trace by decrementally reducing the window size
    from window - 1 down to 5.
    """
    smoothed_end = []
    for i in range(window - 1, 4, -1):
        end_window = neuron_trace[-i:]
        smoothed_end.append(np.mean(end_window))
    return smoothed_end

def smooth_neuron_traces_per_session(split_sessions_dict_nan_free: dict, window: int = 10):
    """
    Apply a sliding window smoothing to neuron traces for each session in the dictionary.
    
    The start and end frames are handled separately:
    - Start: Uses a progressively increasing window from 5 to the full window size.
    - Middle: Applies the sliding window of the full window size.
    - End: Uses a progressively decreasing window from the full window size down to 5.
    
    @param split_sessions_dict_nan_free: Dictionary with cleaned neuron traces (split by session)
    @param window: The size of the sliding window. Default is 10 frames.
    @return: A new dictionary with smoothed neuron traces for each session.
    """
    # Initialize a new dictionary to store smoothed traces
    smoothed_sessions_dict = {}

    # Iterate through each session in the original dictionary
    for session_key, session_data in split_sessions_dict_nan_free.items():
        # Initialize a list to store smoothed traces for all neurons in the session
        smoothed_neurons = []

        # Apply smoothing to each neuron trace (each column in session_data matrix)
        for neuron_trace in session_data.T:  # Transpose to iterate over columns (neurons)
            # Smooth the start frames
            smoothed_neuron = handle_start(neuron_trace, window)
            
            # Perform sliding window averaging for the main part
            for i in range(len(neuron_trace) - window + 1):
                middle_window = neuron_trace[i:i + window]
                smoothed_neuron.append(np.mean(middle_window))
            
            # Smooth the end frames
            smoothed_neuron.extend(handle_end(neuron_trace, window))

            # Add the smoothed neuron data to the list
            smoothed_neurons.append(smoothed_neuron)
        
        # Convert the list of smoothed neurons back to a matrix format (time x neurons)
        smoothed_session_data = np.array(smoothed_neurons).T  # Transpose back to original format
        
        # Store the smoothed session data in the new dictionary
        smoothed_sessions_dict[session_key] = smoothed_session_data

    return smoothed_sessions_dict



# #RECURYSIVE



# def recursive_handle_start(neuron_trace, window=5, target_window=10):
#     """
#     Recursively smooth the start of the neuron trace by incrementally increasing the window size
#     from 5 up to the target window size (10 by default).
    
#     @param neuron_trace: The full neuron trace (array of fluorescence data).
#     @param window: The current window size (starts at 5).
#     @param target_window: The target window size to stop recursion (default is 10).
#     @return: A list of smoothed values for the start frames.
#     """
#     # Base case: if the window reaches the target window size, return an empty list
#     if window > target_window:
#         return []
    
#     # Calculate the mean of the current window at the start of the neuron trace
#     window_mean = np.mean(neuron_trace[:window])
    
#     # Recursive call with an increased window size
#     return [window_mean] + recursive_handle_start(neuron_trace, window + 1, target_window)


# def recursive_handle_end(neuron_trace, window=9, target_window=5):
#     """
#     Recursively smooth the end of the neuron trace by decrementally reducing the window size
#     from the current window down to the target window size (5 by default).
    
#     @param neuron_trace: The full neuron trace (array of fluorescence data).
#     @param window: The current window size (starts at 9).
#     @param target_window: The target window size to stop recursion (default is 5).
#     @return: A list of smoothed values for the end frames.
#     """
#     # Base case: if the window reaches the target window size, return an empty list
#     if window < target_window:
#         return []
    
#     # Calculate the mean of the current window at the end of the neuron trace
#     window_mean = np.mean(neuron_trace[-window:])
    
#     # Recursive call with a reduced window size
#     return [window_mean] + recursive_handle_end(neuron_trace, window - 1, target_window)


# def smooth_neuron_traces_per_session(split_sessions_dict_nan_free: dict, window: int = 10):
#     """
#     Apply a sliding window smoothing to neuron traces for each session in the dictionary.
    
#     Start and end frames are handled separately using recursive functions:
#     - Start: Uses a progressively increasing window from 5 up to the full window size.
#     - Middle: Applies a sliding window of the full window size.
#     - End: Uses a progressively decreasing window from the full window size down to 5.
    
#     @param split_sessions_dict_nan_free: Dictionary with cleaned neuron traces (split by session)
#     @param window: The size of the sliding window for the main part. Default is 10 frames.
#     @return: A new dictionary with smoothed neuron traces for each session.
#     """
#     # Initialize a new dictionary to store smoothed traces
#     smoothed_sessions_dict = {}

#     # Iterate through each session in the original dictionary
#     for session_key, session_data in split_sessions_dict_nan_free.items():
#         # Initialize a list to store smoothed traces for all neurons in the session
#         smoothed_neurons = []

#         # Apply smoothing to each neuron trace (each column in session_data matrix)
#         for neuron_trace in session_data.T:  # Transpose to iterate over columns (neurons)
#             # Smooth the start frames using recursive handling
#             smoothed_neuron = recursive_handle_start(neuron_trace, window=5, target_window=window)
            
#             # Perform sliding window averaging for the main part
#             for i in range(len(neuron_trace) - window + 1):
#                 middle_window = neuron_trace[i:i + window]
#                 smoothed_neuron.append(np.mean(middle_window))
            
#             # Smooth the end frames using recursive handling
#             smoothed_neuron.extend(recursive_handle_end(neuron_trace, window - 1, target_window=5))

#             # Add the smoothed neuron data to the list
#             smoothed_neurons.append(smoothed_neuron)
        
#         # Convert the list of smoothed neurons back to a matrix format (time x neurons)
#         smoothed_session_data = np.array(smoothed_neurons).T  # Transpose back to original format
        
#         # Store the smoothed session data in the new dictionary
#         smoothed_sessions_dict[session_key] = smoothed_session_data

#     return smoothed_sessions_dict



#--------------------------
# Normalization correction

#check diff max mean? see mattermost


#--------------------------
#TODO DECIDE for normalization after normalization correction!    
#TODO remove NaN beforehand

def z_score_normalize_neuron_traces_per_session(split_sessions_dict_nan_free: dict):
    """
    Z-score normalize the neuron traces for each session in split_sessions_dict_nan_free.
    Each neuron's trace will be normalized separately within each session.
    
    @param split_sessions_dict_nan_free: Dictionary containing cleaned neuron traces split by session.
    @return: A new dictionary with z-score normalized neuron traces for each session.
    """
    # Initialize a new dictionary to store normalized traces
    z_score_normalized_dict = {}

    # Iterate through each session in the original dictionary
    for session_key, session_data in split_sessions_dict_nan_free.items():
        # Initialize a list to store normalized traces for all neurons in the session
        z_score_neurons = []
        
        # Z-score normalize each neuron trace (each column in session_data)
        for neuron_trace in session_data.T:  # Transpose to iterate over neurons
            # Calculate mean and standard deviation for the neuron trace
            mean = np.mean(neuron_trace)
            std = np.std(neuron_trace)
            
            # Perform z-score normalization: (x - mean) / std
            if std != 0:  # Avoid division by zero
                z_score_neuron = (neuron_trace - mean) / std
            else:
                z_score_neuron = neuron_trace  # If std is 0, leave the trace as is
            
            # Append the normalized neuron data to the list
            z_score_neurons.append(z_score_neuron)
        
        # Convert the list of z-score normalized neurons back to a matrix format (time x neurons)
        z_score_session_data = np.array(z_score_neurons).T  # Transpose back to original format
        
        # Store the z-score normalized session data in the new dictionary
        z_score_normalized_dict[session_key] = z_score_session_data

    return z_score_normalized_dict













#########new:

import numpy as np


def calculate_initial_smoothed_points(neuron_trace, max_window):
    """Helper function to calculate initial smoothed points with increasing window sizes."""
    initial_points = []
    for i in range(5, max_window + 1):  # Start from a window of 5 up to max_window
        initial_points.append(np.mean(neuron_trace[:i]))
    return initial_points

def apply_sliding_window_smoothing(neuron_trace, window_size=10):
    """Helper function to apply sliding window smoothing with specified window size."""
    smoothed_trace = calculate_initial_smoothed_points(neuron_trace, window_size)
    
    # Main sliding window smoothing
    for i in range(window_size, len(neuron_trace) - window_size + 1):
        smoothed_trace.append(np.mean(neuron_trace[i - window_size + 1:i + 1]))
    
    # Final smoothing points for last frames
    remaining_frames = len(neuron_trace) - len(smoothed_trace)
    for i in range(remaining_frames):
        smoothed_trace.append(np.mean(neuron_trace[-(5 + i):]))
        
    return smoothed_trace

# Option 1: Non-Recursive Smoothing Function
def smooth_sessions_non_recursive(split_sessions_dict_nan_free, window_size=10):
    """Smooth each neuron trace in each session without recursion."""
    split_sessions_dict_nan_free_smoothed = {}
    
    for session_key, session_data in split_sessions_dict_nan_free.items():
        smoothed_data = []
        
        for neuron_trace in session_data.T:  # Transpose for easier access per neuron
            smoothed_trace = apply_sliding_window_smoothing(neuron_trace, window_size)
            smoothed_data.append(smoothed_trace)
        
        split_sessions_dict_nan_free_smoothed[session_key] = np.column_stack(smoothed_data)
    
    return split_sessions_dict_nan_free_smoothed

# Option 2: Recursive Smoothing Function
def smooth_sessions_recursive(split_sessions_dict_nan_free, window_size=10):
    """Smooth each neuron trace in each session with recursion."""
    
    def recursive_smoothing(trace, start_index=0, smoothed_trace=[]):
        if start_index < 5:
            smoothed_trace.append(np.mean(trace[:start_index + 5]))
        elif start_index >= len(trace) - 5:
            remaining_frames = len(trace) - start_index
            smoothed_trace.append(np.mean(trace[-(5 + remaining_frames - 1):]))
            if start_index == len(trace) - 1:
                return smoothed_trace
        else:
            smoothed_trace.append(np.mean(trace[start_index - 4:start_index + 6]))
        
        return recursive_smoothing(trace, start_index + 1, smoothed_trace)
    
    split_sessions_dict_nan_free_smoothed = {}
    for session_key, session_data in split_sessions_dict_nan_free.items():
        smoothed_data = []
        
        for neuron_trace in session_data.T:
            smoothed_trace = recursive_smoothing(neuron_trace)
            smoothed_data.append(smoothed_trace)
        
        split_sessions_dict_nan_free_smoothed[session_key] = np.column_stack(smoothed_data)
    
    return split_sessions_dict_nan_free_smoothed

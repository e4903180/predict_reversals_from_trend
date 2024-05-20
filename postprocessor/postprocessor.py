import numpy as np
import pandas as pd
import torch


class Postprocesser:
    def __init__(self):
        pass
    
    def modify_rows(self, arr):
        for row in arr:
            # Find the first index in the row where the change occurs
            change_index = np.where(np.diff(row) != 0)[0]
            if change_index.size > 0:
                first_change_index = change_index[0] + 1
                # Set all values after the first change to the value at the change index
                row[first_change_index:] = row[first_change_index]
        return arr

    def remove_short_sequences(self, arr, x):
        """
        Remove sequences in the array that are shorter than x, considering both 0 to 1 and 1 to 0 changes.

        :param arr: The input array
        :param x: The minimum sequence length to keep
        :return: The modified array
        """
        # Identify the changes in the array
        change_indices = np.where(np.diff(arr) != 0)[0] + 1
        # Include the start and end of the array
        change_indices = np.insert(change_indices, 0, 0)
        change_indices = np.append(change_indices, len(arr))
        
        for i in range(len(change_indices) - 1):
            # Calculate the length of the sequence
            seq_length = change_indices[i+1] - change_indices[i]
            if seq_length < x:
                # Set the values of short sequences to the value preceding the sequence
                arr[change_indices[i]:change_indices[i+1]] = arr[change_indices[i] - 1]
        return arr

    # def process_signals(self, y_data, dates, filter):
    #     binary_data = (y_data > 0.5).to(torch.int32)
    #     flatten_binary_data = binary_data[:, :, 1].flatten() # 數據經過one-hot encoding，只取第二個值才能還原至原始Trend
        
    #     if filter != 'False':
    #         flatten_binary_data = self.remove_short_sequences(flatten_binary_data, filter)
        
    #     signals = np.full(flatten_binary_data.shape, '', dtype=object)

    #     for i in range(1, len(flatten_binary_data)):
    #         # downward to upward
    #         if flatten_binary_data[i-1] == 1 and flatten_binary_data[i] == 0:
    #             signals[i] = 'Buy'
    #         # upward to downward
    #         elif flatten_binary_data[i-1] == 0 and flatten_binary_data[i] == 1:
    #             signals[i] = 'Sell'

    #     non_empty_signals = np.where(signals != '')[0]
    #     # if non_empty_signals.size > 0:
    #     #     first_signal_index = non_empty_signals[0]
    #     #     last_signal_index = non_empty_signals[-1]
    #     #     signals[first_signal_index] += ' (first)'
    #     #     signals[last_signal_index] += ' (last)'

    #     flat_dates = dates.flatten()
    #     return pd.DataFrame({'Date': flat_dates, 'Signal': signals})
    
    def process_signals(self, max_indices, dates, filter):
        # max_indices = self.modify_rows(max_indices)
        flatten_max_indices = max_indices.flatten()
        if filter != 'False':
            flatten_max_indices = self.remove_short_sequences(flatten_max_indices, filter)
        # signals = np.full(flatten_max_indices.shape, '', dtype=object)
        signals = np.zeros(flatten_max_indices.shape)
        for i in range(1, len(flatten_max_indices)):
            # downward to upward
            if flatten_max_indices[i-1] == 1 and flatten_max_indices[i] == 0:
                # signals[i] = 'Buy'
                signals[i] = 1
            # upward to downward
            elif flatten_max_indices[i-1] == 0 and flatten_max_indices[i] == 1:
                # signals[i] = 'Sell'
                signals[i] = -1

        # non_empty_signals = np.where(signals != '')[0]
        # if non_empty_signals.size > 0:
        #     first_signal_index = non_empty_signals[0]
        #     last_signal_index = non_empty_signals[-1]
        #     signals[first_signal_index] += ' (first)'
        #     signals[last_signal_index] += ' (last)'

        flat_dates = dates.flatten()
        return pd.DataFrame({'Date': flat_dates, 'Signal': signals})
    
    def change_values_after_first_reverse_point(self, max_indices:torch.Tensor):
        for idx, sub_y in enumerate(max_indices):
            array = sub_y.numpy()
            transition_found = False
            for i in range(1, len(array)):
                if not (array[i] == array[i-1]).all():
                    array[i:] = array[i]
                    transition_found = True
                    break
            if not transition_found:
                array = sub_y.numpy()
            
            max_indices[idx] = torch.tensor(array)
        return max_indices

    def get_first_trend_reversal_signals(self, max_indices):
        """
        This function calculates the first trend reversal signal for each row of an array.
        The signal indicates the first change from upward to downward (0 to 1) or
        downward to upward (1 to 0) within each row.

        Parameters:
        - max_indices (ndarray): A 2D numpy array with trend indices (1 for upward, 0 for downward).

        Returns:
        - signals (ndarray): A 1D numpy array containing the first trend reversal signals
                            for each row: 1 for downward to upward, -1 for upward to downward, 
                            and 0 if no reversal is found.
        """
        # Initialize an array to store the signals
        signals = np.zeros(max_indices.shape[0])

        # Iterate over each row to determine the first trend reversal
        for idx in range(max_indices.shape[0]):
            for i in range(1, max_indices.shape[1]):
                # downward to upward
                if max_indices[idx][i - 1] == 1 and max_indices[idx][i] == 0:
                    signals[idx] = 1
                    break
                # upward to downward
                elif max_indices[idx][i - 1] == 0 and max_indices[idx][i] == 1:
                    signals[idx] = -1
                    break

        return signals
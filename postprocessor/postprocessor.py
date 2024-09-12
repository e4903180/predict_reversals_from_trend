import numpy as np
import pandas as pd
import torch

class Postprocessor:
    """
    A class used to handle the postprocessing of model predictions for trend reversals.

    Attributes:
        params (dict): Parameters for controlling the postprocessing behavior.
    """
    
    def __init__(self, params):
        """
        Initializes the Postprocessor with given parameters.

        Args:
            params (dict): A dictionary of parameters for postprocessing.
        """
        self.params = params

    def change_values_after_first_reverse_point(self, trend_indices: torch.Tensor):
        """
        Modify the values in the trend_indices tensor after the first trend reversal.

        Args:
            trend_indices (torch.Tensor): Tensor containing trend indices.

        Returns:
            torch.Tensor: Tensor with values changed after the first reversal point.
        """
        for idx, sub_y in enumerate(trend_indices):
            array = sub_y.detach().numpy()
            transition_found = False
            for i in range(1, len(array)):
                if not (array[i] == array[i-1]).all():
                    array[i:] = array[i]
                    transition_found = True
                    break
            if not transition_found:
                array = sub_y.detach().numpy()
            
            trend_indices[idx] = torch.tensor(array)
        return trend_indices
    
    def get_first_trend_reversal_and_idx_signals(self, trend_indices):
        """
        Calculates the first trend reversal signal for each row of an array.

        Args:
            trend_indices (ndarray): A 2D numpy array with trend indices (1 for downward, 0 for upward).

        Returns:
            tuple: A tuple containing:
                - reverse_signals (ndarray): Array with the first trend reversal signals for each row.
                - reverse_idx (ndarray): Array with the indices of the first trend reversal for each row.
        """
        reverse_signals = np.zeros(trend_indices.shape[0], dtype=int)
        reverse_idx = np.zeros(trend_indices.shape[0], dtype=int)

        for idx in range(trend_indices.shape[0]):
            for i in range(1, trend_indices.shape[1]):
                if trend_indices[idx][i - 1] == 1 and trend_indices[idx][i] == 0:  # Downward to upward
                    reverse_signals[idx] = -1  # Valley
                    reverse_idx[idx] = i
                    break
                elif trend_indices[idx][i - 1] == 0 and trend_indices[idx][i] == 1:  # Upward to downward
                    reverse_signals[idx] = 1  # Peak
                    reverse_idx[idx] = i
                    break
        return reverse_signals, reverse_idx
    
    def get_trade_signals(self, reverse_signals, reverse_idx, test_dates, target_dataset):
        """
        Generates trade signals based on the reversal signals.

        Args:
            reverse_signals (ndarray): Array of reversal signals.
            reverse_idx (ndarray): Array of indices where reversals occur.
            test_dates (ndarray): Array of test dates.
            target_dataset (pd.DataFrame): The target dataset.

        Returns:
            pd.DataFrame: DataFrame with generated trade signals.
        """
        trade_signals = pd.DataFrame(index=target_dataset.loc[test_dates[0][0]:test_dates[-1][-1]].index, columns=['Order'])
        for idx in range(0, reverse_idx.shape[0]):
            if reverse_signals[idx] == 1:  # Peak
                trade_signals.loc[test_dates[idx][reverse_idx[idx]], 'Order'] = 'Sell'
            elif reverse_signals[idx] == -1:  # Valley
                trade_signals.loc[test_dates[idx][reverse_idx[idx]], 'Order'] = 'Buy'
        return trade_signals
    
    def compare_reverse_predictions(self, y_preds_reverse_idx, y_preds_reverse_signals, y_test_reverse_idx, y_test_reverse_signals):
        """
        Compares predicted reversal indices and signals with actual values.

        Args:
            y_preds_reverse_idx (ndarray): Predicted reversal indices.
            y_preds_reverse_signals (ndarray): Predicted reversal signals.
            y_test_reverse_idx (ndarray): Actual reversal indices.
            y_test_reverse_signals (ndarray): Actual reversal signals.

        Returns:
            pd.DataFrame: DataFrame containing the comparison results.
        """
        reverse_info = {
            'predicted_reverse_idx': y_preds_reverse_idx,
            'predicted_reverse_signals': y_preds_reverse_signals,
            'predicted_reverse_label': None,
            'actual_reverse_idx': y_test_reverse_idx,
            'actual_reverse_signals': y_test_reverse_signals,
            'actual_reverse_label': None
        }
        
        reverse_difference = pd.DataFrame(reverse_info)
        
        reverse_difference['reverse_signal_correct'] = \
            reverse_difference['predicted_reverse_signals'] == reverse_difference['actual_reverse_signals']
        reverse_difference['reverse_idx_difference'] = reverse_difference.apply(
            lambda row: (row['predicted_reverse_idx'] - row['actual_reverse_idx']) if row['reverse_signal_correct'] else None, 
            axis=1
        )
        
        reverse_idx_difference_max = self.params.get('reverse_idx_difference_max', 5)
        reverse_idx_difference_min = self.params.get('reverse_idx_difference_min', -2)
        reverse_difference['predict_in_range'] = \
            (reverse_difference['reverse_idx_difference'] <= reverse_idx_difference_max) & \
                (reverse_difference['reverse_idx_difference'] >= reverse_idx_difference_min)
        
        label_map = {-1: 'Valley', 0: 'No reversal', 1: 'Peak'}
        reverse_difference['actual_reverse_label'] = reverse_difference['actual_reverse_signals'].map(label_map)
        reverse_difference['predicted_reverse_label'] = reverse_difference['predicted_reverse_signals'].map(label_map)
        
        return reverse_difference

    def calculate_reversal_dates(self, reverse_signals, reverse_idx, test_dates, target_dataset):
        """
        Determines where reversals occur within the given data.

        Args:
            reverse_signals (ndarray): Array of reversal signals.
            reverse_idx (ndarray): Array of indices where reversals occur.
            test_dates (ndarray): Array of test dates.
            target_dataset (pd.DataFrame): The target dataset.

        Returns:
            ndarray: Array indicating the dates where reversals occur.
        """
        reversal_dates = pd.DataFrame(index=target_dataset.loc[test_dates[0][0]:test_dates[-1][-1]].index, columns=['Reversals'])
        for idx in range(0, reverse_idx.shape[0]):
            if reverse_signals[idx] == 1:  # Peak
                reversal_dates.loc[test_dates[idx][reverse_idx[idx]], 'Reversals'] = 1
            elif reverse_signals[idx] == -1:  # Valley
                reversal_dates.loc[test_dates[idx][reverse_idx[idx]], 'Reversals'] = -1
        reversal_dates.replace(np.nan, 0, inplace=True)
        return reversal_dates['Reversals'].values
    
    def calculate_reversal_dates_with_signals(self, reverse_signals, reverse_idx, test_dates, target_dataset):
        """
        Determines where reversals occur, skipping non-reversal dates.

        Args:
            reverse_signals (ndarray): Array of reversal signals.
            reverse_idx (ndarray): Array of indices where reversals occur.
            test_dates (ndarray): Array of test dates.
            target_dataset (pd.DataFrame): The target dataset.

        Returns:
            tuple: A tuple containing:
                - ndarray: Array indicating the overlap of reversals with passing signals.
                - list: List of reversal signal indices.
                - list: List of reversal indices.
        """
        reversal_dates = pd.DataFrame(index=target_dataset.loc[test_dates[0][0]:test_dates[-1][-1]].index, columns=['Reversals'])
        counter = 0
        valid_reverse_signals = []
        valid_reverse_indices = []

        while counter < reverse_idx.shape[0]:
            if reverse_signals[counter] == 1:  # Peak
                reversal_dates.loc[test_dates[counter][reverse_idx[counter]], 'Reversals'] = 1
                valid_reverse_signals.append(counter)
                valid_reverse_indices.append(reverse_idx[counter])
                counter += reverse_idx[counter]
            elif reverse_signals[counter] == -1:  # Valley
                reversal_dates.loc[test_dates[counter][reverse_idx[counter]], 'Reversals'] = -1
                valid_reverse_signals.append(counter)
                valid_reverse_indices.append(reverse_idx[counter])
                counter += reverse_idx[counter]
            else:
                counter += 1
        reversal_dates.replace(np.nan, 0, inplace=True)
        return reversal_dates['Reversals'].values, valid_reverse_signals, valid_reverse_indices
    
    def postprocess_predictions(self, y_preds, y_test, test_dates, target_dataset):
        """
        Post-processes the model predictions to generate trade signals and other metrics.

        Args:
            y_preds (torch.Tensor): Predicted values from the model.
            y_test (torch.Tensor): Ground truth values.
            test_dates (ndarray): Array of test dates.
            target_dataset (pd.DataFrame): The target dataset.

        Returns:
            dict: Dictionary with postprocessed results including:
                - 'y_preds_indices': Processed predicted indices.
                - 'y_test_indices': Processed ground truth indices.
                - 'test_trade_signals': DataFrame with test trade signals.
                - 'predicted_trade_signals': DataFrame with predicted trade signals.
                - 'passing_trade_signals': DataFrame with passing trade signals.
                - 'comparison_summary': DataFrame with comparison results.
                - 'filtered_reversal_dates': Array with filtered reversal dates.
                - 'reversal_dates_test': Array with reversal dates for the test set.
                - 'valid_signals': List of valid signals.
                - 'valid_indices': List of valid indices.
        """
        y_preds_indices = self.change_values_after_first_reverse_point(y_preds)
        y_test_indices = self.change_values_after_first_reverse_point(y_test)
        
        y_preds_reverse_signals, y_preds_reverse_idx = \
            self.get_first_trend_reversal_and_idx_signals(y_preds_indices)
        y_test_reverse_signals, y_test_reverse_idx = \
            self.get_first_trend_reversal_and_idx_signals(y_test_indices)
        
        test_trade_signals = \
            self.get_trade_signals(y_test_reverse_signals, y_test_reverse_idx, test_dates, target_dataset)
        predicted_trade_signals = \
            self.get_trade_signals(y_preds_reverse_signals, y_preds_reverse_idx, test_dates, target_dataset)
        
        passing_trade_signals = \
            self.get_trade_signals(y_test_reverse_signals, y_test_reverse_idx, test_dates, target_dataset)
        
        comparison_summary = \
            self.compare_reverse_predictions(y_preds_reverse_idx, y_preds_reverse_signals, y_test_reverse_idx, y_test_reverse_signals)
        
        filtered_reversal_dates, valid_signals, valid_indices = \
            self.calculate_reversal_dates_with_signals(y_test_reverse_signals, y_test_reverse_idx, test_dates, target_dataset)
        reversal_dates_test = self.calculate_reversal_dates(y_test_reverse_signals, y_test_reverse_idx, test_dates, target_dataset)
        
        return {
            'y_preds_indices': y_preds_indices,
            'y_test_indices': y_test_indices,
            'test_trade_signals': test_trade_signals,
            'predicted_trade_signals': predicted_trade_signals,
            'passing_trade_signals': passing_trade_signals,
            'comparison_summary': comparison_summary,
            'filtered_reversal_dates': filtered_reversal_dates,
            'reversal_dates_test': reversal_dates_test,
            'valid_signals': valid_signals,
            'valid_indices': valid_indices
        }

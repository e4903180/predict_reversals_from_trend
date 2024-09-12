import json
from preprocessor.preprocessor import Preprocessor
from model.model import Model
from postprocessor.postprocessor import Postprocessor
from evaluator.evaluator import Evaluator
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import time
import pickle
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class ReversePrediction():
    """A class for performing trend reversal prediction using deep learning models."""

    def set_seed(self, seed_value):
        """Sets the random seed for reproducibility across different libraries.

        Args:
            seed_value (int): The seed value to be used for random number generation.
        """
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    def run(self, params):
        """Executes the trend reversal prediction workflow, including preprocessing, model training, and evaluation.

        Args:
            params (dict): Dictionary containing various parameters for the model, training, and evaluation.

        Returns:
            tuple: A tuple containing validation results and test results.
        """
        self.set_seed(42)
        
        # Preprocess data
        preprocessor = Preprocessor(params)
        X_train, y_train, X_val, y_val, X_test, y_test, train_dates, test_dates, val_dates, target_dataset = \
            preprocessor.get_datasets()
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Train the model
        start_time = time.time()
        model_wrapper = Model(params)
        model, history = model_wrapper.train_model(train_loader, val_loader, \
            num_epochs=params['training_epoch_num'], patience=params['patience'])
        end_time = time.time()
        execution_time = end_time - start_time

        # Save the trained model
        torch.save(model, params['save_path']['trained_model_path'])
        
        # Post-process and evaluate on validation data
        y_preds_val = model(X_val)
        postprocessor = Postprocessor(params)

        # Call the postprocess_signals method
        postprocess_results = postprocessor.postprocess_predictions(y_preds_val, y_val, val_dates, target_dataset)

        # Extract values from the results dictionary
        y_preds_val_reverse_idx = postprocess_results['y_preds_indices']
        y_val_reverse_idx = postprocess_results['y_test_indices']
        val_trade_signals = postprocess_results['test_trade_signals']
        pred_val_trade_signals = postprocess_results['predicted_trade_signals']
        pass_pred_trade_signals_val = postprocess_results['passing_trade_signals']
        val_reverse_difference = postprocess_results['comparison_summary']
        reversals_pred_val_pass = postprocess_results['filtered_reversal_dates']
        reversals_val = postprocess_results['reversal_dates_test']
        y_reverse_signals_val = postprocess_results['valid_signals']
        y_reverse_idx_val = postprocess_results['valid_indices']

        # Initialize online training history
        online_training_history = {
            'loss': [],
            'binary_accuracy': []
        }
        
        evaluator = Evaluator(params)
        val_result = evaluator.evaluate_and_generate_results(model, y_val, y_val_reverse_idx, y_preds_val, \
            y_preds_val_reverse_idx, val_reverse_difference.iloc[y_reverse_signals_val], history, online_training_history, \
            reversals_val, reversals_pred_val_pass, pred_val_trade_signals, pass_pred_trade_signals_val, \
            target_dataset, val_dates, execution_time, show=False)
        
        # Save validation results
        with open(params['save_path']['val_summary_save_path'], 'w') as f:
            json.dump(val_result, f)

        # Post-process and evaluate on test data
        y_preds = model(X_test)
        postprocessor = Postprocessor(params)

        # Call the postprocess_signals method
        postprocess_results = postprocessor.postprocess_predictions(y_preds, y_test, test_dates, target_dataset)

        # Extract values from the results dictionary
        y_preds_reverse_idx = postprocess_results['y_preds_indices']
        y_test_reverse_idx = postprocess_results['y_test_indices']
        test_trade_signals = postprocess_results['test_trade_signals']
        pred_trade_signals = postprocess_results['predicted_trade_signals']
        pass_pred_trade_signals = postprocess_results['passing_trade_signals']
        reverse_difference = postprocess_results['comparison_summary']
        reversals_pred_pass = postprocess_results['filtered_reversal_dates']
        reversals_test = postprocess_results['reversal_dates_test']
        y_reverse_signals = postprocess_results['valid_signals']
        y_reverse_idx = postprocess_results['valid_indices']

        evaluator = Evaluator(params)
        result = evaluator.evaluate_and_generate_results(model, y_test, y_test_reverse_idx, y_preds, \
            y_preds_reverse_idx, reverse_difference.iloc[y_reverse_signals], history, online_training_history, \
            reversals_test, reversals_pred_pass, pred_trade_signals, pass_pred_trade_signals, \
                target_dataset, test_dates, execution_time, show=False)

        # Save test results
        with open(params['save_path']['summary_save_path'], 'w') as f:
            json.dump(result, f)
            
        return val_result, result

if __name__ == '__main__':
    with open('parameters.json') as f:
        params = json.load(f)

    reverse_prediction = ReversePrediction()
    val_result, result = reverse_prediction.run(params)
    print('Validation results:', val_result)
    print('Test results:', result)
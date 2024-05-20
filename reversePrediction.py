import json
from preprocessor.preprocessor_pytorch import Preprocessor
from model.model_pytorch import Model
from postprocessor.postprocessor import Postprocesser
from evaluator.evaluator_pytorch import Evaluator
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

class ReversePrediction():
    def set_seed(self, seed_value):
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)


    def run(self, params):
        self.set_seed(42)
        preprocessor = Preprocessor(params)
        X_train, y_train, X_val, y_val, X_test, y_test, test_dates, X_newest, x_newest_date, y_newest_date, target_symbol_data, y_train_transition_log, y_val_transition_log, y_test_transition_log = preprocessor.get_multiple_data()
        start_time = time.time()
        model_wrapper = Model(params=params)
        model, history, y_preds, online_training_losses, online_training_acc = \
            model_wrapper.run(X_train, y_train, X_test, y_test, X_val, y_val)
        end_time = time.time()
        execution_time = end_time - start_time
        y_preds = torch.tensor(y_preds, dtype=torch.float32)
        y_preds_original = y_preds.clone()
        y_pred_newest = model.forward(X_newest)
        y_pred_newest = torch.tensor(y_pred_newest, dtype=torch.float32)
        postprocessor = Postprocesser()

        y_test_max_indices = np.argmax(y_test, axis=-1)
        y_preds_max_indices = np.argmax(y_preds, axis=-1)
        y_pred_newest_max_indices = np.argmax(y_pred_newest, axis=-1)
            
        if params["filter_reverse_trend_preds"] == "True":
            y_preds_max_indices = postprocessor.change_values_after_first_reverse_point(y_preds_max_indices)
            
        if params["filter_reverse_trend_preds"] == "True":
            y_pred_newest_max_indices = postprocessor.change_values_after_first_reverse_point(y_pred_newest_max_indices)
            
            test_trade_signals = postprocessor.process_signals(y_test_max_indices, test_dates, False)
        pred_trade_signals = postprocessor.process_signals(y_preds_max_indices, test_dates, params['filter'])
        newest_trade_signals = postprocessor.process_signals(y_pred_newest_max_indices, y_newest_date, False)

        # Get first trend reversal signals
        test_signal = postprocessor.get_first_trend_reversal_signals(y_test_max_indices)
        pred_signal = postprocessor.get_first_trend_reversal_signals(y_preds_max_indices)
        evaluator = Evaluator(params=params)
        model_summary, trend_confusion_matrix_info, reversed_trend_confusion_matrix_info, signal_confusion_matrix_info, \
            roc_auc, pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance, backtesting_report, trade_summary, execution_time = \
            evaluator.generate_numericale_data(model, y_test, y_preds, test_signal, pred_signal, test_trade_signals, pred_trade_signals, target_symbol_data, execution_time)

        evaluator.get_plots(y_test, y_preds, y_preds_original, test_trade_signals, pred_trade_signals, target_symbol_data, history, online_training_acc, online_training_losses, pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance, y_test_max_indices, y_preds_max_indices, pred_signal, test_signal, show='False')

        return model_summary, trend_confusion_matrix_info, reversed_trend_confusion_matrix_info, signal_confusion_matrix_info, \
            roc_auc, pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance, backtesting_report, trade_summary, execution_time


if __name__ == '__main__':            
    open('progress.txt', 'w').close()
    open('log.txt', 'w').close()
    root_path = 'DNN_Projects_weights_learningRate'
    for floder in tqdm.tqdm(os.listdir(root_path), file=open('progress.txt', 'a')):
        first_path = os.path.join(root_path, floder)
        for subfloder in tqdm.tqdm(os.listdir(first_path), file=open('progress.txt', 'a')):
            second_path = os.path.join(first_path, subfloder)
            print(second_path, file=open('progress.txt', 'a'))
            params = json.load(open(os.path.join(second_path, 'parameters.json'), 'r'))
            reversePrediction = ReversePrediction()
            reversePrediction.set_seed(42)
            try: 
                model_summary, trend_confusion_matrix_info, reversed_trend_confusion_matrix_info, signal_confusion_matrix_info, \
                roc_auc, pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance, backtesting_report, trade_summary, execution_time = reversePrediction.run(params)
                trend_confusion_matrix_info = trend_confusion_matrix_info.to_json()
                reversed_trend_confusion_matrix_info = reversed_trend_confusion_matrix_info.to_json()
                signal_confusion_matrix_info = signal_confusion_matrix_info.to_json()
                roc_auc_json = json.dumps(roc_auc)
                model_summary_json = json.dumps(model_summary)
                backtesting_report_json = json.dumps(backtesting_report)
                pred_days_difference_results_json = pred_days_difference_results.to_json()
                pred_days_difference_abs_mean_json = json.dumps(pred_days_difference_abs_mean)
                pred_in_advance_json = json.dumps(pred_in_advance)
                trade_summary_json = json.dumps(trade_summary)
                execution_time_json = json.dumps(execution_time)

                response = {
                    'msg': 'Received!',
                    'usingData': params,
                    'model_summary': model_summary_json,
                    'trend_confusion_matrix_info': trend_confusion_matrix_info,
                    'reversed_trend_confusion_matrix_info': reversed_trend_confusion_matrix_info,
                    'signal_confusion_matrix_info': signal_confusion_matrix_info,
                    'roc_auc': roc_auc_json,
                    'pred_days_difference_results': pred_days_difference_results_json,
                    'pred_days_difference_abs_mean': pred_days_difference_abs_mean_json,
                    'pred_in_advance': pred_in_advance_json,
                    'backtesting_report': backtesting_report_json,
                    'trade_summary': trade_summary_json,
                    'execution_time': execution_time_json,
                }
                
                with open(params['save_path']['summary_save_path'], 'w') as f:
                    json.dump(response, f)
                    
                # response
                print('done', file=open('progress.txt', 'a'))
            except Exception as e:
                print(e, file=open('progress.txt', 'a'))
                continue


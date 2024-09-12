import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import backtrader as bt
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from io import StringIO
import sys
import os
import json
from sklearn.metrics import roc_curve, precision_recall_curve, auc


class Evaluator:
    """A class for evaluating predictions related to market reversals and trends."""

    def __init__(self, params):
        """Initializes the Evaluator with the given parameters.

        Args:
            params (dict): Configuration or parameters needed for evaluation.
        """
        self.params = params

    def generate_model_summary(self, model):
        total = sum([param.nelement() for param in model.parameters()])
        model_summary = f'{model}, \nNumber of parameter: {total}'
        return model_summary
    
    def evaluate_trend_predictions(self, y_test_max_indices, y_preds_max_indices, show=False, save_path=None):
        """Evaluates binary classification of trend predictions.

        Args:
            y_test_max_indices (array-like): True indices for trends.
            y_preds_max_indices (array-like): Predicted indices for trends.
            show (bool, optional): Whether to display the confusion matrix. Defaults to False.
            save_path (str, optional): Path to save the confusion matrix plot. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing accuracy, precision, recall, and F1 score.
        """
        # Convert to class labels if necessary
        y_test_labels = np.where(
            y_test_max_indices.flatten() == 0, 'uptrend', 'downtrend')
        y_preds_labels = np.where(
            y_preds_max_indices.flatten() == 0, 'uptrend', 'downtrend')

        # Compute confusion matrix
        label_names = ['uptrend', 'downtrend']
        cm = confusion_matrix(
            y_test_labels, y_preds_labels, labels=label_names)

        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Trend Confusion Matrix')

        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_preds_labels)
        precision = precision_score(
            y_test_labels, y_preds_labels, pos_label='uptrend')
        recall = recall_score(
            y_test_labels, y_preds_labels, pos_label='uptrend')
        f1 = f1_score(y_test_labels, y_preds_labels, pos_label='uptrend')

        confusion_matrix_info = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [
                                             precision], 'Recall': [recall], 'F1 Score': [f1]})

        # Annotate metrics on the plot
        plt.xlabel(
            f"Predicted Label\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        plt.ylabel(f'Actual Label\n')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return confusion_matrix_info
    
    def evaluate_reversals_predictions_two_type(self, y_test_max_indices, y_preds_max_indices, show=False, save_path=None):
        """Evaluates binary classification of reversal predictions.

        Args:
            y_test_max_indices (array-like): True indices for reversals.
            y_preds_max_indices (array-like): Predicted indices for reversals.
            show (bool, optional): Whether to display the confusion matrix. Defaults to False.
            save_path (str, optional): Path to save the confusion matrix plot. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing accuracy, precision, recall, and F1 score.
        """
        # Convert to class labels if necessary
        y_test_labels = np.where(
            y_test_max_indices.flatten() == 0, 'No Reversal', 'Reversal')
        y_preds_labels = np.where(
            y_preds_max_indices.flatten() == 0, 'No Reversal', 'Reversal')

        # Compute confusion matrix
        label_names = ['Reversal', 'No Reversal']
        cm = confusion_matrix(
            y_test_labels, y_preds_labels, labels=label_names)

        sns.heatmap(cm, annot=True, cmap="Blues", fmt='d',
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Reversal Confusion Matrix')

        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, y_preds_labels)
        precision = precision_score(
            y_test_labels, y_preds_labels, pos_label='Reversal')
        recall = recall_score(
            y_test_labels, y_preds_labels, pos_label='Reversal')
        f1 = f1_score(y_test_labels, y_preds_labels, pos_label='Reversal')

        confusion_matrix_info = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [
                                             precision], 'Recall': [recall], 'F1 Score': [f1]})

        # Annotate metrics on the plot
        plt.xlabel(f"Predicted Label\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, \
            Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        plt.ylabel(f'Actual Label\n')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return confusion_matrix_info

    def evaluate_reversals_predictions_three_type(self, reversals_test, reversals_pred_pass, show=False, save_path=None):
        """Evaluates ternary classification of reversal predictions.

        Args:
            reversals_test (array-like): True labels for reversals.
            reversals_pred_pass (array-like): Predicted labels for reversals.
            show (bool, optional): Whether to display the confusion matrix. Defaults to False.
            save_path (str, optional): Path to save the confusion matrix plot. Defaults to None.

        Returns:
            tuple: A tuple containing two DataFrames - one for class-wise metrics and one for overall metrics.
        """
        label_names = ['Peak', 'No Reversal', 'Valley']
        reversals_mapping = {0: 'No Reversal', 1: 'Peak', -1: 'Valley'}
        true_labels = pd.Series(reversals_test).map(reversals_mapping).values
        pred_labels = pd.Series(reversals_pred_pass).map(
            reversals_mapping).values

        # Compute overall metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision_macro = precision_score(
            true_labels, pred_labels, average='macro', labels=label_names)
        recall_macro = recall_score(
            true_labels, pred_labels, average='macro', labels=label_names)
        f1_macro = f1_score(true_labels, pred_labels,
                            average='macro', labels=label_names)
        precision_micro = precision_score(
            true_labels, pred_labels, average='micro', labels=label_names)
        recall_micro = recall_score(
            true_labels, pred_labels, average='micro', labels=label_names)
        f1_micro = f1_score(true_labels, pred_labels,
                            average='micro', labels=label_names)

        # Compute metrics for each class
        precision_per_class = precision_score(
            true_labels, pred_labels, average=None, labels=label_names)
        recall_per_class = recall_score(
            true_labels, pred_labels, average=None, labels=label_names)
        f1_per_class = f1_score(true_labels, pred_labels,
                                average=None, labels=label_names)

        # Create a DataFrame for class-wise metrics
        class_confusion_matrix_info = pd.DataFrame({
            'Accuracy': [accuracy] * len(label_names),
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1 Score': f1_per_class
        }, index=label_names)

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=label_names)

        # Create a DataFrame for overall metrics
        overall_confusion_matrix_info = pd.DataFrame({
            'Accuracy': [accuracy, accuracy],
            'Precision': [precision_macro, precision_micro],
            'Recall': [recall_macro, recall_micro],
            'F1 Score': [f1_macro, f1_micro]
        }, index=['Macro', 'Micro'])

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)

        # Add labels and titles
        plt.ylabel("True Label")
        plt.xlabel(f"Predicted Label\nAccuracy: {accuracy:.4f}, Precision Macro: {precision_macro:.4f}, \
            Recall Macro: {recall_macro:.4f}, F1 Score Macro: {f1_macro:.4f}")
        plt.title("Reversal Confusion Matrix")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return class_confusion_matrix_info, overall_confusion_matrix_info

    def evaluate_reversal_type_predictions(self, true_labels, pred_labels, show=False, save_path=None):
        """Evaluates ternary classification of reversal type predictions.

        Args:
            true_labels (array-like): True labels for reversals.
            pred_labels (array-like): Predicted labels for reversals.
            show (bool, optional): Whether to display the confusion matrix. Defaults to False.
            save_path (str, optional): Path to save the confusion matrix plot. Defaults to None.

        Returns:
            tuple: A tuple containing two DataFrames - one for class-wise metrics and one for overall metrics.
        """
        label_names = ['No Reversal', 'Peak', 'Valley']

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=label_names)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision_macro = precision_score(
            true_labels, pred_labels, average='macro', labels=label_names)
        recall_macro = recall_score(
            true_labels, pred_labels, average='macro', labels=label_names)
        f1_macro = f1_score(true_labels, pred_labels,
                            average='macro', labels=label_names)
        precision_micro = precision_score(
            true_labels, pred_labels, average='micro', labels=label_names)
        recall_micro = recall_score(
            true_labels, pred_labels, average='micro', labels=label_names)
        f1_micro = f1_score(true_labels, pred_labels,
                            average='micro', labels=label_names)

        # Class-specific metrics
        precision_per_class = precision_score(
            true_labels, pred_labels, average=None, labels=label_names)
        recall_per_class = recall_score(
            true_labels, pred_labels, average=None, labels=label_names)
        f1_per_class = f1_score(true_labels, pred_labels,
                                average=None, labels=label_names)

        # Create DataFrame for class-specific metrics
        class_confusion_matrix_info = pd.DataFrame({
            'Accuracy': [accuracy] * len(label_names),
            'Precision': precision_per_class,
            'Recall': recall_per_class,
            'F1 Score': f1_per_class
        }, index=label_names)

        # Create DataFrame for overall metrics
        overall_confusion_matrix_info = pd.DataFrame({
            'Accuracy': [accuracy, accuracy],
            'Precision': [precision_macro, precision_micro],
            'Recall': [recall_macro, recall_micro],
            'F1 Score': [f1_macro, f1_micro]
        }, index=['Macro', 'Micro'])

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Reversal Confusion Matrix')
        plt.ylabel("True Label")
        plt.xlabel(
            f"Predicted Label\nAccuracy: {accuracy:.4f}, Precision Macro: {precision_macro:.4f}, \
            Recall Macro: {recall_macro:.4f}, F1 Score Macro: {f1_macro:.4f}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return class_confusion_matrix_info, overall_confusion_matrix_info

    def plot_training_curve(self, history, show=False, save_path=None):
        """Plots the training and validation curves for loss and accuracy.

        Args:
            history (dict): Dictionary containing training history with keys 
                            'loss', 'val_loss', 'binary_accuracy', and 'val_binary_accuracy'.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot training loss and validation loss
        ax1.plot(history['loss'], label='Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        rollback_epoch = history.get('rollback_epoch')
        if rollback_epoch is not None:
            ax1.axvline(x=rollback_epoch, color='r',
                        linestyle='--', label='Rollback Epoch')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()

        # Plot training accuracy and validation accuracy
        ax2.plot(history['binary_accuracy'], label='Accuracy')
        ax2.plot(history['val_binary_accuracy'], label='Validation Accuracy')
        if rollback_epoch is not None:
            ax2.axvline(x=rollback_epoch, color='r',
                        linestyle='--', label='Rollback Epoch')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend()

        # Set overall title and layout
        fig.suptitle(f'Training Curve, rollback_epoch: {rollback_epoch}')
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_online_training_curve(self, online_training_history, show=False, save_path=None):
        """Plots the online training curve for loss and accuracy over time.

        Args:
            online_training_history (dict): Dictionary containing online training history with keys 
                                            'loss' and 'binary_accuracy'.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save_path (str, optional): Path to save the plot. Defaults to None.
        """
        losses = online_training_history['loss']
        acc = online_training_history['binary_accuracy']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Online Training Curve')

        # Plot online training loss
        ax1.plot(losses, color='tab:blue')
        ax1.set_title('Online Training Loss')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Plot online training accuracy
        ax2.plot(acc, color='tab:red')
        ax2.set_title('Online Training Accuracy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def _kbar(self, open, close, high, low, pos, ax):
        """Draws a candlestick bar on the given axis for stock prices.

        Args:
            open (float): Opening price.
            close (float): Closing price.
            high (float): Highest price.
            low (float): Lowest price.
            pos (float): Position on the x-axis.
            ax (matplotlib.axes.Axes): The axis to draw on.
        """
        if close > open:
            color = 'green'   # Rise
            height = close - open
            bottom = open
        else:
            color = 'red'     # Fall
            height = open - close
            bottom = close

        # Draw the candlestick
        ax.bar(pos, height=height, bottom=bottom, width=0.6, color=color)
        ax.vlines(pos, high, low, color=color)

    def plot_days_difference_bar_chart(self, reverse_difference, show=False, save_path=None):
        """Plots a bar chart of the difference in reverse index days.

        Args:
            reverse_difference (pd.DataFrame): DataFrame containing reverse differences with columns
                                                'predicted_reverse_label', 'actual_reverse_label',
                                                'reverse_signal_correct', 'reverse_idx_difference',
                                                'actual_reverse_signals', 'predicted_reverse_signals'.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save_path (str, optional): Path to save the plot. Defaults to None.

        Returns:
            tuple: Mean absolute reverse index difference and count of differences within range.
        """
        plt.figure(figsize=(14, 6))

        # Filter out rows where no reversal was predicted or actual
        reverse_difference_filtered = reverse_difference[
            (reverse_difference['predicted_reverse_label'] != 'No reversal') &
            (reverse_difference['actual_reverse_label'] != 'No reversal')
        ]

        reverse_idx_difference = reverse_difference_filtered[
            reverse_difference_filtered['reverse_signal_correct']
        ]['reverse_idx_difference']

        # Filter differences within specified range
        reverse_in_range = reverse_idx_difference[
            (reverse_idx_difference <= self.params['reverse_idx_difference_max']) &
            (reverse_idx_difference >=
             self.params['reverse_idx_difference_min'])
        ]

        # Calculate actual and predicted reversal counts
        actual_reverse_num = reverse_difference[
            reverse_difference['actual_reverse_signals'] != 0
        ].shape[0]
        predict_reverse_num = reverse_difference[
            reverse_difference['predicted_reverse_signals'] != 0
        ].shape[0]

        # Plot bar chart
        plt.bar(x=range(len(reverse_idx_difference)),
                height=reverse_idx_difference.reset_index(drop=True))
        for idx in range(len(reverse_idx_difference)):
            plt.text(idx, reverse_idx_difference.iloc[idx], str(
                int(reverse_idx_difference.iloc[idx])), ha='center', va='bottom')

        plt.xticks(ticks=range(len(reverse_idx_difference)))
        plt.title(
            f'Bar plot of reverse idx difference \n Mean: {round(abs(reverse_idx_difference.mean()), 2)} day, \n'
            f'actual reversal num: {actual_reverse_num}, predict reversal num: {predict_reverse_num}, Predict in range: {len(reverse_in_range)}'
        )
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return round(abs(reverse_idx_difference.mean()), 2), len(reverse_in_range)

    def plot_roc_pr_curve(self, y_test_max_indices, y_preds, show=False, save_path=None):
        """Plots ROC and Precision-Recall curves.

        Args:
            y_test_max_indices (np.ndarray): Array of true binary labels.
            y_preds (np.ndarray): Array of predicted probabilities.
            show (bool, optional): Whether to display the plot. Defaults to False.
            save_path (str, optional): Path to save the plot. Defaults to None.

        Returns:
            tuple: ROC AUC and PR AUC scores.
        """
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(
            y_test_max_indices.reshape(-1), y_preds.reshape(-1).detach().numpy())
        roc_auc = auc(fpr, tpr)

        # Compute Precision-Recall curve and AUC
        precision, recall, _ = precision_recall_curve(
            y_test_max_indices.reshape(-1), y_preds.reshape(-1).detach().numpy())
        pr_auc = auc(recall, precision)

        # Create subplots for ROC and Precision-Recall curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ROC curve
        ax1.plot(fpr, tpr, label='ROC curve')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(
            f'Receiver Operating Characteristic (ROC) Curve, AUC={roc_auc:.4f}')
        ax1.legend()

        # Plot Precision-Recall curve
        ax2.plot(recall, precision, label='Precision-Recall curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve, AUC={pr_auc:.4f}')
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

        return roc_auc, pr_auc

    def execute_trades(self, trade_signals, target_dataset, initial_cash=10000, per_share_fee=0.000008, min_fee=0.01):
        """Executes trades based on signals and calculates trading details.

        Args:
            trade_signals (pd.DataFrame): DataFrame containing trade signals with columns 'Order' and relevant indices.
            target_dataset (pd.DataFrame): DataFrame containing target dataset with 'Close' prices.
            initial_cash (float, optional): Initial cash available for trading. Defaults to 10000.
            per_share_fee (float, optional): Fee per share for trading. Defaults to 0.000008.
            min_fee (float, optional): Minimum fee per trade. Defaults to 0.01.

        Returns:
            pd.DataFrame: DataFrame with detailed trade execution results including positions, cash, and profits.
        """
        def calculate_fee(shares, price, per_share_fee, min_fee):
            """Calculates the fee for a trade.

            Args:
                shares (int): Number of shares traded.
                price (float): Price per share.
                per_share_fee (float): Fee per share.
                min_fee (float): Minimum fee.

            Returns:
                float: Calculated fee.
            """
            fee = max(shares * per_share_fee * price, min_fee)
            return fee

        def execute_trade(trade_details, idx, per_share_fee, min_fee):
            """Executes a trade and updates trade details.

            Args:
                trade_details (pd.DataFrame): DataFrame with trade details.
                idx (int): Index of the trade to execute.
                per_share_fee (float): Fee per share.
                min_fee (float): Minimum fee.

            Returns:
                pd.DataFrame: Updated trade details.
            """
            order = trade_details['Order'].iloc[idx]
            position_before = trade_details['Position before trading'].iloc[idx]
            cash_before = trade_details['Cash before trading'].iloc[idx]
            close_price = trade_details['Close price'].iloc[idx]

            if order == 'Sell':
                if position_before <= 0:
                    shares_sold = 1
                else:
                    shares_sold = position_before + 1

                sell_fee = calculate_fee(
                    shares_sold, close_price, per_share_fee, min_fee)
                trade_details['Position after trading'].iloc[idx] = position_before - shares_sold
                trade_details['Cash after trading'].iloc[idx] = cash_before + \
                    close_price * shares_sold - sell_fee
                trade_details['Commission Fee'].iloc[idx] = sell_fee

            elif order == 'Buy':
                if position_before >= 0:
                    trade_details['Position after trading'].iloc[idx] = position_before + 1
                    trade_details['Cash after trading'].iloc[idx] = cash_before - close_price
                else:
                    trade_details['Position after trading'].iloc[idx] = 1
                    trade_details['Cash after trading'].iloc[idx] = cash_before - \
                        close_price * (position_before * -1 + 1)

            return trade_details

        # Initialize trade details DataFrame
        trade_details = pd.DataFrame(columns=[
            'Position before trading', 'Cash before trading', 'Close price', 'Order',
            'Position after trading', 'Cash after trading', 'Profit change',
            'Profit', 'Commission Fee', 'Value', 'Outcome'
        ], index=trade_signals.index)

        trade_details['Order'] = trade_signals['Order']
        trade_details['Close price'] = target_dataset['Close'].loc[trade_signals.index].values
        trade_details.dropna(subset=['Order'], inplace=True)
        trade_details['Position before trading'] = 0
        trade_details['Cash before trading'] = initial_cash
        trade_details['Commission Fee'] = 0
        trade_details['Value'] = initial_cash
        trade_details['Profit change'] = 0

        # Execute trades
        for idx in range(len(trade_details)):
            if idx > 0:
                trade_details['Position before trading'].iloc[idx] = trade_details['Position after trading'].iloc[idx - 1]
                trade_details['Cash before trading'].iloc[idx] = trade_details['Cash after trading'].iloc[idx - 1]

            trade_details = execute_trade(
                trade_details, idx, per_share_fee, min_fee)

        # Calculate final values and profits
        trade_details['Value'] = trade_details['Cash after trading'] + \
            trade_details['Position after trading'] * \
            trade_details['Close price']
        trade_details['Profit'] = trade_details['Value'] - initial_cash
        trade_details['Profit change'] = trade_details['Profit'] - \
            trade_details['Profit'].shift(1)

        trade_details.loc[trade_details['Profit change']
                          > 0, 'Outcome'] = 'win'
        trade_details.loc[trade_details['Profit change']
                          <= 0, 'Outcome'] = 'lose'

        return trade_details

    def execute_trades_with_stop_loss(self, trade_signals, target_dataset, initial_cash=10000, per_share_fee=0.000008, min_fee=0.01):
        def calculate_fee(shares, price, per_share_fee, min_fee):
            fee = max(shares * per_share_fee * price, min_fee)
            return fee

        def execute_trade(trade_details, idx, per_share_fee, min_fee, initial_cash):
            order = trade_details['Order'].iloc[idx]
            position_before = trade_details['Position before trading'].iloc[idx]
            cash_before = trade_details['Cash before trading'].iloc[idx]
            close_price = trade_details['Close price'].iloc[idx]

            if order == 'Sell':
                if position_before <= 0:
                    shares_sold = 1
                else:
                    shares_sold = position_before + 1

                sell_fee = calculate_fee(
                    shares_sold, close_price, per_share_fee, min_fee)
                trade_details['Position after trading'].iloc[idx] = position_before - shares_sold
                trade_details['Cash after trading'].iloc[idx] = cash_before + \
                    close_price * shares_sold - sell_fee
                trade_details['Commission Fee'].iloc[idx] = sell_fee
                trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                    trade_details['Position after trading'].iloc[idx] * \
                    trade_details['Close price'].iloc[idx]
                trade_details['Stop sell'].iloc[idx] = trade_details['Close price'].iloc[idx]*1.1
                trade_details['Stop buy'].iloc[idx] = np.nan
            elif order == 'Buy':
                if position_before >= 0:
                    shares_Buy = 1
                else:
                    shares_Buy = position_before * -1 + 1
                trade_details['Position after trading'].iloc[idx] = position_before + shares_Buy
                trade_details['Cash after trading'].iloc[idx] = cash_before - \
                    close_price * shares_Buy
                trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                    trade_details['Position after trading'].iloc[idx] * \
                    trade_details['Close price'].iloc[idx]
                trade_details['Stop buy'].iloc[idx] = trade_details['Close price'].iloc[idx]*0.9
                trade_details['Stop sell'].iloc[idx] = np.nan
            else:
                if close_price >= trade_details['Stop sell'].iloc[idx-1]:
                    trade_details['Order'].iloc[idx] = 'Stop loss from sell'
                    shares_Buy = trade_details['Position before trading'].iloc[idx-1] * -1
                    trade_details['Position after trading'].iloc[idx] = trade_details['Position before trading'].iloc[idx-1] + shares_Buy
                    trade_details['Cash after trading'].iloc[idx] = trade_details['Cash before trading'].iloc[idx -
                                                                                                              1] - close_price * shares_Buy
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx]

                elif close_price <= trade_details['Stop buy'].iloc[idx-1]:
                    trade_details['Order'].iloc[idx] = 'Stop loss from buy'
                    shares_sold = trade_details['Position before trading'].iloc[idx-1]
                    sell_fee = calculate_fee(
                        shares_sold, close_price, per_share_fee, min_fee)
                    trade_details['Position after trading'].iloc[idx] = trade_details['Position before trading'].iloc[idx-1] - shares_sold
                    trade_details['Cash after trading'].iloc[idx] = trade_details['Cash before trading'].iloc[idx -
                                                                                                              1] + close_price * shares_sold - sell_fee
                    trade_details['Commission Fee'].iloc[idx] = sell_fee
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx]

                else:
                    trade_details['Position after trading'].iloc[idx] = position_before
                    trade_details['Cash after trading'].iloc[idx] = cash_before
                    if idx > 0:
                        trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                            trade_details['Position after trading'].iloc[idx] * \
                            trade_details['Close price'].iloc[idx]
                        trade_details['Stop buy'].iloc[idx] = trade_details['Stop buy'].iloc[idx-1]
                        trade_details['Stop sell'].iloc[idx] = trade_details['Stop sell'].iloc[idx-1]

            trade_details['Profit'].iloc[idx] = trade_details['Value'].iloc[idx] - initial_cash
            if idx > 0:
                trade_details['Profit change'].iloc[idx] = trade_details['Profit'].iloc[idx] - \
                    trade_details['Profit'].iloc[idx-1]
            return trade_details

        trade_details = pd.DataFrame(columns=['Position before trading', 'Cash before trading', 'Close price', 'Order',
                                              'Position after trading', 'Cash after trading', 'Profit change',
                                              'Profit', 'Commission Fee', 'Value', 'Outcome', 'Stop buy', 'Stop sell'], index=trade_signals.index)

        trade_details['Order'] = trade_signals['Order']
        trade_details['Close price'] = target_dataset['Close'].loc[trade_signals.index].values
        trade_details['Position before trading'] = 0
        trade_details['Cash before trading'] = initial_cash
        trade_details['Commission Fee'] = 0
        trade_details['Value'] = initial_cash
        trade_details['Profit change'] = 0

        for idx in range(len(trade_details)):
            if idx > 0:
                trade_details['Position before trading'].iloc[idx] = trade_details['Position after trading'].iloc[idx - 1]
                trade_details['Cash before trading'].iloc[idx] = trade_details['Cash after trading'].iloc[idx - 1]

            trade_details = execute_trade(
                trade_details, idx, per_share_fee, min_fee, initial_cash)
        trade_details.loc[trade_details['Profit change']
                          > 0, 'Outcome'] = 'win'
        trade_details.loc[trade_details['Profit change']
                          <= 0, 'Outcome'] = 'lose'
        trade_details.dropna(subset=['Order'], inplace=True)
        return trade_details

    def execute_trades_with_stop_loss_stop_win(self, trade_signals, target_dataset, initial_cash=10000, per_share_fee=0.000008, min_fee=0.01):
        def calculate_fee(shares, price, per_share_fee, min_fee):
            fee = max(shares * per_share_fee * price, min_fee)
            return fee

        def execute_trade(trade_details, idx, per_share_fee, min_fee, initial_cash):
            order = trade_details['Order'].iloc[idx]
            position_before = trade_details['Position before trading'].iloc[idx]
            cash_before = trade_details['Cash before trading'].iloc[idx]
            close_price = trade_details['Close price'].iloc[idx]

            if order == 'Sell':
                if position_before <= 0:
                    shares_sold = 1
                else:
                    shares_sold = position_before + 1

                sell_fee = calculate_fee(
                    shares_sold, close_price, per_share_fee, min_fee)
                trade_details['Position after trading'].iloc[idx] = position_before - shares_sold
                trade_details['Cash after trading'].iloc[idx] = cash_before + \
                    close_price * shares_sold - sell_fee
                trade_details['Commission Fee'].iloc[idx] = sell_fee
                trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                    trade_details['Position after trading'].iloc[idx] * \
                    trade_details['Close price'].iloc[idx]
                trade_details['Stop sell loss'].iloc[idx] = trade_details['Close price'].iloc[idx]*1.1
                trade_details['Stop sell win'].iloc[idx] = trade_details['Close price'].iloc[idx]*0.9
                trade_details['Stop buy loss'].iloc[idx] = np.nan
                trade_details['Stop buy win'].iloc[idx] = np.nan
            elif order == 'Buy':
                if position_before >= 0:
                    shares_Buy = 1
                else:
                    shares_Buy = position_before * -1 + 1
                trade_details['Position after trading'].iloc[idx] = position_before + shares_Buy
                trade_details['Cash after trading'].iloc[idx] = cash_before - \
                    close_price * shares_Buy
                trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                    trade_details['Position after trading'].iloc[idx] * \
                    trade_details['Close price'].iloc[idx]
                trade_details['Stop buy loss'].iloc[idx] = trade_details['Close price'].iloc[idx]*0.9
                trade_details['Stop buy win'].iloc[idx] = trade_details['Close price'].iloc[idx]*1.1
                trade_details['Stop sell loss'].iloc[idx] = np.nan
                trade_details['Stop sell win'].iloc[idx] = np.nan
            else:
                if close_price >= trade_details['Stop sell loss'].iloc[idx-1]:
                    trade_details['Order'].iloc[idx] = 'Stop loss from sell'
                    shares_Buy = trade_details['Position before trading'].iloc[idx-1] * -1
                    trade_details['Position after trading'].iloc[idx] = trade_details['Position before trading'].iloc[idx-1] + shares_Buy
                    trade_details['Cash after trading'].iloc[idx] = trade_details['Cash before trading'].iloc[idx -
                                                                                                              1] - close_price * shares_Buy
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx]

                elif close_price <= trade_details['Stop buy loss'].iloc[idx-1]:
                    trade_details['Order'].iloc[idx] = 'Stop loss from buy'
                    shares_sold = trade_details['Position before trading'].iloc[idx-1]
                    sell_fee = calculate_fee(
                        shares_sold, close_price, per_share_fee, min_fee)
                    trade_details['Position after trading'].iloc[idx] = trade_details['Position before trading'].iloc[idx-1] - shares_sold
                    trade_details['Cash after trading'].iloc[idx] = trade_details['Cash before trading'].iloc[idx -
                                                                                                              1] + close_price * shares_sold - sell_fee
                    trade_details['Commission Fee'].iloc[idx] = sell_fee
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx]

                elif close_price <= trade_details['Stop sell win'].iloc[idx-1]:
                    trade_details['Order'].iloc[idx] = 'Stop win from sell'
                    shares_Buy = trade_details['Position before trading'].iloc[idx-1] * -1
                    trade_details['Position after trading'].iloc[idx] = trade_details['Position before trading'].iloc[idx-1] + shares_Buy
                    trade_details['Cash after trading'].iloc[idx] = trade_details['Cash before trading'].iloc[idx -
                                                                                                              1] - close_price * shares_Buy
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx]

                elif close_price >= trade_details['Stop buy win'].iloc[idx-1]:
                    trade_details['Order'].iloc[idx] = 'Stop win from buy'
                    shares_sold = trade_details['Position before trading'].iloc[idx-1]
                    sell_fee = calculate_fee(
                        shares_sold, close_price, per_share_fee, min_fee)
                    trade_details['Position after trading'].iloc[idx] = trade_details['Position before trading'].iloc[idx-1] - shares_sold
                    trade_details['Cash after trading'].iloc[idx] = trade_details['Cash before trading'].iloc[idx -
                                                                                                              1] + close_price * shares_sold - sell_fee
                    trade_details['Commission Fee'].iloc[idx] = sell_fee
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx]

                else:
                    trade_details['Position after trading'].iloc[idx] = position_before
                    trade_details['Cash after trading'].iloc[idx] = cash_before
                    if idx > 0:
                        trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                            trade_details['Position after trading'].iloc[idx] * \
                            trade_details['Close price'].iloc[idx]
                        trade_details['Stop buy loss'].iloc[idx] = trade_details['Stop buy loss'].iloc[idx-1]
                        trade_details['Stop buy win'].iloc[idx] = trade_details['Stop buy win'].iloc[idx-1]
                        trade_details['Stop sell loss'].iloc[idx] = trade_details['Stop sell loss'].iloc[idx-1]
                        trade_details['Stop sell win'].iloc[idx] = trade_details['Stop sell win'].iloc[idx-1]

            trade_details['Profit'].iloc[idx] = trade_details['Value'].iloc[idx] - initial_cash
            if idx > 0:
                trade_details['Profit change'].iloc[idx] = trade_details['Profit'].iloc[idx] - \
                    trade_details['Profit'].iloc[idx-1]
            return trade_details

        trade_details = pd.DataFrame(columns=['Position before trading', 'Cash before trading', 'Close price', 'Order',
                                              'Position after trading', 'Cash after trading', 'Profit change',
                                              'Profit', 'Commission Fee', 'Value', 'Outcome', 'Stop buy loss', 'Stop buy win', 'Stop sell loss', 'Stop sell win'], index=trade_signals.index)

        trade_details['Order'] = trade_signals['Order']
        trade_details['Close price'] = target_dataset['Close'].loc[trade_signals.index].values
        # trade_details.dropna(subset=['Order'], inplace=True)
        trade_details['Position before trading'] = 0
        trade_details['Cash before trading'] = initial_cash
        trade_details['Commission Fee'] = 0
        trade_details['Value'] = initial_cash
        trade_details['Profit change'] = 0

        for idx in range(len(trade_details)):
            if idx > 0:
                trade_details['Position before trading'].iloc[idx] = trade_details['Position after trading'].iloc[idx - 1]
                trade_details['Cash before trading'].iloc[idx] = trade_details['Cash after trading'].iloc[idx - 1]

            trade_details = execute_trade(
                trade_details, idx, per_share_fee, min_fee, initial_cash)

        trade_details.loc[trade_details['Profit change']
                          > 0, 'Outcome'] = 'win'
        trade_details.loc[trade_details['Profit change']
                          <= 0, 'Outcome'] = 'lose'
        trade_details.dropna(subset=['Order'], inplace=True)
        return trade_details

    def plot_trading_results(self, target_dataset, result_trade_details, test_dates, show=False, save_path=None):
        fig, ax = plt.subplots(2, 1, figsize=(
            20, 12), sharex=True, height_ratios=[3, 1])

        for idx in target_dataset.loc[test_dates[0][0]: test_dates[-1][-1]].index:
            self._kbar(target_dataset.loc[idx]['Open'], target_dataset.loc[idx]['Close'],
                       target_dataset.loc[idx]['High'], target_dataset.loc[idx]['Low'], idx, ax[0])

        for idx in result_trade_details.index:
            if result_trade_details['Order'].loc[idx] == 'Sell':
                ax[0].scatter(idx, target_dataset.loc[idx]['High']*1.005,
                              color='darkgreen', label='Sell', zorder=5, marker='v', s=100)
            elif result_trade_details['Order'].loc[idx] == 'Buy':
                ax[0].scatter(idx, target_dataset.loc[idx]['Low']*0.995,
                              color='darkorange', label='Buy', zorder=5, marker='^', s=100)
            elif result_trade_details['Order'].loc[idx] == 'Stop loss from buy':
                ax[0].scatter(idx, target_dataset.loc[idx]['High']*1.005, color='blue',
                              label='Stop loss from buy', zorder=5, marker='v', s=100)
            elif result_trade_details['Order'].loc[idx] == 'Stop loss from sell':
                ax[0].scatter(idx, target_dataset.loc[idx]['Low']*0.995, color='blue',
                              label='Stop loss from sell', zorder=5, marker='^', s=100)
            elif result_trade_details['Order'].loc[idx] == 'Stop win from buy':
                ax[0].scatter(idx, target_dataset.loc[idx]['High']*1.005, color='purple',
                              label='Stop win from buy', zorder=5, marker='v', s=100)
            elif result_trade_details['Order'].loc[idx] == 'Stop win from sell':
                ax[0].scatter(idx, target_dataset.loc[idx]['Low']*0.995, color='purple',
                              label='Stop win from sell', zorder=5, marker='^', s=100)

        ax[1].plot(result_trade_details.index,
                   result_trade_details['Profit'], label='Profit')

        ax[1].xaxis.set_major_locator(MonthLocator(interval=3))
        ax[1].xaxis.set_major_formatter(DateFormatter('%Y-%m'))

        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].set_title('Stock Price')
        ax[1].set_title('Profit')
        ax[1].legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def execute_trades_long_only(self, trade_signals, target_dataset, initial_cash=100000, per_share_fee=0.000008, min_fee=0.01):
        def calculate_fee(shares, price, per_share_fee, min_fee):
            fee = max(shares * per_share_fee * price, min_fee)
            return fee

        def execute_trade(trade_details, idx, per_share_fee, min_fee, initial_cash):
            order = trade_details['Order'].iloc[idx]
            position_before = trade_details['Position before trading'].iloc[idx]
            cash_before = trade_details['Cash before trading'].iloc[idx]
            close_price = trade_details['Close price'].iloc[idx]

            if order == 'Sell':
                if position_before <= 0:
                    shares_sold = 0
                    trade_details['Order'].iloc[idx] = np.nan
                else:
                    shares_sold = position_before

                sell_fee = calculate_fee(
                    shares_sold, close_price, per_share_fee, min_fee)
                trade_details['Position after trading'].iloc[idx] = position_before - shares_sold
                trade_details['Cash after trading'].iloc[idx] = cash_before + \
                    close_price * shares_sold - sell_fee
                trade_details['Commission Fee'].iloc[idx] = sell_fee
                trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                    trade_details['Position after trading'].iloc[idx] * \
                    trade_details['Close price'].iloc[idx]
                trade_details['Stop sell loss'].iloc[idx] = trade_details['Close price'].iloc[idx]*1.1
                trade_details['Stop sell win'].iloc[idx] = trade_details['Close price'].iloc[idx]*0.9
                trade_details['Stop buy loss'].iloc[idx] = np.nan
                trade_details['Stop buy win'].iloc[idx] = np.nan
            elif order == 'Buy':
                if cash_before <= 0:
                    shares_Buy = 0
                    trade_details['Order'].iloc[idx] = np.nan
                else:
                    shares_Buy = 1

                trade_details['Position after trading'].iloc[idx] = position_before + shares_Buy
                trade_details['Cash after trading'].iloc[idx] = cash_before - \
                    close_price * shares_Buy
                trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                    trade_details['Position after trading'].iloc[idx] * \
                    trade_details['Close price'].iloc[idx]
                trade_details['Stop buy loss'].iloc[idx] = trade_details['Close price'].iloc[idx]*0.9
                trade_details['Stop buy win'].iloc[idx] = trade_details['Close price'].iloc[idx]*1.1
                trade_details['Stop sell loss'].iloc[idx] = np.nan
                trade_details['Stop sell win'].iloc[idx] = np.nan
            else:
                trade_details['Position after trading'].iloc[idx] = position_before
                trade_details['Cash after trading'].iloc[idx] = cash_before
                if idx > 0:
                    trade_details['Value'].iloc[idx] = trade_details['Cash after trading'].iloc[idx] + \
                        trade_details['Position after trading'].iloc[idx] * \
                        trade_details['Close price'].iloc[idx]
                    trade_details['Stop buy loss'].iloc[idx] = trade_details['Stop buy loss'].iloc[idx-1]
                    trade_details['Stop buy win'].iloc[idx] = trade_details['Stop buy win'].iloc[idx-1]
                    trade_details['Stop sell loss'].iloc[idx] = trade_details['Stop sell loss'].iloc[idx-1]
                    trade_details['Stop sell win'].iloc[idx] = trade_details['Stop sell win'].iloc[idx-1]

            trade_details['Profit'].iloc[idx] = trade_details['Value'].iloc[idx] - initial_cash
            if idx > 0:
                trade_details['Profit change'].iloc[idx] = trade_details['Profit'].iloc[idx] - \
                    trade_details['Profit'].iloc[idx-1]
            return trade_details

        trade_details = pd.DataFrame(columns=['Position before trading', 'Cash before trading', 'Close price', 'Order',
                                              'Position after trading', 'Cash after trading', 'Profit change',
                                              'Profit', 'Commission Fee', 'Value', 'Outcome', 'Stop buy loss', 'Stop buy win', 'Stop sell loss', 'Stop sell win'], index=trade_signals.index)

        trade_details['Order'] = trade_signals['Order']
        trade_details['Close price'] = target_dataset['Close'].loc[trade_signals.index].values
        # trade_details.dropna(subset=['Order'], inplace=True)
        trade_details['Position before trading'] = 0
        trade_details['Cash before trading'] = initial_cash
        trade_details['Commission Fee'] = 0
        trade_details['Value'] = initial_cash
        trade_details['Profit change'] = 0

        for idx in range(len(trade_details)):
            if idx > 0:
                trade_details['Position before trading'].iloc[idx] = trade_details['Position after trading'].iloc[idx - 1]
                trade_details['Cash before trading'].iloc[idx] = trade_details['Cash after trading'].iloc[idx - 1]

            trade_details = execute_trade(
                trade_details, idx, per_share_fee, min_fee, initial_cash)

        trade_details.loc[trade_details['Profit change']
                          > 0, 'Outcome'] = 'win'
        trade_details.loc[trade_details['Profit change']
                          <= 0, 'Outcome'] = 'lose'
        trade_details.dropna(subset=['Order'], inplace=True)
        return trade_details

    def evaluate_and_generate_results(self, model, y_data, y_data_max_indices, y_preds, y_preds_max_indices, reverse_difference, history, online_training_history,
                                      reversals_data, reversals_pred_pass,
                                      pred_trade_signals, pass_pred_trade_signals, target_dataset, dates, execution_time, show=False):
        # Generate model summary
        model_summary = self.generate_model_summary(model)

        # Evaluate trend predictions and get confusion matrix info
        trend_confusion_matrix_info = self.evaluate_trend_predictions(
            y_data_max_indices, y_preds_max_indices, show=show, save_path=self.params['save_path']['trend_confusion_matrix_save_path'])
        class_reversals_confusion_matrix_info, overall_reversals_confusion_matrix_info = self.evaluate_reversals_predictions_three_type(
            reversals_data, reversals_pred_pass, show=show, save_path=self.params['save_path']['reversal_confusion_three_type_matrix_save_path'])
        class_confusion_matrix_info, overall_confusion_matrix_info = self.evaluate_reversal_type_predictions(
            reverse_difference['actual_reverse_label'], reverse_difference['predicted_reverse_label'], \
                show=show, save_path=self.params['save_path']['reversal_confusion_type_matrix_save_path'])
        pass_reversal_confusion_matrix_info = self.evaluate_reversals_predictions_two_type(
            reversals_data, reversals_pred_pass, show=show, save_path=self.params['save_path']['pass_reversal_confusion_matrix_save_path'])

        # Evaluate reversal predictions and get confusion matrix info
        # Plot ROC-PR curve and get JSON string
        roc_auc, pr_auc = self.plot_roc_pr_curve(
            y_data_max_indices, y_preds, show=show, save_path=self.params['save_path']['roc_pr_curve_save_path'])
        # Plot days difference bar chart
        reverse_idx_difference_mean, reverse_in_range_num = self.plot_days_difference_bar_chart(
            reverse_difference, show=show, save_path=self.params['save_path']['pred_days_difference_bar_chart_save_path'])
        # Plot training curve and online training curve
        self.plot_training_curve(
            history, show=show, save_path=self.params['save_path']['training_curve_save_path'])
        self.plot_online_training_curve(online_training_history, show=show,
                                        save_path=self.params['save_path']['online_training_curve_save_path'])

        # Execute trades and plot trading results
        trade_details = self.execute_trades(
            pass_pred_trade_signals, target_dataset)
        self.plot_trading_results(target_dataset, trade_details, dates, show=show,
                                  save_path=self.params['save_path']['trading_details_kbar_save_path'])
        trade_details_with_stop_loss = self.execute_trades_with_stop_loss(
            pass_pred_trade_signals, target_dataset)
        self.plot_trading_results(target_dataset, trade_details_with_stop_loss, dates, show=show,
                                  save_path=self.params['save_path']['trading_details_with_stop_loss_kbar_save_path'])
        trade_details_with_stop_loss_stop_win = self.execute_trades_with_stop_loss_stop_win(
            pass_pred_trade_signals, target_dataset)
        self.plot_trading_results(target_dataset, trade_details_with_stop_loss_stop_win, dates, show=show,
                                  save_path=self.params['save_path']['trading_details_with_stop_loss_stop_win_kbar_save_path'])
        trade_details_long_only = self.execute_trades_long_only(
            pass_pred_trade_signals, target_dataset)
        self.plot_trading_results(target_dataset, trade_details_long_only, dates, show=show,
                                  save_path=self.params['save_path']['trading_details_long_only_kbar_save_path'])

        # Convert results to JSON format
        model_summary_json = json.dumps(model_summary)
        roc_auc_json = json.dumps(roc_auc)
        pr_auc_json = json.dumps(pr_auc)
        trend_confusion_matrix_info_json = trend_confusion_matrix_info.to_json()
        class_confusion_matrix_info_json = class_confusion_matrix_info.to_json()
        overall_confusion_matrix_info_json = overall_confusion_matrix_info.to_json()
        trade_details_json = trade_details.to_json()
        trade_details_with_stop_loss_json = trade_details_with_stop_loss.to_json()
        trade_details_with_stop_loss_stop_win_json = trade_details_with_stop_loss_stop_win.to_json()
        trade_details_long_only_json = trade_details_long_only.to_json()
        reverse_difference_json = reverse_difference.to_json()
        pass_reversal_confusion_matrix_info_json = pass_reversal_confusion_matrix_info.to_json()
        class_reversals_confusion_matrix_info_json = class_reversals_confusion_matrix_info.to_json()
        overall_reversals_confusion_matrix_info_json = overall_reversals_confusion_matrix_info.to_json()

        # Prepare response dictionary
        result = {
            'usingData': self.params,
            'model_summary': model_summary_json,
            'trend_confusion_matrix_info': trend_confusion_matrix_info_json,
            'class_reversals_confusion_matrix_info': class_reversals_confusion_matrix_info_json,
            'overall_reversals_confusion_matrix_info': overall_reversals_confusion_matrix_info_json,
            'pass_reversal_confusion_matrix_info': pass_reversal_confusion_matrix_info_json,
            'class_confusion_matrix_info': class_confusion_matrix_info_json,
            'overall_confusion_matrix_info': overall_confusion_matrix_info_json,
            'roc_auc': roc_auc_json,
            'pr_auc': pr_auc_json,
            'trade_details': trade_details_json,
            'trade_details_with_stop_loss': trade_details_with_stop_loss_json,
            'trade_details_with_stop_loss_stop_win': trade_details_with_stop_loss_stop_win_json,
            'trade_details_long_only': trade_details_long_only_json,
            'execution_time': execution_time,
            'reverse_difference': reverse_difference_json,
            'reverse_idx_difference_mean': reverse_idx_difference_mean,
            'reverse_in_range_num': reverse_in_range_num,
            'y_data': y_data.tolist(),
            'y_preds': y_preds.tolist(),
        }

        return result

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
from scipy.signal import argrelextrema

class Evaluator:
    def __init__(self, params):
        self.params = params
        pass

    def get_and_plot_trend_confusion_matrix(self, y_test, y_preds, average='macro',show='False',save_path='plots/trend_confusion_matrix.png'):
        # Convert to class labels if necessary
        y_test = np.argmax(y_test.reshape(-1, y_test.shape[-1]), axis=1)
        y_preds = np.argmax(y_preds.reshape(-1, y_preds.shape[-1]), axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_preds)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Trend Confusion Matrix')

        # Calculate metrics
        precision = precision_score(y_test, y_preds, average=average)
        recall = recall_score(y_test, y_preds, average=average)
        accuracy = accuracy_score(y_test, y_preds)
        f1 = f1_score(y_test, y_preds, average=average)
        confusion_matrix_info = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [
                                             precision], 'Recall': [recall], 'F1 Score': [f1], 'confusion_matrix': [cm]})
        
        # Annotate metrics on the plot
        plt.xlabel(
            f'Predicted\n\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')
        plt.ylabel(f'Actual\n')
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
            print(f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')
        elif show=='False':
            plt.close()

        return confusion_matrix_info

    def get_and_plot_reversal_confusion_matrix(self, test_signal, pred_signal, figsize=(10, 7), average='macro', show='False', save_path='plots/reversal_confusion_matrix.png'):
        """
        This function computes the confusion matrix and classification metrics 
        (accuracy, precision, recall, and F1 score), then plots the confusion 
        matrix as a heatmap with annotated metrics.

        Parameters:
        - true_labels (ndarray): Array of true labels.
        - pred_labels (ndarray): Array of predicted labels.
        - figsize (tuple): Size of the plot.
        - average (str): The averaging method for precision, recall, and F1 score.
        """
        pred_labels = self._change_labels(pred_signal, label_type='signal', abbreviation='False')
        true_labels = self._change_labels(test_signal, label_type='signal', abbreviation='False')

        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average=average)
        recall = recall_score(true_labels, pred_labels, average=average)
        f1 = f1_score(true_labels, pred_labels, average=average)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=["Peak", "Flat", "Valley"])
        confusion_matrix_info = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [
                                precision], 'Recall': [recall], 'F1 Score': [f1], 'confusion_matrix': [cm]})
        # Plotting the confusion matrix
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Peak", "Flat", "Valley"],
            yticklabels=["Peak", "Flat", "Valley"]
        )
        
        # Adding labels and titles
        plt.ylabel("True Label")
        plt.xlabel(f"Predicted Label\n\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        plt.title("Reversal Confusion Matrix")
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
            print(f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')
        elif show=='False':
            plt.close()
        
        return confusion_matrix_info
    
    def get_and_plot_signal_confustion_matrix(self, test_trade_signals, pred_trade_signals, average='macro', show='False', save_path='plots/signal_confusion_matrix.png'):
        
        pred_labels = self._change_labels(pred_trade_signals['Signal'], label_type='trade_signal',abbreviation='False')
        true_labels = self._change_labels(test_trade_signals['Signal'], label_type='trade_signal', abbreviation='False')

        # Compute metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average=average)
        recall = recall_score(true_labels, pred_labels, average=average)
        f1 = f1_score(true_labels, pred_labels, average=average)

        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        confusion_matrix_info = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [
                                precision], 'Recall': [recall], 'F1 Score': [f1], 'confusion_matrix': [cm]})

        # Use Seaborn to draw the confusion matrix
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Buy", "Hold", "Sell"],
                    yticklabels=["Buy", "Hold", "Sell"])
        plt.ylabel('True Label')
        plt.xlabel(f'Predicted Label\n\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        plt.title('Trade Signal Confusion Matrix')
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
            print(f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}')
        elif show=='False':
            plt.close()
            
        return confusion_matrix_info

    def plot_training_curve(self, history, show='False',save_path='plots/training_curve.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Training Curve')
        # Plot loss and validation loss
        ax1.plot(history['loss'], label='Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax1.legend()

        # Plot accuracy and validation accuracy
        ax2.plot(history['binary_accuracy'], label='Accuracy')
        ax2.plot(history['val_binary_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True)
        ax2.legend()
        plt.tight_layout()
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
        elif show=='False':
            plt.close()

    def plot_online_training_curve(self, acc, losses, show='False',save_path='plots/online_training_curve.png'):
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Online Training Curve')
        # Plot loss on the second subplot
        ax1.plot(losses, color='tab:blue')
        ax1.set_title('Online Training Loss')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Plot accuracy on the first subplot
        ax2.plot(acc, color='tab:red')
        ax2.set_title('Online Training Accuracy')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0, 1])
        ax2.grid(True)

        # Adjust the layout
        plt.title('Online Training Curve')
        plt.tight_layout()
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
        elif show=='False':
            plt.close()

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
                arr[change_indices[i]:change_indices[i+1]
                    ] = arr[change_indices[i] - 1]
        return arr

    def plot_predictions(self, y_test, y_preds, filter, show='False',save_path='plots/predictions.png'):
        # Convert one-hot encoded arrays to integer labels
        y_test_labels = np.argmax(y_test, axis=-1).flatten()
        y_preds_labels = np.argmax(y_preds, axis=-1).flatten()
        if filter != 'False':
            y_preds_labels = self.remove_short_sequences(
                y_preds_labels.clone(), filter)
        plt.figure(figsize=(32, 6))
        # Plotting y_test
        plt.plot(y_test_labels, label='y_test')

        # Plotting y_preds
        plt.plot(y_preds_labels, label='y_preds')

        # Adding labels and legend
        plt.title('Predictions vs True Labels')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend(fontsize=20)
        if save_path!='False':
            plt.savefig(save_path)
        # Display the plot
        if show=='True':
            plt.show()
        elif show=='False':
            plt.close()

    def plot_trading_signals(self, data, trade_signals, x_start=0, x_stop=-1, show='False',save_path='plots/trading_details_kbar.png'):
        stock_data = data[['Open', 'High', 'Low', 'Close']
                          ].loc[data.index.isin(trade_signals['Date'])]
        stock_data['pred_signal'] = trade_signals['Signal'].values

        fig, ax = plt.subplots(figsize=(32, 6))
        for i in stock_data['pred_signal'].index[x_start:x_stop]:
            self._kbar(stock_data['Open'].loc[i], stock_data['Close'].loc[i],
                       stock_data['High'].loc[i], stock_data['Low'].loc[i], i, ax)

        self._plot_signals(trade_signals, stock_data, x_start, x_stop, ax)
        ax.set_title(
            f'Trading Details, from {stock_data.index[x_start].date()} to {stock_data.index[x_stop].date()}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_xticks(stock_data.index[x_start:x_stop])
        # ax.set_xticklabels(stock_data.index[x_start:x_stop].strftime('%Y-%m-%d'), rotation=30, ha='right', fontsize=6)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(6)
        plt.grid()
        plt.legend()
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
        elif show=='False':
            plt.close()

    def _kbar(self, open, close, high, low, pos, ax):  # for US stocks
        if close > open:
            color = 'green'   # rise
            height = close - open
            bottom = open
        else:
            color = 'red'     # fall
            height = open - close
            bottom = close
        ax.bar(pos, height=height, bottom=bottom, width=0.6, color=color)
        ax.vlines(pos, high, low, color=color)

    def _plot_signals(self, trade_signals, stock_data, x_start, x_stop, ax):
        # buy_signals = trade_signals.loc[x_start:x_stop][(
        #     trade_signals['Signal'] == 'Buy') | (trade_signals['Signal'] == 'Buy (first)')]
        buy_signals = trade_signals.loc[x_start:x_stop][(
            trade_signals['Signal'] == 1)]
        for i in buy_signals['Date']:
            if i in stock_data.index:
                ax.scatter(i, stock_data.loc[i, 'Low'] -
                           50, marker='^', color='green', s=100)

        # sell_signals = trade_signals.loc[x_start:x_stop][(
        #     trade_signals['Signal'] == 'Sell') | (trade_signals['Signal'] == 'Sell (first)')]
        sell_signals = trade_signals.loc[x_start:x_stop][(
            trade_signals['Signal'] == 0)]
        for i in sell_signals['Date']:
            if i in stock_data.index:
                ax.scatter(
                    i, stock_data.loc[i, 'High'] + 50, marker='v', color='red', s=100)

    def _change_labels(self, signal, label_type='signal', abbreviation='False'):
        if label_type == 'signal':
            if abbreviation == 'True':
                signal = np.where(signal.astype(str) == '1.0', "V", signal)
                signal = np.where(signal.astype(str) == '0.0', "F", signal)
                signal = np.where(signal.astype(str) == '-1.0', "P", signal)
            elif abbreviation == 'False':
                signal = np.where(signal.astype(str) == '1.0', "Valley", signal)
                signal = np.where(signal.astype(str) == '0.0', "Flat", signal)
                signal = np.where(signal.astype(str) == '-1.0', "Peak", signal)
        elif label_type == 'trade_signal':
            if abbreviation == 'True':
                signal = np.where(signal.astype(str) == '1.0', "B", signal)
                signal = np.where(signal.astype(str) == '0.0', "H", signal)
                signal = np.where(signal.astype(str) == '-1.0', "S", signal)
            elif abbreviation == 'False':
                signal = np.where(signal.astype(str) == '1.0', "Buy", signal)
                signal = np.where(signal.astype(str) == '0.0', "Hold", signal)
                signal = np.where(signal.astype(str) == '-1.0', "Sell", signal)
        return signal
        
    def plot_stock_data_with_signals(
        self, 
        stock_data, 
        pred_trade_signals, 
        test_trade_signals, 
        y_test_max_indices=None, 
        y_preds_original=None,  
        y_preds_max_indices=None, 
        pred_signal=None,
        test_signal=None,
        plot_type="reversal",
        show='False',
        save_path='plots/stock_data_with_signals.png'
        ):
        
        pred_signal = self._change_labels(pred_signal, label_type='signal',abbreviation='True')
        test_signal = self._change_labels(test_signal, label_type='signal', abbreviation='True')
        y_preds_original_max_indices = np.argmax(y_preds_original, axis=-1)
        
        data_filtered = stock_data.loc[pred_trade_signals['Date']]
        # Determine the subplots configuration based on plot_type
        if plot_type == "reversal":
            n_subplots = 3
            height_ratios = [5, 1, 1]
            subplots_to_use = [0, 1, 2]
        elif plot_type == "trend":
            n_subplots = 4
            height_ratios = [5, 1, 2, 1]
            subplots_to_use = [0, 3, 4, 6]
        elif plot_type == "trend_before_after":
            n_subplots = 4
            height_ratios = [5, 1, 1, 1]
            subplots_to_use = [0, 3, 5, 6]

        # Create the subplots with appropriate ratios
        fig, axes = plt.subplots(
            n_subplots, 1, figsize=(20, 15), gridspec_kw={"height_ratios": height_ratios}, sharex=True
        )

        # Extract data
        dates = data_filtered.index
        close_prices = data_filtered["Close"]
        moving_average = data_filtered["MA"]

        
        # Plot main price and moving average data on the first subplot
        ax1 = axes[0]
        for i in data_filtered.index:
            self._kbar(
                data_filtered["Open"].loc[i],
                data_filtered["Close"].loc[i],
                data_filtered["High"].loc[i],
                data_filtered["Low"].loc[i],
                i,
                ax1,
            )
        ax1.plot(dates, moving_average, label="Moving Average", linestyle="--", color="red")

        # Identify and plot local maxima and minima for price and moving average
        local_maxima = argrelextrema(close_prices.values, np.greater, order=20)[0]
        local_minima = argrelextrema(close_prices.values, np.less, order=20)[0]
        malocal_maxima = argrelextrema(moving_average.values, np.greater, order=20)[0]
        malocal_minima = argrelextrema(moving_average.values, np.less, order=20)[0]

        ax1.scatter(
            dates[local_maxima],
            close_prices[local_maxima],
            color="green",
            label="Local Maxima",
        )
        ax1.scatter(
            dates[local_minima],
            close_prices[local_minima],
            color="orange",
            label="Local Minima",
        )
        ax1.scatter(
            data_filtered.iloc[malocal_maxima].index,
            moving_average.iloc[malocal_maxima],
            label="Local MA Maxima",
            color="darkgreen",
            zorder=5,
        )
        ax1.scatter(
            data_filtered.iloc[malocal_minima].index,
            moving_average.iloc[malocal_minima],
            label="Local MA Minima",
            color="darkorange",
            zorder=5,
        )

        ax1.set_title("Stock Data with Moving Average")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)

        # Conditional plotting based on plot_type
        if plot_type == "reversal":
            fig.suptitle("Stock Data with Reversal Signals")
            # Actual trend reversal
            ax2 = axes[1]
            ax2.plot(test_trade_signals["Date"], test_trade_signals["Signal"])
            ax2.set_title("Actual Trend Reversal")
            ax2.set_xlabel("Date")
            ax2.set_yticks([-1, 0, 1])
            ax2.set_ylim(-1.1, 1.1)
            ax2.grid(True)

            # Predicted trend reversal
            ax3 = axes[2]
            ax3.plot(pred_trade_signals["Date"], pred_trade_signals["Signal"])
            ax3.set_title("Predicted Trend Reversal")
            ax3.set_xlabel("Date")
            ax3.set_yticks([-1, 0, 1])
            ax3.set_ylim(-1.1, 1.1)
            ax3.grid(True)

        elif plot_type == "trend":
            fig.suptitle("Stock Data with Trend Signals")
            # Actual trend
            ax4 = axes[1]
            for i in range(0, y_test_max_indices.shape[0]):
                ax4.plot(dates[i*y_test_max_indices.shape[1]:(i+1)*y_test_max_indices.shape[1]], y_test_max_indices[i, :])
                ax4.text(dates[i*y_test_max_indices.shape[1]], 1, test_signal[i], fontsize=12)

            ax4.set_title("Actual Trend")
            ax4.set_xlabel("Date")
            ax4.set_yticks([0, 0.5, 1])
            ax4.set_ylim(-0.1, 1.1)
            ax4.grid(True)

            # Predicted trend
            ax5 = axes[2]
            for i in range(0, y_preds_original.shape[0]):
                ax5.plot(dates[i*y_preds_original.shape[1]:(i+1)*y_preds_original.shape[1]], y_preds_original[i, :, 1])
                ax5.text(dates[i*y_preds_original.shape[1]], 1, pred_signal[i], fontsize=12)
            ax5.set_title("Predicted Trend (Original)")
            ax5.set_xlabel("Date")
            ax5.set_yticks([0, 0.5, 1])
            ax5.set_ylim(-0.1, 1.1)
            ax5.grid(True)

            # Predicted trend with filtering
            ax7 = axes[3]
            for i in range(0, y_preds_max_indices.shape[0]):
                ax7.plot(dates[i*y_preds_max_indices.shape[1]:(i+1)*y_preds_max_indices.shape[1]], y_preds_max_indices[i, :])
                ax7.text(dates[i*y_preds_max_indices.shape[1]], 1, pred_signal[i], fontsize=12)
            ax7.set_title("Predicted Trend (Filtered)")
            ax7.set_xlabel("Date")
            ax7.set_yticks([0, 0.5, 1])
            ax7.set_ylim(-0.1, 1.1)
            ax7.grid(True)

        elif plot_type == "trend_before_after":
            fig.suptitle("Stock Data with Trend Signals")
            # Actual trend
            ax4 = axes[1]
            for i in range(0, y_test_max_indices.shape[0]):
                ax4.plot(dates[i*y_test_max_indices.shape[1]:(i+1)*y_test_max_indices.shape[1]], y_test_max_indices[i, :])
                ax4.text(dates[i*y_test_max_indices.shape[1]], 1, test_signal[i], fontsize=12)

            ax4.set_title("Actual Trend")
            ax4.set_xlabel("Date")
            ax4.set_yticks([0, 0.5, 1])
            ax4.set_ylim(-0.1, 1.1)
            ax4.grid(True)

            # Predicted trend with basic filtering
            ax6 = axes[2]
            for i in range(0, y_preds_original_max_indices.shape[0]):
                ax6.plot(dates[i*y_preds_original_max_indices.shape[1]:(i+1)*y_preds_original_max_indices.shape[1]], y_preds_original_max_indices[i, :])
                ax6.text(dates[i*y_preds_original_max_indices.shape[1]], 1, pred_signal[i], fontsize=12)

            ax6.set_title("Predicted Trend (Before Filter)")
            ax6.set_xlabel("Date")
            ax6.set_yticks([0, 0.5, 1])
            ax6.set_ylim(-0.1, 1.1)
            ax6.grid(True)

            # Predicted trend after additional filtering
            ax7 = axes[3]
            for i in range(0, y_preds_max_indices.shape[0]):
                ax7.plot(dates[i*y_preds_max_indices.shape[1]:(i+1)*y_preds_max_indices.shape[1]], y_preds_max_indices[i, :])
                ax7.text(dates[i*y_preds_max_indices.shape[1]], 1, pred_signal[i], fontsize=12)
            ax7.set_title("Predicted Trend (After Filter)")
            ax7.set_yticks([0, 0.5, 1])
            ax7.set_ylim(-0.1, 1.1)
            ax7.grid(True)

        plt.tight_layout()
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
        elif show=='False':
            plt.close()

    def find_closest_date(self, pred_trade_signals, test_trade_signals, in_advance_lim=10):
        pred_trade_signals_filtered = pred_trade_signals[pred_trade_signals['Signal'].notna() & (
            pred_trade_signals['Signal'] != 0)]
        test_trade_signals_filtered = test_trade_signals[test_trade_signals['Signal'].notna() & (
            test_trade_signals['Signal'] != 0)]
        pred_trade_signals_filtered = pred_trade_signals.iloc[np.nonzero(pred_trade_signals['Signal'].values)[0][0]:np.nonzero(pred_trade_signals['Signal'].values)[0][-1]+1]
        test_trade_signals_filtered = test_trade_signals.iloc[np.nonzero(test_trade_signals['Signal'].values)[0][0]:np.nonzero(test_trade_signals['Signal'].values)[0][-1]+1]
        
        # Creating a new DataFrame to store the results
        pred_days_difference_results = pred_trade_signals_filtered.copy()
        pred_days_difference_results['ClosestDateInTest'] = pd.NaT
        pred_days_difference_results['DaysDifference'] = pd.NA

        # Iterating through each row in pred_days_difference_results to find the closest date and days difference
        for index, row in pred_days_difference_results.iterrows():
            signal, pred_date = row['Signal'], row['Date']
            same_signal_df = test_trade_signals_filtered[(test_trade_signals_filtered['Signal'] == signal) & (test_trade_signals_filtered['Signal']!=0)].copy(
            )

            if not same_signal_df.empty:
                same_signal_df['DateDifference'] = (
                    same_signal_df['Date'] - pred_date)
                closest_date = same_signal_df.loc[same_signal_df['DateDifference'].abs(
                ).idxmin()]
                pred_days_difference_results.at[index,
                                                'ClosestDateInTest'] = closest_date['Date']
                pred_days_difference_results.at[index,
                                                'DaysDifference'] = closest_date['DateDifference'].days
        pred_days_difference_results.dropna(inplace=True)
        pred_days_difference_abs_mean = pred_days_difference_results['DaysDifference'].abs().mean()
        pred_in_advance = pred_days_difference_results[(pred_days_difference_results['DaysDifference'] > 0) & (
            pred_days_difference_results['DaysDifference'] <= in_advance_lim)].shape[0]
        
        return pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance

    def plot_days_difference_bar_chart(self, pred_days_difference_results, pred_days_difference_mean, pred_in_advance, show='False',save_path='plots/pred_days_difference_bar_chart.png'):
        # Create bar plot
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(pred_days_difference_results)),
                        pred_days_difference_results['DaysDifference'], color='blue', alpha=0.7)

        for idx in range(len(pred_days_difference_results)):
            plt.text(idx, pred_days_difference_results['DaysDifference'].iloc[idx], str(pred_days_difference_results['DaysDifference'].iloc[idx]), ha='center', va='bottom')
            plt.text(idx, pred_days_difference_results['DaysDifference'].iloc[idx]+15, str(pred_days_difference_results['Date'].iloc[idx].date()), ha='center', va='bottom', rotation=90)

        plt.title(f'Bar plot of pred days difference results\nMean: {pred_days_difference_mean:.4f}\nPredict in advance: {pred_in_advance}')
        plt.xlabel('Index')
        plt.ylabel('Difference Value')
        plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        if save_path!='False':
            plt.savefig(save_path)
        if show=='True':
            plt.show()
            print(f'Average Difference: {pred_days_difference_mean}, Predict in advance: {pred_in_advance}')
        elif show=='False':
            plt.close()
        return pred_days_difference_mean, pred_in_advance

    def plot_roc_pr_curve(self, y_test, y_preds, show='False',save_path='plots/roc_pr_curve.png'):
        # Compute ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_test.argmax(dim=-1).flatten(), y_preds.argmax(dim=-1).flatten())
        roc_auc = auc(fpr, tpr)
        # Compute Precision-Recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_test.argmax(dim=-1).flatten(), y_preds.argmax(dim=-1).flatten())
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ROC curve
        ax1.plot(fpr, tpr, label='ROC curve')
        ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Add diagonal dashed line
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'Receiver Operating Characteristic (ROC) Curve, AUC={roc_auc:.4f}')
        ax1.legend()

        # Plot Precision-Recall curve
        ax2.plot(recall, precision, label='Precision-Recall curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()
        # Adjust spacing between subplots
        plt.tight_layout()
        if save_path!='False':
            plt.savefig(save_path)
        # Show the plot
        if show=='True':
            plt.show()
            print(f'ROC AUC: {roc_auc:.4f}')
        elif show=='False':
            plt.close()

        return roc_auc

    def perform_backtesting(self, stock_data, trade_signals):
        trade_strategy = self.params['trade_strategy']

        buffer = StringIO()
        sys.stdout = buffer

        # Initialize cerebro engine
        cerebro = bt.Cerebro()
        # Create a data feed from stock data
        data_feed = bt.feeds.PandasData(dataname=stock_data)
        # Add data feed to cerebro
        cerebro.adddata(data_feed)

        # Define and add strategy
        class SignalStrategy(bt.Strategy):
            def __init__(self):
                # Map dates to signals for quick lookup
                self.signal_dict = \
                    dict((pd.Timestamp(date).to_pydatetime().date(), signal)
                         for date, signal in zip(trade_signals['Date'],
                                                 trade_signals['Signal']))

            def log(self, txt, dt=None):
                # Logging function for this strategy
                dt = dt or self.datas[0].datetime.date(0)
                print(f'{dt.isoformat()}, {txt}')

            def next(self):
                # Get the current date
                current_date = self.datas[0].datetime.date(0)
                # Check if there's a signal for this date
                signal = self.signal_dict.get(current_date)
                current_price = self.datas[0].open[0]*1.005

                if trade_strategy == 'single':
                    # Original single share buy/sell logic
                    if signal == 'Buy (first)' or signal == 'Buy (last)':
                        # Buy logic
                        self.buy(size=1)
                        self.log("SINGLE BUY EXECUTED")
                    elif signal == 'Sell (first)' or signal == 'Sell (last)':
                        # Sell logic
                        self.sell(size=1)
                        self.log("SINGLE SELL EXECUTED")
                    elif signal == 'Buy':
                        # Buy logic
                        self.buy(size=2)
                        self.log("DOUBLE BUY EXECUTED")
                    elif signal == 'Sell':
                        # Sell logic
                        self.sell(size=2)
                        self.log("DOUBLE SELL EXECUTED")

                elif trade_strategy == 'all':
                    # Buy/Sell as many shares as possible
                    if signal == 'Buy (first)' or signal == 'Buy (last)':
                        cash = self.broker.getcash()
                        size_to_buy = int(cash / current_price*1.005)  # Only whole shares
                        self.buy(size=size_to_buy)
                        self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                    elif signal == 'Sell (first)' or signal == 'Sell (last)':
                        cash = self.broker.getcash()
                        size_to_sell = int(cash / current_price*1.005)
                        self.sell(size=size_to_sell)
                        self.log(f"SELL EXECUTED, size_to_sell:{size_to_sell}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                    elif signal == 'Buy':
                        current_position = np.absolute(self.getposition(self.datas[0]).size)
                        cash = self.broker.getcash()
                        if cash > (current_position * current_price*1.005):
                            size_to_buy = np.absolute(current_position)
                            self.buy(size=size_to_buy)
                            self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                            cash = self.broker.getcash() - current_position*current_price*1.005
                            size_to_buy = int(cash / current_price*1.005)  # Only whole shares
                            self.buy(size=size_to_buy)
                            self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                        else:
                            size_to_buy = int(cash / current_price*1.005)
                            self.buy(size=size_to_buy)
                            self.log(f"BUY EXECUTED, size_to_buy:{size_to_buy}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")
                    elif signal == 'Sell':
                        current_position = np.absolute(self.getposition(self.datas[0]).size)
                        size_to_sell = current_position*2
                        self.sell(size=size_to_sell)
                        self.log(f"SELL EXECUTED, size_to_sell:{size_to_sell}, signal:{signal}, cash:{self.broker.getcash()}, position:{self.getposition(self.datas[0]).size}")

            def notify_order(self, order):
                if order.status in [order.Completed]:
                    cash = self.broker.getcash()
                    value = self.broker.getvalue()
                    if order.isbuy():
                        self.log(
                            f'BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}, Cash: {cash}, Value: {value}')
                    elif order.issell():
                        self.log(
                            f'SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Commission: {order.executed.comm}, Cash: {cash}, Value: {value}')

        # Add strategy to cerebro
        cerebro.addstrategy(SignalStrategy)
        # Set initial cash, commission, etc.
        cerebro.broker.setcash(10000.0)
        cerebro.broker.setcommission(commission=0.005)
        # You can add more code here to analyze the results
        # Add analyzers to cerebro
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
        # Run the backtest
        strategies = cerebro.run()
        backtesting_report = dict()
        # Extracting and displaying results
        strategy = strategies[0]
        backtesting_report['sharpe_ratio'] = strategy.analyzers.sharpe_ratio.get_analysis(
        )
        backtesting_report['drawdown'] = strategy.analyzers.drawdown.get_analysis()
        backtesting_report['trade_analyzer'] = strategy.analyzers.trade_analyzer.get_analysis(
        )
        backtesting_report['final_value'] = cerebro.broker.getvalue()
        backtesting_report['pnl'] = backtesting_report['final_value'] - \
            cerebro.broker.startingcash
        backtesting_report['pnl_pct'] = backtesting_report['pnl'] / \
            cerebro.broker.startingcash * 100.0
        backtesting_report['total_return'] = backtesting_report['final_value'] / \
            cerebro.broker.startingcash

        # Plotting the results
        sys.stdout = sys.__stdout__
        trade_summary = buffer.getvalue()
        buffer.close()
        # img = cerebro.plot(style='candlestick', iplot=False, volume=False, savefig=True)
        # img[0][0].savefig(f'plots/backtesting_kbar.png')
        return strategies, backtesting_report, trade_summary

    def generate_model_summary(self, model):
        total = sum([param.nelement() for param in model.parameters()])
        model_summary = f'{model}, \nNumber of parameter: {total}'
        return model_summary

    def generate_numericale_data(self, model, y_test, y_preds, test_signal, pred_signal, test_trade_signals, pred_trade_signals, stock_data, execution_time):
        model_summary = self.generate_model_summary(model)
        trend_confusion_matrix_info = self.get_and_plot_trend_confusion_matrix(
            y_test, y_preds, average='macro', show='False', save_path='False')
        reversed_trend_confusion_matrix_info = self.get_and_plot_reversal_confusion_matrix(
            test_signal, pred_signal, average='macro', show='False', save_path='False')
        signal_confusion_matrix_info = self.get_and_plot_signal_confustion_matrix(
            test_trade_signals, pred_trade_signals, average='macro', show='False', save_path='False')
        roc_auc = self.plot_roc_pr_curve(
            y_test, y_preds, show='False', save_path='False')
        pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance = self.find_closest_date(
            pred_trade_signals, test_trade_signals)
        
        backtesting_report, trade_summary = '', ''
        # backtest_results, backtesting_report, trade_summary = self.perform_backtesting(
        #     stock_data, pred_trade_signals)
        # def convert_dict(obj):
        #     if isinstance(obj, dict):
        #         return {k: convert_dict(v) for k, v in obj.items()}
        #     elif isinstance(obj, list):
        #         return [convert_dict(v) for v in obj]
        #     elif isinstance(obj, (np.int32, np.int64)):  # Add other NumPy types as needed
        #         return int(obj)
        #     elif isinstance(obj, np.float32):  # Example for NumPy float
        #         return float(obj)
        #     else:
        #         return obj
        # backtesting_report = convert_dict(backtesting_report)
        
        return model_summary, trend_confusion_matrix_info, reversed_trend_confusion_matrix_info, signal_confusion_matrix_info, roc_auc, pred_days_difference_results, pred_days_difference_abs_mean, pred_in_advance, backtesting_report, trade_summary, execution_time
    
    def get_plots(self, y_test, y_preds, y_preds_original, test_trade_signals, pred_trade_signals, stock_data, history, online_training_acc, online_training_losses, pred_days_difference_results, pred_days_difference_mean, pred_in_advance, y_test_max_indices=None, y_preds_max_indices=None, pred_signal=None, test_signal=None, show='False'):
        self.plot_training_curve(history, show=show, save_path=self.params['save_path']['training_curve_save_path'])
        self.plot_online_training_curve(online_training_acc, online_training_losses, show=show, save_path=self.params['save_path']['online_training_curve_save_path'])
        self.plot_predictions(y_test, y_preds, filter='False', show=show, save_path=self.params['save_path']['predictions_save_path'])
        self.get_and_plot_reversal_confusion_matrix(test_signal, pred_signal, show=show, save_path=self.params['save_path']['reversal_confusion_matrix_save_path'])
        self.get_and_plot_trend_confusion_matrix(y_test, y_preds, show=show, save_path=self.params['save_path']['trend_confusion_matrix_save_path'])
        self.get_and_plot_signal_confustion_matrix(test_trade_signals, pred_trade_signals, show=show, save_path=self.params['save_path']['signal_confusion_matrix_save_path'])
        self.plot_roc_pr_curve(y_test, y_preds, show=show, save_path=self.params['save_path']['roc_pr_curve_save_path'])
        self.plot_stock_data_with_signals(stock_data=stock_data, pred_trade_signals=pred_trade_signals, test_trade_signals=test_trade_signals, y_test_max_indices=y_test_max_indices, y_preds_original=y_preds_original, y_preds_max_indices=y_preds_max_indices, pred_signal=pred_signal, test_signal=test_signal, plot_type="reversal", show='False', save_path=self.params['save_path']['stock_data_with_signals_reversal_save_path'])
        self.plot_stock_data_with_signals(stock_data=stock_data, pred_trade_signals=pred_trade_signals, test_trade_signals=test_trade_signals, y_test_max_indices=y_test_max_indices, y_preds_original=y_preds_original, y_preds_max_indices=y_preds_max_indices, pred_signal=pred_signal, test_signal=test_signal, plot_type="trend", show='False', save_path=self.params['save_path']['stock_data_with_signals_trend_save_path'])
        self.plot_stock_data_with_signals(stock_data=stock_data, pred_trade_signals=pred_trade_signals, test_trade_signals=test_trade_signals, y_test_max_indices=y_test_max_indices, y_preds_original=y_preds_original, y_preds_max_indices=y_preds_max_indices, pred_signal=pred_signal, test_signal=test_signal, plot_type="trend_before_after", show='False', save_path=self.params['save_path']['stock_data_with_signals_trend_before_after_save_path'])
        self.plot_days_difference_bar_chart(pred_days_difference_results, pred_days_difference_mean, pred_in_advance, show=show, save_path=self.params['save_path']['pred_days_difference_bar_chart_save_path'])
        self.plot_trading_signals(data=stock_data, trade_signals=pred_trade_signals, x_start=0, x_stop=-1, show=show, save_path=self.params['save_path']['trading_details_kbar_save_path'])
        
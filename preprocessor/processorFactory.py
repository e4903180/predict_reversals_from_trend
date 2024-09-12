import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class CleanerBase(ABC):
    """Abstract base class for data processors that handle data cleaning tasks."""

    @abstractmethod
    def check(self, data):
        """Check the data for issues such as missing values.

        Args:
            data (pd.DataFrame): The input data to be checked.

        Returns:
            pd.Series: A series indicating the number of issues found in each column.
        """
        pass

    @abstractmethod
    def clean(self, data):
        """Clean the data by addressing the identified issues.

        Args:
            data (pd.DataFrame): The input data to be cleaned.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        pass


class CleanerMissingValue(CleanerBase):
    """Concrete class for checking and handling missing data in a DataFrame."""

    def check(self, data):
        """Check for missing data in the DataFrame.

        Args:
            data (pd.DataFrame): The input data to check for missing values.

        Returns:
            pd.Series: A series indicating the number of missing values in each column.
        """
        return data.isnull().sum()

    def clean(self, data, strategy='auto'):
        """Handle missing data based on the chosen strategy.

        Args:
            data (pd.DataFrame): The input data to clean.
            strategy (str): The strategy to handle missing data. Options include:
                - 'auto': Automatically remove rows with missing values at the start and fill remaining with forward fill.
                - 'drop': Drop rows with any missing values.
                - 'fillna': Fill missing values with the previous valid value (forward fill).
                - 'none': Do nothing, leave the data unchanged.

        Returns:
            pd.DataFrame: The cleaned data.

        Raises:
            ValueError: If an invalid strategy is provided.
        """
        if strategy == 'auto':
            # Remove rows with missing values at the start of the DataFrame
            while data.iloc[0].isnull().any():
                data = data.iloc[1:]
            # Fill remaining missing values with forward fill method
            data.fillna(method='ffill', inplace=True)
        elif strategy == 'drop':
            # Drop rows with any missing values
            data.dropna(inplace=True)
        elif strategy == 'fillna':
            # Fill missing values with the previous valid value
            data.fillna(method='ffill', inplace=True)
        elif strategy == 'none':
            # Do nothing, leave the data unchanged
            pass
        else:
            raise ValueError("Invalid strategy provided.")
        return data


class ProcessorFactory:
    """Factory class to create data processors and standardize data."""

    @staticmethod
    def get_cleaner(clean_type, *args, **kwargs):
        """Create a data cleaner based on the provided type.

        Args:
            clean_type (str): The type of cleaner to create (e.g., 'MissingData').
            *args: Additional positional arguments for the cleaner.
            **kwargs: Additional keyword arguments for the cleaner.

        Returns:
            CleanerBase: An instance of the requested cleaner.

        Raises:
            ValueError: If the provided clean_type is not recognized.
        """
        if clean_type == "MissingData":
            return CleanerMissingValue(*args, **kwargs)
        else:
            raise ValueError(f"Processor type {clean_type} not recognized.")

    @staticmethod
    def get_standardize_method(data, method='MinMaxScaler'):
        """Standardize the data using the specified method.

        Args:
            data (pd.DataFrame): The data to standardize.
            method (str): The method to use for standardization ('StandardScaler', 'MinMaxScaler').

        Returns:
            np.ndarray: The standardized data.

        Raises:
            ValueError: If the provided method is not recognized.
        """
        if method == 'StandardScaler':
            scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaler method: {method}.")
        return scaler.fit_transform(data)

    @staticmethod
    def split_datasets(data, train_split_ratio=0.7, val_split_ratio=0.1, target_col="Trend", feature_cols=None):
        """Standardize the data and split it into training, validation, and test sets.

        Args:
            data (pd.DataFrame): The data to standardize and split.
            train_split_ratio (float): The ratio of data to be used for training.
            val_split_ratio (float): The ratio of data to be used for validation.D
            target_col (str): The name of the target column.
            feature_cols (list of str): List of feature column names. If None, all columns are used.

        Returns:
            tuple: A tuple containing the training, validation, and test datasets 
            (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        if not feature_cols:
            feature_cols = data.columns.to_list()

        x_data = data[feature_cols]
        y_data = data[target_col]
        train_split_idx = int(len(x_data) * train_split_ratio)
        val_split_idx = int(len(x_data) * (train_split_ratio + val_split_ratio))
        X_train = x_data.iloc[:train_split_idx]
        y_train = y_data.iloc[:train_split_idx]
        X_val = x_data.iloc[train_split_idx:val_split_idx]
        y_val = y_data.iloc[train_split_idx:val_split_idx]
        X_test = x_data.iloc[val_split_idx:]
        y_test = y_data.iloc[val_split_idx:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def standardize_and_split_data(x_data, y_data, look_back, predict_steps, slide_steps=1):
        """Prepare the data for multi-step prediction and apply standardization within each sliding window.

        Args:
            x_data (pd.DataFrame): The input features data.
            y_data (pd.DataFrame): The target data.
            look_back (int): The number of time steps to look back.
            predict_steps (int): The number of steps to predict forward.
            slide_steps (int): The sliding window steps.

        Returns:
            tuple: A tuple containing the processed data arrays (x_data_multistep, y_data_multistep) 
            and corresponding dates (x_date, y_date).
        """
        x_date = []
        y_date = []
        x_data_multistep = []
        y_data_multistep = []

        for i in range(0, len(x_data) - look_back - predict_steps + 1, slide_steps):
            x_date.append(x_data.index[i:i + look_back])
            y_date.append(x_data.index[i + look_back:i + look_back + predict_steps])
            x_window = x_data.iloc[i:i + look_back].values
            y_window = y_data.iloc[i + look_back:i + look_back + predict_steps].values
            x_window_standardized = ProcessorFactory.get_standardize_method(x_window)
            x_data_multistep.append(x_window_standardized)
            y_data_multistep.append(y_window)

        return np.array(x_data_multistep), np.array(y_data_multistep), np.array(x_date), np.array(y_date)

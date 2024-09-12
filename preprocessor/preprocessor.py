import yfinance as yf
from torch import from_numpy
import torch
from preprocessor.processorFactory import ProcessorFactory
from preprocessor.featureFactory import FeatureFactory


class Preprocessor:
    """Class for fetching, processing, and preparing model data."""

    def __init__(self, params, start_date=None, end_date=None):
        """Initializes the Preprocessor with parameters and date range.

        Args:
            params (dict): Parameters for data processing.
            start_date (str): Start date for data fetching.
            end_date (str): End date for data fetching.
        """
        self.params = params
        self.start_date = start_date
        self.end_date = end_date

    def fetch_stock_data(self, stock_symbol, start_date=None, end_date=None):
        """Fetch stock data from Yahoo Finance.

        Args:
            stock_symbol (str): The stock symbol to fetch data for.
            start_date (str): Start date for data fetching.
            end_date (str): End date for data fetching.

        Returns:
            pd.DataFrame: The fetched stock data.
        """
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        return yf.download(stock_symbol, start=self.start_date, end=self.end_date)

    def add_feature(self, data, feature_type, *args, **kwargs):
        """Add a feature to the data using the specified feature type.

        Args:
            data (pd.DataFrame): Input data to which the feature should be added.
            feature_type (str): The type of feature to compute.

        Returns:
            pd.DataFrame: Data with the new feature added.
        """
        feature = FeatureFactory.get_feature(feature_type)
        return feature.compute(data, *args, **kwargs)

    def add_data_cleaner(self, data, clean_type='MissingData', strategy='auto'):
        """Method to check and clean the data using a specific processor.

        Args:
            data (pd.DataFrame): Input data to be checked and cleaned.
            clean_type (str): The type of cleaning processor to use.
            strategy (str): The strategy for handling missing data.

        Returns:
            pd.DataFrame: Cleaned data.
            pd.Series: Series indicating the number of issues found in each column.
        """
        processor = ProcessorFactory.get_cleaner(clean_type)
        issues = processor.check(data)
        data = processor.clean(data, strategy=strategy)
        return data, issues

    def process_data(self, data, train_split_ratio=0.7, val_split_ratio=0.1, target_col="Trend",
                     feature_cols=None, look_back=64, predict_steps=16, 
                     train_slide_steps=1, test_slide_steps=16):
        """Use ProcessorFactory to standardize and split the data, and prepare it for multi-step prediction if required.

        Args:
            data (pd.DataFrame): The data to process.
            train_split_ratio (float): The ratio of data to use for training.
            val_split_ratio (float): The ratio of data to use for validation.
            target_col (str): The target column name.
            feature_cols (list): The list of feature column names.
            look_back (int): The number of time steps to look back.
            predict_steps (int): The number of steps to predict forward.
            train_slide_steps (int): The sliding window steps for training data.
            test_slide_steps (int): The sliding window steps for test data.

        Returns:
            tuple: A tuple containing processed training, validation, and test data along with corresponding dates.
        """
        X_train, y_train, X_val, y_val, X_test, y_test = ProcessorFactory.split_datasets(
            data, train_split_ratio, val_split_ratio, target_col, feature_cols)

        if look_back and predict_steps:
            X_train, y_train, train_dates, _ = ProcessorFactory.standardize_and_split_data(
                X_train, y_train, look_back, predict_steps, train_slide_steps)
            X_test, y_test, _, test_dates = ProcessorFactory.standardize_and_split_data(
                X_test, y_test, look_back, predict_steps, test_slide_steps)
            X_val, y_val, _, val_dates = ProcessorFactory.standardize_and_split_data(
                X_val, y_val, look_back, predict_steps, train_slide_steps)
            X_train = from_numpy(X_train).float()
            y_train = from_numpy(y_train).float()
            X_val = from_numpy(X_val).float()
            y_val = from_numpy(y_val).float()
            X_test = from_numpy(X_test).float()
            y_test = from_numpy(y_test).float()
            return X_train, y_train, X_val, y_val, X_test, y_test, train_dates, test_dates, val_dates
        else:
            raise ValueError(
                "Invalid look_back or predict_steps provided for data preparation.")

    def change_values_after_first_reverse_point(self, y: torch.Tensor):
        """Change values in the tensor after the first reverse point.

        Args:
            y (torch.Tensor): The tensor containing target values.

        Returns:
            torch.Tensor: Modified tensor with values changed after the first reverse point.
        """
        y_copy = y.clone().detach()
        modified_y = torch.zeros_like(y_copy)
        for idx, sub_y in enumerate(y_copy):
            array = sub_y.numpy()
            transition_found = False
            for i in range(1, len(array)):
                if not (array[i] == array[i-1]).all():
                    array[i:] = array[i]
                    transition_found = True
                    break
            if not transition_found:
                array = sub_y.numpy()
            
            modified_y[idx] = torch.tensor(array)
        return modified_y
        
    def get_datasets(self):
        """Fetch, process, and prepare multiple datasets for model training and evaluation.

        This method handles the entire workflow of fetching stock data for multiple symbols, 
        applying features, cleaning the data, and preparing the data for training, validation, 
        and testing.

        Returns:
            tuple: A tuple containing the following:
                - X_train (torch.Tensor): The processed training feature data.
                - y_train (torch.Tensor): The processed training labels.
                - X_val (torch.Tensor): The processed validation feature data.
                - y_val (torch.Tensor): The processed validation labels.
                - X_test (torch.Tensor): The processed test feature data.
                - y_test (torch.Tensor): The processed test labels.
                - train_dates (list): Dates corresponding to the training data.
                - test_dates (list): Dates corresponding to the test data.
                - val_dates (list): Dates corresponding to the validation data.
                - test_dataset (pd.DataFrame): The raw test dataset.
        """
        def get_symbol_dataset(symbol):
            """Fetch and process data for a specific symbol.

            Args:
                symbol (str): The stock symbol to fetch data for.

            Returns:
                pd.DataFrame: The processed dataset with features added and cleaned.
            """
            dataset = self.fetch_stock_data(symbol, self.params['start_date'], self.params['stop_date'])
            for feature_params in self.params['features_params']:
                feature_type = feature_params["type"]
                dataset = self.add_feature(dataset, feature_type, **feature_params)
            dataset, issues_detected = self.add_data_cleaner(dataset,
                clean_type=self.params['data_cleaning']['clean_type'], strategy=self.params['data_cleaning']['strategy'])
            return dataset

        def process_datasets(symbols):
            """Process datasets for multiple symbols.

            Args:
                symbols (list of str): The stock symbols to fetch and process data for.

            Returns:
                tuple: A tuple containing processed features, labels, corresponding dates, and raw datasets.
            """
            X_datasets, y_datasets, dates_list, processed_datasets = [], [], [], []
            for symbol in symbols:
                dataset = get_symbol_dataset(symbol)
                sub_X, sub_y, _, _, _, _, sub_dates, _, _ = \
                    self.process_data(dataset, train_split_ratio=self.params['train_split_ratio'], val_split_ratio=self.params['val_split_ratio'],
                                    target_col=self.params['target_col'],
                                    feature_cols=self.params['feature_cols'], look_back=self.params['look_back'],
                                    predict_steps=self.params['predict_steps'],
                                    train_slide_steps=self.params['train_slide_steps'],
                                    test_slide_steps=self.params['test_slide_steps'])
                X_datasets.append(sub_X)
                y_datasets.append(sub_y)
                dates_list.append(sub_dates)
                processed_datasets.append(dataset)
            return X_datasets, y_datasets, dates_list, processed_datasets

        # Fetch and process training datasets using the indices specified in the parameters
        train_indices = self.params['train_indices']
        test_indices = self.params['test_indices']
        X_train_datasets, y_train_datasets, train_dates, processed_datasets = process_datasets(train_indices)

        # Fetch and process the test dataset using the specified indices
        test_dataset = get_symbol_dataset(test_indices)
        _, _, X_val, y_val, X_test, y_test, _, test_dates, val_dates = \
            self.process_data(test_dataset, train_split_ratio=self.params['train_split_ratio'], val_split_ratio=self.params['val_split_ratio'], 
                            target_col=self.params['target_col'],
                            feature_cols=self.params['feature_cols'], look_back=self.params['look_back'],
                            predict_steps=self.params['predict_steps'],
                            train_slide_steps=self.params['train_slide_steps'],
                            test_slide_steps=self.params['test_slide_steps'])

        # Combine the training datasets from different indices into a single dataset
        num_samples = min(len(X) for X in X_train_datasets)
        X_train_combined = [X_train_datasets[idx][i] for i in range(num_samples) for idx in range(len(X_train_datasets))]
        y_train_combined = [y_train_datasets[idx][i] for i in range(num_samples) for idx in range(len(y_train_datasets))]
        X_train = torch.stack(X_train_combined, dim=0)
        y_train = torch.stack(y_train_combined, dim=0)

        # Apply any additional processing, such as reversing trend adjustments, to the datasets
        if self.params['filter_reverse_trend_train_test']:
            y_train = self.change_values_after_first_reverse_point(y_train)
            y_val = self.change_values_after_first_reverse_point(y_val)
            y_test = self.change_values_after_first_reverse_point(y_test)

        # Print the shapes of the datasets for verification
        print("Training set shape:", X_train.shape)
        print("Validation set shape:", X_val.shape)
        print("Test set shape:", X_test.shape)

        # Return the processed datasets along with their corresponding dates
        return X_train, y_train, X_val, y_val, X_test, y_test, train_dates, test_dates, val_dates, test_dataset


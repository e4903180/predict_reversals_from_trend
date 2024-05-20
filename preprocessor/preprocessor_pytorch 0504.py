import yfinance as yf
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import talib
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import from_numpy
import torch
from sklearn.model_selection import train_test_split
import tqdm
import exchange_calendars as xcals



class FeatureBase(ABC):
    """
    Abstract base class for all features.
    """
    @abstractmethod
    def compute(self, data=None, *args, **kwargs):
        """
        Abstract method to compute the feature value for the given data.
        """
        pass


class IndicatorTrend(FeatureBase):
    """
    Indicator to calculate the trend based on various methods.
    """

    def compute(self, data, *args, **kwargs):
        """
        Compute the trend for the given data using the specified method.
        """
        method = kwargs.get('method', 'MA')
        ma_days = kwargs.get('ma_days', 20)
        oder_days = kwargs.get('oder_days', 20)

        if method == 'MA':
            return self.calculate_trend_MA(data, ma_days=ma_days, oder_days=oder_days)
        elif method == 'LocalExtrema':
            return self.calculate_trend_LocalExtrema(data, oder_days=oder_days)
        else:
            raise ValueError(f"Invalid trend calculation method: {method}")

    def calculate_trend_MA(self, data, ma_days=20, oder_days=5):
        """
        Calculate trend using Moving Average method.
        TODO: 修改成最高點與最低點決定trend
        """
        # data['MA'] = data['Close'].rolling(window=ma_days).mean()
        data['MA'] = talib.MA(data['Close'], timeperiod=ma_days)
        # data['Trend'] = np.nan
        # n = len(data)

        # for i in range(n - trend_days + 1):
        #     if all(data['MA'].iloc[i + j] < data['MA'].iloc[i + j + 1] for j in range(trend_days - 1)):
        #         data['Trend'].iloc[i:i + trend_days] = 0 # up trend
        #     elif all(data['MA'].iloc[i + j] > data['MA'].iloc[i + j + 1] for j in range(trend_days - 1)):
        #         data['Trend'].iloc[i:i + trend_days] = 1 # down trend
        # data['Trend'].fillna(method='ffill', inplace=True)
        # return data.drop(columns=['MA'])
    
        local_max_indices = argrelextrema(
            data['MA'].values, np.greater_equal, order=oder_days)[0]
        local_min_indices = argrelextrema(
            data['MA'].values, np.less_equal, order=oder_days)[0]
        data['Local Max'] = data.iloc[local_max_indices]['Close']
        data['Local Min'] = data.iloc[local_min_indices]['Close']
        
        data['Trend'] = np.nan
        prev_idx = None
        prev_trend = None
        prev_type = None

        for idx in sorted(np.concatenate([local_max_indices, local_min_indices])):
            if idx in local_max_indices:
                current_type = "max"
            else:
                current_type = "min"
            if prev_trend is None:
                if current_type == "max":
                    prev_trend = 1
                else:
                    prev_trend = 0
            else:
                if prev_type == "max" and current_type == "min":
                    data.loc[prev_idx:idx, 'Trend'] = 1
                    prev_trend = 1 # down trend
                elif prev_type == "min" and current_type == "max":
                    data.loc[prev_idx:idx, 'Trend'] = 0
                    prev_trend = 0 # up trend
                else:
                    if current_type == "max":
                        data.loc[prev_idx:idx, 'Trend'] = 0
                        prev_trend = 0 # up trend
                    else:
                        data.loc[prev_idx:idx, 'Trend'] = 1
                        prev_trend = 1 # down trend
            prev_idx = idx
            prev_type = current_type
        data['Trend'].fillna(method='ffill', inplace=True)
        return data.drop(columns=['MA','Local Max', 'Local Min'])

    def calculate_trend_LocalExtrema(self, data, price_type='Close', oder_days=20):
        """
        Calculate trend using Local Extrema method.
        """
        if price_type == 'Close':
            local_max_indices = argrelextrema(
                data['Close'].values, np.greater_equal, order=oder_days)[0]
            local_min_indices = argrelextrema(
                data['Close'].values, np.less_equal, order=oder_days)[0]
            data['Local Max'] = data.iloc[local_max_indices]['Close']
            data['Local Min'] = data.iloc[local_min_indices]['Close']
        elif price_type == 'HighLow':
            local_max_indices = argrelextrema(
                data['High'].values, np.greater_equal, order=oder_days)[0]
            local_min_indices = argrelextrema(
                data['Low'].values, np.less_equal, order=oder_days)[0]
            data['Local Max'] = data.iloc[local_max_indices]['High']
            data['Local Min'] = data.iloc[local_min_indices]['Low']
        else:
            raise ValueError(f"Invalid price type: {price_type}")
        
        data['Trend'] = np.nan
        prev_idx = None
        prev_trend = None
        prev_type = None

        for idx in sorted(np.concatenate([local_max_indices, local_min_indices])):
            if idx in local_max_indices:
                current_type = "max"
            else:
                current_type = "min"
            if prev_trend is None:
                if current_type == "max":
                    prev_trend = 1
                else:
                    prev_trend = 0
            else:
                if prev_type == "max" and current_type == "min":
                    data.loc[prev_idx:idx, 'Trend'] = 1
                    prev_trend = 1 # down trend
                elif prev_type == "min" and current_type == "max":
                    data.loc[prev_idx:idx, 'Trend'] = 0
                    prev_trend = 0 # up trend
                else:
                    if current_type == "max":
                        data.loc[prev_idx:idx, 'Trend'] = 0
                        prev_trend = 0 # up trend
                    else:
                        data.loc[prev_idx:idx, 'Trend'] = 1
                        prev_trend = 1 # down trend
            prev_idx = idx
            prev_type = current_type
        data['Trend'].fillna(method='ffill', inplace=True)
        return data.drop(columns=['Local Max', 'Local Min'])


class IndicatorMACD(FeatureBase):
    """
    Indicator to calculate the Moving Average Convergence Divergence (MACD).
    """

    def compute(self, data, *args, **kwargs):
        fastperiod = kwargs.get('fastperiod', 5)
        slowperiod = kwargs.get('slowperiod', 10)
        signalperiod = kwargs.get('signalperiod', 9)
        data['MACD_dif'], data['MACD_dem'], data['MACD_histogram'] = talib.MACD(
            data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return data


class IndicatorROC(FeatureBase):
    def compute(self, data, *args, **kwargs):
        trend_days = kwargs.get('trend_days', 5)
        data['ROC'] = talib.ROC(data['Close'], timeperiod=trend_days)
        return data


class IndicatorStochasticOscillator(FeatureBase):
    def compute(self, data, *args, **kwargs):
        trend_days = kwargs.get('trend_days', 5)
        data['StoK'], data['StoD'] = talib.STOCH(
            data['High'], data['Low'], data['Close'], fastk_period=trend_days, slowk_period=3, slowd_period=3)
        return data


class IndicatorCCI(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 14)
        data['CCI'] = talib.CCI(data['High'], data['Low'],
                                data['Close'], timeperiod=timeperiod)
        return data


class IndicatorRSI(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 14)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=timeperiod)
        return data

class IndicatorVMA(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 20)
        data['VMA'] = talib.MA(data['Volume'], timeperiod=timeperiod)
        return data
    
class IndicatorMA(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 20)
        data['MA'] = talib.MA(data['Close'], timeperiod=timeperiod)
        return data


class IndicatorPctChange(FeatureBase):
    def compute(self, data, *args, **kwargs):
        data['pctChange'] = data['Close'].pct_change() * 100
        return data


class TreasuryYieldThirteenWeek(FeatureBase):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        thirteen_week_treasury_yield = yf.download(
            "^IRX", start_date, end_date)["Close"]
        data['13W Treasury Yield'] = thirteen_week_treasury_yield
        return data


class TreasuryYieldFiveYear(FeatureBase):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        five_year_treasury_yield = yf.download(
            "^FVX", start_date, end_date)["Close"]
        data['5Y Treasury Yield'] = five_year_treasury_yield
        return data


class TreasuryYieldTenYear(FeatureBase):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        ten_year_treasury_yield = yf.download(
            "^TNX", start_date, end_date)["Close"]
        data['10Y Treasury Yield'] = ten_year_treasury_yield
        return data


class TreasuryYieldThirtyYear(FeatureBase):
    def compute(self, data, *args, **kwargs):
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        thirty_year_treasury_yield = yf.download(
            "^TYX", start_date, end_date)["Close"]
        data['30Y Treasury Yield'] = thirty_year_treasury_yield
        return data
# Add other features here as needed

class IndicatorBollingerBands(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 20)
        nbdevup = kwargs.get('nbdevup', 2)
        nbdevdn = kwargs.get('nbdevdn', 2)
        data['upperband'], data['middleband'], data['lowerband'] = talib.BBANDS(
            data['Close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
        return data

class IndicatorATR(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 14)
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=timeperiod)
        return data

class IndicatorOBV(FeatureBase):
    def compute(self, data, *args, **kwargs):
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        return data

# Placeholder for Fibonacci Retracements since it's more of a visual tool and not directly calculable through a simple function.
# You would typically use this by selecting significant price points (peaks and troughs) and then calculating the Fibonacci levels manually or through a graphical interface.
    
class IndicatorParabolicSAR(FeatureBase):
    def compute(self, data, *args, **kwargs):
        start = kwargs.get('start', 0.02)
        increment = kwargs.get('increment', 0.02)
        maximum = kwargs.get('maximum', 0.2)
        data['Parabolic SAR'] = talib.SAR(data['High'], data['Low'], acceleration=start, maximum=maximum)
        return data

class IndicatorMOM(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 10)
        data['MOM'] = talib.MOM(data['Close'], timeperiod=timeperiod)
        return data

class IndicatorWilliamsR(FeatureBase):
    def compute(self, data, *args, **kwargs):
        lookback_period = kwargs.get('lookback_period', 14)
        data['Williams %R'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=lookback_period)
        return data

class IndicatorChaikinMF(FeatureBase):
    def compute(self, data, *args, **kwargs):
        timeperiod = kwargs.get('timeperiod', 20)
        data['Chaikin MF'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=timeperiod)
        return data
    
    
class FeatureFactory:
    """
    Factory class dedicated to creating various technical features.
    """
    @staticmethod
    def get_feature(feature_type):
        """
        Retrieve the desired feature based on the specified type.
        """
        features = {
            "Trend": IndicatorTrend,
            "MACD": IndicatorMACD,
            "ROC": IndicatorROC,
            "Stochastic Oscillator": IndicatorStochasticOscillator,
            "CCI": IndicatorCCI,
            "RSI": IndicatorRSI,
            "MA": IndicatorMA,
            "VMA": IndicatorVMA,
            "pctChange": IndicatorPctChange,
            "13W Treasury Yield": TreasuryYieldThirteenWeek,
            "5Y Treasury Yield": TreasuryYieldFiveYear,
            "10Y Treasury Yield": TreasuryYieldTenYear,
            "30Y Treasury Yield": TreasuryYieldThirtyYear,
            "Bollinger Bands": IndicatorBollingerBands,
            "ATR": IndicatorATR,
            "OBV": IndicatorOBV,
            "Parabolic SAR": IndicatorParabolicSAR,
            "MOM": IndicatorMOM,
            "Williams %R": IndicatorWilliamsR,
            "Chaikin MF": IndicatorChaikinMF,
            # Add other features here as needed
        }
        feature = features.get(feature_type)
        if feature is None:
            raise ValueError(f"Invalid feature type: {feature_type}")
        return feature()


class CleanerBase(ABC):
    """Abstract base class for data processors."""
    @abstractmethod
    def check(self, data):
        """Method to check the data for issues."""
        pass

    @abstractmethod
    def clean(self, data):
        """Method to clean the data from identified issues."""
        pass


class CleanerMissingValue(CleanerBase):
    """Concrete class for checking and handling missing data."""

    def check(self, data):
        """Check for missing data in the dataframe."""
        return data.isnull().sum()

    def clean(self, data, strategy='auto'):
        """Handle missing data based on the chosen strategy."""
        if strategy == 'auto':
            while data.iloc[0].isnull().any():
                data = data.iloc[1:]
            data.fillna(method='ffill', inplace=True)
        elif strategy == 'drop':
            data.dropna(inplace=True)
        elif strategy == 'fillna':
            data.fillna(method='ffill', inplace=True)
        elif strategy == 'none':
            pass
        else:
            raise ValueError("Invalid strategy provided.")
        return data


class ProcessorFactory:
    """Factory class to creat data processors."""
    @staticmethod
    def get_cleaner(clean_type, *args, **kwargs):
        """creat a data processor based on the provided type."""
        if clean_type == "MissingData":
            return CleanerMissingValue(*args, **kwargs)
        else:
            raise ValueError(f"Processor type {clean_type} not recognized.")

    @staticmethod
    def get_standardize_method(data, method='MinMaxScaler'):
        """Standardize the data using the specified method."""
        if method == 'StandardScaler':
            scaler = StandardScaler()
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaler method: {method}.")
        return scaler.fit_transform(data)

    @staticmethod
    def standardize_and_split_data(data, split_ratio=0.7, 
                                   target_col="Trend", feature_cols=None):
        """Standardize the data and split it into training and testing sets."""
        if not feature_cols:
            feature_cols = data.columns.to_list()
        # x_data 設為target_col以外的其他欄位
        # x_data = data.drop(columns=[target_col])
        x_data = data[feature_cols]
        y_data = pd.get_dummies(data[target_col], prefix='Trend')
        split_idx = int(len(x_data) * split_ratio)
        if split_idx < 1 or split_idx >= len(x_data):
            raise ValueError(
                "Invalid split ratio leading to incorrect data partitioning.")
        X_test = x_data.iloc[split_idx:]
        y_test = y_data.iloc[split_idx:]
        X_train = x_data.iloc[:split_idx]
        y_train = y_data.iloc[:split_idx]
        return X_train, y_train, X_test, y_test

    @staticmethod
    def prepare_multistep_data(x_data, y_data, look_back, predict_steps, slide_steps=1):
        """
        Prepare the data for multi-step prediction 
        and apply standardization within each sliding window.
        """
        x_date = []
        y_date = []
        x_data_multistep = []
        y_data_multistep = []

        for i in range(0, len(x_data) - look_back - predict_steps + 1, slide_steps):
            x_date.append(x_data.index[i:i + look_back])
            y_date.append(
                x_data.index[i + look_back:i + look_back + predict_steps])
            x_window = x_data.iloc[i:i + look_back].values
            y_window = y_data.iloc[i + look_back:i +
                                   look_back + predict_steps].values
            x_window_standardized = ProcessorFactory.get_standardize_method(
                x_window)
            x_data_multistep.append(x_window_standardized)
            y_data_multistep.append(y_window)
        return np.array(x_data_multistep), np.array(y_data_multistep), \
            np.array(x_date), np.array(y_date)

    @staticmethod
    def preprocess_for_prediction(x_data, look_back, slide_steps=1):
        """
        Prepare multiple instances of x_data for multi-step prediction.
        """
        x_date = []
        x_data_multistep = []

        for i in range(0, len(x_data) - look_back + 1, slide_steps):
            x_date.append(x_data.index[i:i + look_back])
            x_window = x_data.iloc[i:i + look_back].values
            x_window_standardized = ProcessorFactory.get_standardize_method(
                x_window)
            x_data_multistep.append(x_window_standardized)
        return np.array(x_data_multistep), np.array(x_date)


def generate_stock_price_data(dates, base_price=1000, amplitude=500, slope=0.1, frequency_multiplier=200):
    """
    生成模拟股票价格数据。

    :param dates: 日期序列，pd.DatetimeIndex对象。
    :param base_price: 基础价格，默认为1000。
    :param amplitude: 正弦波的振幅，默认为50。
    :param slope: 每天的价格变化斜率，默认为0.1。
    :param frequency_multiplier: 频率乘数，用于调整频率，默认为200。
    :return: 包含Open, High, Low, Close, Adj Close, Volume字段的DataFrame。
    """
    days = len(dates)  # 数据集包含的天数
    
    # 生成时间序列（0到days-1）
    time_series = np.arange(days)

    # 生成Close价格，加上斜率，调整频率
    close_prices = base_price + amplitude * np.sin(np.linspace(0, 2 * np.pi * frequency_multiplier, days)) + slope * time_series

    # 生成Open价格（除了第一天外，都是前一天的Close）
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]  # 第一天的Open设置为第一天的Close

    # 生成High和Low价格
    high_prices = np.maximum(open_prices, close_prices) + 1
    low_prices = np.minimum(open_prices, close_prices) - 1

    # 创建DataFrame
    dataset = pd.DataFrame({
        "Date": dates,
        "Open": open_prices,
        "High": high_prices,
        "Low": low_prices,
        "Adj Close": close_prices,
        "Volume": 1000,  # 假设每天的交易量为1000
        "Close": close_prices
    })

    # 设置日期为索引并返回
    dataset.set_index("Date", inplace=True)
    
    return dataset

class Preprocessor:
    """
    Fetching, processing, and preparing model data.
    """

    def __init__(self, params, start_date=None, end_date=None):
        self.params = params
        self.start_date = start_date
        self.end_date = end_date
        self.features = []
        self.processors = []

    # def fetch_stock_data(self, stock_symbol, start_date=None, end_date=None):
    #     """Fetch stock data from Yahoo Finance."""
    #     if start_date:
    #         self.start_date = start_date
    #     if end_date:
    #         self.end_date = end_date
    #     stock_symbol = '^GSPC'
    #     self.start_date = '2001-01-01'
    #     self.end_date = '2021-01-01'
    #     orginal_dataset =  yf.download(stock_symbol, start=self.start_date, end=self.end_date)
    #     return generate_stock_price_data(orginal_dataset.index)
    
    def fetch_stock_data(self, stock_symbol, start_date=None, end_date=None):
        """Fetch stock data from Yahoo Finance."""
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date
        return yf.download(stock_symbol, start=self.start_date, end=self.end_date)

    def add_feature(self, data, feature_type, *args, **kwargs):
        feature = FeatureFactory.get_feature(feature_type)
        return feature.compute(data, *args, **kwargs)

    def add_data_cleaner(self, data, clean_type='MissingData', strategy='auto'):
        """Method to check and clean the data using a specific processor."""
        processor = ProcessorFactory.get_cleaner(clean_type)
        issues = processor.check(data)
        data = processor.clean(data, strategy=strategy)
        return data, issues
    
    def reshape_data(self, data):
        return data.reshape(data.shape[0], data.shape[2], data.shape[1])
    
    def process_data(self, data, split_ratio=0.7, target_col="Trend",
                      feature_cols=None, look_back=64, predict_steps=16, 
                      train_slide_steps=1, test_slide_steps=16, reshape="False"):
        """
        Use ProcessorFactory to standardize and split the data, 
        and prepare it for multi-step prediction if required.
        """
        X_train, y_train, X_test, y_test = ProcessorFactory.standardize_and_split_data(
            data, split_ratio, target_col, feature_cols)

        if look_back and predict_steps:
            X_train, y_train, train_dates, _ = ProcessorFactory.prepare_multistep_data(
                X_train, y_train, look_back, predict_steps, train_slide_steps)
            X_test, y_test, _, test_dates = ProcessorFactory.prepare_multistep_data(
                X_test, y_test, look_back, predict_steps, test_slide_steps)
            if reshape == "True":
                X_train = self.reshape_data(X_train)
                X_test = self.reshape_data(X_test)
                y_train = self.reshape_data(y_train)
                y_test = self.reshape_data(y_test)
            X_train = from_numpy(X_train).float()
            y_train = from_numpy(y_train).float()
            X_test = from_numpy(X_test).float()
            y_test = from_numpy(y_test).float()
            return X_train, y_train, X_test, y_test, train_dates, test_dates
        else:
            raise ValueError(
                "Invalid look_back or predict_steps provided for data preparation.")
        
    def create_x_newest_data(self, data, look_back=64, feature_cols=None, reshape=False):
        """
        Create the newest X data for prediction using a specified number of the latest records.
        """

        if look_back > len(data):
            raise ValueError(
                "look_back exceeds the total number of available records.")
            
        if not feature_cols:
            feature_cols = data.columns.to_list()
        # x_data 設為target_col以外的其他欄位
        # x_data = data.drop(columns=[target_col])
        x_data = data[feature_cols]
        
        newest_data = x_data.tail(look_back)
        X_newest, x_date = ProcessorFactory.preprocess_for_prediction(
            newest_data, look_back)
        if reshape == "True":
            X_newest = self.reshape_data(X_newest)
        X_newest = from_numpy(X_newest).float()
        nyse = xcals.get_calendar("NYSE")
        y_date = nyse.sessions_window(x_date[0][-1], self.params['predict_steps']+1)[1:]
        return X_newest, x_date, np.array(y_date)
    
    def change_values_after_first_reverse_point(self, y:torch.Tensor):
        y_copy = y.clone().detach()
        modified_y = torch.zeros_like(y_copy)
        transition_log = torch.zeros(y_copy.shape[0])
        for idx, sub_y in enumerate(y_copy):
            array = sub_y.numpy()
            transition_found = False
            for i in range(1, len(array)):
                if not (array[i] == array[i-1]).all():
                    array[i:] = array[i]
                    transition_found = True
                    transition_log[idx] = 1
                    break
            if not transition_found:
                array = sub_y.numpy()
            
            modified_y[idx] = torch.tensor(array)
        return modified_y, transition_log
    
    def get_multiple_data(self):
        train_indices = self.params['train_indices']
        test_indice = self.params['test_indices']
        X_train_datasets = []
        y_train_datasets = []
        train_dates_list = []
        processed_datasets = []

        for symbol in train_indices:
            dataset = self.fetch_stock_data(symbol, self.params['start_date'], self.params['stop_date'])
            for single_feature_params in self.params['features_params']:
                feature_type = single_feature_params["type"]
                dataset = self.add_feature(dataset, feature_type, **single_feature_params)
            dataset, issues_detected = self.add_data_cleaner(dataset,
                clean_type=self.params['data_cleaning']['clean_type'], strategy=self.params['data_cleaning']['strategy'])
            sub_X_train, sub_y_train, _, _, sub_train_dates, _ = \
                self.process_data(dataset, split_ratio=self.params['split_ratio'], target_col=self.params['target_col'],
                                        feature_cols=self.params['feature_cols'], look_back=self.params['look_back'],
                                        predict_steps=self.params['predict_steps'],
                                        train_slide_steps=self.params['train_slide_steps'],
                                        test_slide_steps=self.params['test_slide_steps'],
                                        reshape=self.params['model_params'][self.params['model_type']]['reshape'])

            X_train_datasets.append(sub_X_train)
            y_train_datasets.append(sub_y_train)
            train_dates_list.append(sub_train_dates)
            processed_datasets.append(dataset)
            
        test_dataset = self.fetch_stock_data(test_indice, self.params['start_date'], self.params['stop_date'])
        for single_feature_params in self.params['features_params']:
            feature_type = single_feature_params["type"]
            test_dataset = self.add_feature(test_dataset, feature_type, **single_feature_params)
        test_dataset, issues_detected = self.add_data_cleaner(test_dataset,
            clean_type=self.params['data_cleaning']['clean_type'], strategy=self.params['data_cleaning']['strategy'])
        _, _, X_test, y_test, _, test_dates = \
            self.process_data(test_dataset, split_ratio=self.params['split_ratio'], target_col=self.params['target_col'],
                                    feature_cols=self.params['feature_cols'], look_back=self.params['look_back'],
                                    predict_steps=self.params['predict_steps'],
                                    train_slide_steps=self.params['train_slide_steps'],
                                    test_slide_steps=self.params['test_slide_steps'],
                                    reshape=self.params['model_params'][self.params['model_type']]['reshape'])
        X_newest, x_newest_date, y_date = self.create_x_newest_data(
            test_dataset, 
            look_back=self.params['look_back'], 
            feature_cols=self.params['feature_cols'], 
            reshape=self.params['model_params'][self.params['model_type']]['reshape'])
        
        num_samples = min(len(X) for X in X_train_datasets)
        X_train_combined, y_train_combined = [], []

        for i in range(num_samples):
            for idx in range(len(X_train_datasets)):
                X_train_combined.append(X_train_datasets[idx][i])
                y_train_combined.append(y_train_datasets[idx][i])

        X_train = torch.stack(X_train_combined, dim=0)
        y_train = torch.stack(y_train_combined, dim=0)
        # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Shuffle X_train and y_train with the same random index
        random_index = torch.randperm(X_train.shape[0])
        X_train = X_train[random_index]
        y_train = y_train[random_index]
        val_split_ratio = 0.2
        X_val = X_test[:int(val_split_ratio * X_test.shape[0]), :, :]
        y_val = y_test[:int(val_split_ratio * y_test.shape[0]), :, :]
        
        y_train_transition_log, y_val_transition_log, y_test_transition_log = None, None, None
        if self.params['filter_reverse_trend_train_test'] == "True":
            y_train, y_train_transition_log = self.change_values_after_first_reverse_point(y_train)
            y_val, y_val_transition_log = self.change_values_after_first_reverse_point(y_val)
            y_test, y_test_transition_log = self.change_values_after_first_reverse_point(y_test)

        print("Training set shape:", X_train.shape)
        print("Validation set shape:", X_val.shape)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, test_dates, X_newest, x_newest_date, y_date, test_dataset, y_train_transition_log, y_val_transition_log, y_test_transition_log

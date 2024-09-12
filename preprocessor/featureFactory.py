import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
import talib
from abc import ABC, abstractmethod


class FeatureBase(ABC):
    """
    Abstract base class for all features.
    """

    @abstractmethod
    def compute(self, data=None, *args, **kwargs):
        """
        Abstract method to compute the feature value for the given data.

        Args:
            data (pd.DataFrame, optional): The input data for which the feature needs to be computed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The data with computed feature(s) added.
        """
        pass


class IndicatorTrend(FeatureBase):
    """
    Indicator to calculate the trend based on various methods.
    """

    def compute(self, data, *args, **kwargs):
        """
        Compute the trend for the given data using the specified method.

        Args:
            data (pd.DataFrame): The input data containing price information.
            method (str, optional): The method to use for trend calculation. Defaults to 'LocalExtrema'.
            ma_days (int, optional): The number of days for moving average. Defaults to 20.
            oder_days (int, optional): The number of days used to identify local extrema. Defaults to 20.
            price_type (str, optional): The price type to use ('Close' or 'HighLow'). Defaults to 'Close'.

        Returns:
            pd.DataFrame: The input data with an additional 'Trend' column indicating the calculated trend.
        """
        method = kwargs.get('method', 'LocalExtrema')
        ma_days = kwargs.get('ma_days', 20)
        oder_days = kwargs.get('oder_days', 20)
        price_type = kwargs.get('price_type', 'Close')
        
        if method == 'LocalExtrema':
            return self.calculate_trend_LocalExtrema(data, oder_days=oder_days, price_type=price_type)
        else:
            raise ValueError(f"Invalid trend calculation method: {method}")

    def calculate_trend_LocalExtrema(self, data, price_type='Close', oder_days=20):
        """
        Calculate trend using Local Extrema method.

        Args:
            data (pd.DataFrame): The input data containing price information.
            price_type (str, optional): The price type to use ('Close' or 'HighLow'). Defaults to 'Close'.
            oder_days (int, optional): The number of days used to identify local extrema. Defaults to 20.

        Returns:
            pd.DataFrame: The input data with an additional 'Trend' column indicating the calculated trend.
        """
        def filter_extrema(local_max_indices, local_min_indices, data):
            filtered_max_indices = []
            filtered_min_indices = []
            i, j = 0, 0
            while i < len(local_max_indices) and j < len(local_min_indices):
                if local_max_indices[i] < local_min_indices[j]:
                    max_index = local_max_indices[i]
                    filtered_max_indices.append(max_index)
                    while i < len(local_max_indices) - 1 and local_max_indices[i + 1] < local_min_indices[j]:
                        if data['Close'][local_max_indices[i + 1]] > data['Close'][max_index]:
                            max_index = local_max_indices[i + 1]
                            filtered_max_indices[-1] = max_index
                        i += 1
                    i += 1
                else:
                    min_index = local_min_indices[j]
                    filtered_min_indices.append(min_index)
                    while j < len(local_min_indices) - 1 and local_min_indices[j + 1] < local_max_indices[i]:
                        if data['Close'][local_min_indices[j + 1]] < data['Close'][min_index]:
                            min_index = local_min_indices[j + 1]
                            filtered_min_indices[-1] = min_index
                        j += 1
                    j += 1

            while i < len(local_max_indices):
                filtered_max_indices.append(local_max_indices[i])
                i += 1
            while j < len(local_min_indices):
                filtered_min_indices.append(local_min_indices[j])
                j += 1

            return filtered_max_indices, filtered_min_indices

        if price_type == 'Close':
            local_max_indices = argrelextrema(data['Close'].values, np.greater_equal, order=oder_days)[0]
            local_min_indices = argrelextrema(data['Close'].values, np.less_equal, order=oder_days)[0]
        elif price_type == 'HighLow':
            local_max_indices = argrelextrema(data['High'].values, np.greater_equal, order=oder_days)[0]
            local_min_indices = argrelextrema(data['Low'].values, np.less_equal, order=oder_days)[0]

        filtered_max_indices, filtered_min_indices = filter_extrema(local_max_indices, local_min_indices, data)

        data['Local Max'] = np.nan
        data['Local Min'] = np.nan
        data['Local Max'].iloc[filtered_max_indices] = data['Close'].iloc[filtered_max_indices]
        data['Local Min'].iloc[filtered_min_indices] = data['Close'].iloc[filtered_min_indices]

        data['Trend'] = np.nan
        prev_idx = None
        prev_trend = None
        prev_type = None

        for idx in sorted(filtered_max_indices + filtered_min_indices):
            if idx in filtered_max_indices:
                current_type = "max"
            else:
                current_type = "min"
            if prev_trend is None:
                if current_type == "max":
                    prev_trend = 1  # down trend
                else:
                    prev_trend = 0  # up trend
            else:
                if prev_type == "max" and current_type == "min":
                    data.loc[prev_idx:idx, 'Trend'] = 1  # down trend
                    prev_trend = 1
                elif prev_type == "min" and current_type == "max":
                    data.loc[prev_idx:idx, 'Trend'] = 0  # up trend
                    prev_trend = 0

            prev_idx = idx
            prev_type = current_type

        data['Trend'].fillna(method='ffill', inplace=True)
        return data.drop(columns=['Local Max', 'Local Min'])


class IndicatorMACD(FeatureBase):
    """
    Indicator to calculate the Moving Average Convergence Divergence (MACD).
    """

    def compute(self, data, *args, **kwargs):
        """Compute MACD for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.
            fastperiod (int, optional): The short period for the MACD calculation. Defaults to 5.
            slowperiod (int, optional): The long period for the MACD calculation. Defaults to 10.
            signalperiod (int, optional): The signal period for the MACD calculation. Defaults to 9.

        Returns:
            pd.DataFrame: The input data with added columns for 'MACD_dif', 'MACD_dem', and 'MACD_histogram'.
        """
        fastperiod = kwargs.get('fastperiod', 5)
        slowperiod = kwargs.get('slowperiod', 10)
        signalperiod = kwargs.get('signalperiod', 9)
        data['MACD_dif'], data['MACD_dem'], data['MACD_histogram'] = talib.MACD(
            data['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return data


class IndicatorROC(FeatureBase):
    """Indicator for calculating the Rate of Change (ROC)."""

    def compute(self, data, *args, **kwargs):
        """Compute ROC for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.
            trend_days (int, optional): The period for the ROC calculation. Defaults to 5.

        Returns:
            pd.DataFrame: The input data with an added 'ROC' column.
        """
        trend_days = kwargs.get('trend_days', 5)
        data['ROC'] = talib.ROC(data['Close'], timeperiod=trend_days)
        return data


class IndicatorStochasticOscillator(FeatureBase):
    """Indicator for calculating the Stochastic Oscillator."""

    def compute(self, data, *args, **kwargs):
        """Compute Stochastic Oscillator for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', and 'Close' prices.
            trend_days (int, optional): The period for the fast %K calculation. Defaults to 5.

        Returns:
            pd.DataFrame: The input data with added columns for 'StoK' and 'StoD'.
        """
        trend_days = kwargs.get('trend_days', 5)
        data['StoK'], data['StoD'] = talib.STOCH(
            data['High'], data['Low'], data['Close'], fastk_period=trend_days, slowk_period=3, slowd_period=3)
        return data


class IndicatorCCI(FeatureBase):
    """Indicator for calculating the Commodity Channel Index (CCI)."""

    def compute(self, data, *args, **kwargs):
        """Compute CCI for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', and 'Close' prices.
            timeperiod (int, optional): The period for the CCI calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with an added 'CCI' column.
        """
        timeperiod = kwargs.get('timeperiod', 14)
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=timeperiod)
        return data


class IndicatorRSI(FeatureBase):
    """Indicator for calculating the Relative Strength Index (RSI)."""

    def compute(self, data, *args, **kwargs):
        """Compute RSI for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.
            timeperiod (int, optional): The period for the RSI calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with an added 'RSI' column.
        """
        timeperiod = kwargs.get('timeperiod', 14)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=timeperiod)
        return data


class IndicatorVMA(FeatureBase):
    """Indicator for calculating the Volume Moving Average (VMA)."""

    def compute(self, data, *args, **kwargs):
        """Compute VMA for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Volume'.
            timeperiod (int, optional): The period for the VMA calculation. Defaults to 20.

        Returns:
            pd.DataFrame: The input data with an added 'VMA' column.
        """
        timeperiod = kwargs.get('timeperiod', 20)
        data['VMA'] = talib.MA(data['Volume'], timeperiod=timeperiod)
        return data


class IndicatorMA(FeatureBase):
    """Indicator for calculating the Moving Average (MA)."""

    def compute(self, data, *args, **kwargs):
        """Compute MA for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.
            timeperiod (int, optional): The period for the MA calculation. Defaults to 20.

        Returns:
            pd.DataFrame: The input data with an added 'MA' column.
        """
        timeperiod = kwargs.get('timeperiod', 20)
        data['MA'] = talib.MA(data['Close'], timeperiod=timeperiod)
        return data


class IndicatorPctChange(FeatureBase):
    """Indicator for calculating the percentage change."""

    def compute(self, data, *args, **kwargs):
        """Compute percentage change for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.

        Returns:
            pd.DataFrame: The input data with an added 'pctChange' column representing the percentage change.
        """
        data['pctChange'] = data['Close'].pct_change() * 100
        return data


class TreasuryYieldThirteenWeek(FeatureBase):
    """Indicator for fetching the 13-week Treasury yield."""

    def compute(self, data, *args, **kwargs):
        """Fetch and add 13-week Treasury yield data to the given data.

        Args:
            data (pd.DataFrame): The input data to which the yield data will be added.
            start_date (str): The start date for fetching the yield data.
            end_date (str): The end date for fetching the yield data.

        Returns:
            pd.DataFrame: The input data with an added '13W Treasury Yield' column.
        """
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        thirteen_week_treasury_yield = yf.download(
            "^IRX", start_date, end_date)["Close"]
        data['13W Treasury Yield'] = thirteen_week_treasury_yield
        return data


class TreasuryYieldFiveYear(FeatureBase):
    """Indicator for fetching the 5-year Treasury yield."""

    def compute(self, data, *args, **kwargs):
        """Fetch and add 5-year Treasury yield data to the given data.

        Args:
            data (pd.DataFrame): The input data to which the yield data will be added.
            start_date (str): The start date for fetching the yield data.
            end_date (str): The end date for fetching the yield data.

        Returns:
            pd.DataFrame: The input data with an added '5Y Treasury Yield' column.
        """
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        five_year_treasury_yield = yf.download(
            "^FVX", start_date, end_date)["Close"]
        data['5Y Treasury Yield'] = five_year_treasury_yield
        return data


class TreasuryYieldTenYear(FeatureBase):
    """Indicator for fetching the 10-year Treasury yield."""

    def compute(self, data, *args, **kwargs):
        """Fetch and add 10-year Treasury yield data to the given data.

        Args:
            data (pd.DataFrame): The input data to which the yield data will be added.
            start_date (str): The start date for fetching the yield data.
            end_date (str): The end date for fetching the yield data.

        Returns:
            pd.DataFrame: The input data with an added '10Y Treasury Yield' column.
        """
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        ten_year_treasury_yield = yf.download(
            "^TNX", start_date, end_date)["Close"]
        data['10Y Treasury Yield'] = ten_year_treasury_yield
        return data


class TreasuryYieldThirtyYear(FeatureBase):
    """Indicator for fetching the 30-year Treasury yield."""

    def compute(self, data, *args, **kwargs):
        """Fetch and add 30-year Treasury yield data to the given data.

        Args:
            data (pd.DataFrame): The input data to which the yield data will be added.
            start_date (str): The start date for fetching the yield data.
            end_date (str): The end date for fetching the yield data.

        Returns:
            pd.DataFrame: The input data with an added '30Y Treasury Yield' column.
        """
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        thirty_year_treasury_yield = yf.download(
            "^TYX", start_date, end_date)["Close"]
        data['30Y Treasury Yield'] = thirty_year_treasury_yield
        return data


class VolatilityThreeMonth(FeatureBase):
    """Indicator for fetching the 3-month volatility index."""

    def compute(self, data, *args, **kwargs):
        """Fetch and add 3-month volatility index data to the given data.

        Args:
            data (pd.DataFrame): The input data to which the volatility index data will be added.
            start_date (str): The start date for fetching the volatility index data.
            end_date (str): The end date for fetching the volatility index data.

        Returns:
            pd.DataFrame: The input data with an added '3M Volatility' column.
        """
        start_date = kwargs.get('start_date')
        end_date = kwargs.get('end_date')
        three_month_volatility = yf.download(
            "^VIX3M", start_date, end_date)["Close"]
        data['3M Volatility'] = three_month_volatility
        return data


class IndicatorBollingerBands(FeatureBase):
    """Indicator for calculating Bollinger Bands."""

    def compute(self, data, *args, **kwargs):
        """Compute Bollinger Bands for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.
            timeperiod (int, optional): The time period for the moving average. Defaults to 20.
            nbdevup (int, optional): The number of standard deviations above the moving average for the upper band. Defaults to 2.
            nbdevdn (int, optional): The number of standard deviations below the moving average for the lower band. Defaults to 2.

        Returns:
            pd.DataFrame: The input data with added columns for 'upperband', 'middleband', and 'lowerband'.
        """
        timeperiod = kwargs.get('timeperiod', 20)
        nbdevup = kwargs.get('nbdevup', 2)
        nbdevdn = kwargs.get('nbdevdn', 2)
        data['upperband'], data['middleband'], data['lowerband'] = talib.BBANDS(
            data['Close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
        return data


class IndicatorATR(FeatureBase):
    """Indicator for calculating the Average True Range (ATR)."""

    def compute(self, data, *args, **kwargs):
        """Compute ATR for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', and 'Close' prices.
            timeperiod (int, optional): The time period for the ATR calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with an added 'ATR' column.
        """
        timeperiod = kwargs.get('timeperiod', 14)
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=timeperiod)
        return data


class IndicatorOBV(FeatureBase):
    """Indicator for calculating the On-Balance Volume (OBV)."""

    def compute(self, data, *args, **kwargs):
        """Compute OBV for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices and 'Volume'.

        Returns:
            pd.DataFrame: The input data with an added 'OBV' column.
        """
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        return data


class IndicatorParabolicSAR(FeatureBase):
    """Indicator for calculating the Parabolic Stop and Reverse (SAR)."""

    def compute(self, data, *args, **kwargs):
        """Compute Parabolic SAR for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High' and 'Low' prices.
            start (float, optional): The initial acceleration factor. Defaults to 0.02.
            increment (float, optional): The increment in acceleration factor. Defaults to 0.02.
            maximum (float, optional): The maximum acceleration factor. Defaults to 0.2.

        Returns:
            pd.DataFrame: The input data with an added 'Parabolic SAR' column.
        """
        start = kwargs.get('start', 0.02)
        increment = kwargs.get('increment', 0.02)
        maximum = kwargs.get('maximum', 0.2)
        data['Parabolic SAR'] = talib.SAR(data['High'], data['Low'], acceleration=start, maximum=maximum)
        return data


class IndicatorMOM(FeatureBase):
    """Indicator for calculating the Momentum (MOM)."""

    def compute(self, data, *args, **kwargs):
        """Compute Momentum for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'Close' prices.
            timeperiod (int, optional): The time period for the Momentum calculation. Defaults to 10.

        Returns:
            pd.DataFrame: The input data with an added 'MOM' column.
        """
        timeperiod = kwargs.get('timeperiod', 10)
        data['MOM'] = talib.MOM(data['Close'], timeperiod=timeperiod)
        return data


class IndicatorWilliamsR(FeatureBase):
    """Indicator for calculating the Williams %R."""

    def compute(self, data, *args, **kwargs):
        """Compute Williams %R for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', and 'Close' prices.
            lookback_period (int, optional): The lookback period for the Williams %R calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with an added 'Williams %R' column.
        """
        lookback_period = kwargs.get('lookback_period', 14)
        data['Williams %R'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=lookback_period)
        return data


class IndicatorChaikinMF(FeatureBase):
    """Indicator for calculating the Chaikin Money Flow (CMF)."""

    def compute(self, data, *args, **kwargs):
        """Compute Chaikin Money Flow for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', 'Close', and 'Volume'.
            timeperiod (int, optional): The time period for the Chaikin Money Flow calculation. Defaults to 20.

        Returns:
            pd.DataFrame: The input data with an added 'Chaikin MF' column.
        """
        timeperiod = kwargs.get('timeperiod', 20)
        data['Chaikin MF'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=timeperiod)
        return data
    
    
class IndicatorAroon(FeatureBase):
    """Indicator for calculating the Aroon Up and Down indicators."""

    def compute(self, data, *args, **kwargs):
        """Compute Aroon Up and Down indicators for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High' and 'Low' prices.
            timeperiod (int, optional): The time period for the Aroon calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with added 'Aroon Up' and 'Aroon Down' columns.
        """
        timeperiod = kwargs.get('timeperiod', 14)
        data['Aroon Up'], data['Aroon Down'] = talib.AROON(data['High'], data['Low'], timeperiod=timeperiod)
        return data
    
    
class IndicatorADL(FeatureBase):
    """Indicator for calculating the Accumulation/Distribution Line (ADL)."""

    def compute(self, data, *args, **kwargs):
        """Compute ADL for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', 'Close', and 'Volume'.

        Returns:
            pd.DataFrame: The input data with an added 'ADL' column.
        """
        data['ADL'] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        return data
    
    
class IndicatorADX(FeatureBase):
    """Indicator for calculating the Average Directional Index (ADX)."""

    def compute(self, data, *args, **kwargs):
        """Compute ADX for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', and 'Close' prices.
            timeperiod (int, optional): The time period for the ADX calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with an added 'ADX' column.
        """
        timeperiod = kwargs.get('timeperiod', 14)
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=timeperiod)
        return data
    
    
class IndicatorMFI(FeatureBase):
    """Indicator for calculating the Money Flow Index (MFI)."""

    def compute(self, data, *args, **kwargs):
        """Compute MFI for the given data.

        Args:
            data (pd.DataFrame): The input data containing 'High', 'Low', 'Close', and 'Volume'.
            timeperiod (int, optional): The time period for the MFI calculation. Defaults to 14.

        Returns:
            pd.DataFrame: The input data with an added 'MFI' column.
        """
        timeperiod = kwargs.get('timeperiod', 14)
        data['MFI'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=timeperiod)
        return data

# Add other features here as needed

    
class FeatureFactory:
    """
    Factory class dedicated to creating various technical features.

    This class provides a method to retrieve specific technical indicator features based on 
    the type provided by the user. It acts as a centralized registry of available features.
    """

    @staticmethod
    def get_feature(feature_type):
        """
        Retrieve the desired feature based on the specified type.

        Args:
            feature_type (str): The type of feature to create. This should correspond to one of the 
                                keys in the `features` dictionary.

        Returns:
            FeatureBase: An instance of the requested feature class.

        Raises:
            ValueError: If the provided `feature_type` is not recognized.
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
            "3M Volatility": VolatilityThreeMonth,
            "Bollinger Bands": IndicatorBollingerBands,
            "ATR": IndicatorATR,
            "OBV": IndicatorOBV,
            "Parabolic SAR": IndicatorParabolicSAR,
            "MOM": IndicatorMOM,
            "Williams %R": IndicatorWilliamsR,
            "Chaikin MF": IndicatorChaikinMF,
            "Aroon": IndicatorAroon,
            "ADL": IndicatorADL,
            "ADX": IndicatorADX,
            "MFI": IndicatorMFI
            # Add other features here as needed
        }

        feature = features.get(feature_type)
        if feature is None:
            raise ValueError(f"Invalid feature type: {feature_type}")
        return feature()

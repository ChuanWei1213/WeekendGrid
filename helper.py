import numpy as np
import pandas as pd
from typing import Literal
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import calendar

import cProfile
import pstats
import os
import subprocess
import types
    
def convert_interval(data: pd.DataFrame, t='1h'):
    """
    Convert the interval of price data frequency to t.
    
    Parameters:
        data (DataFrame): DataFrame with columns: time, open, high, low, close, volume.
        t (str): Target frequency (e.g., '1h', '1d').
        
    Returns:
        DataFrame: Resampled DataFrame with aggregated OHLCV data.
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Ensure 'time' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        data['time'] = pd.to_datetime(data['time'])
    
    # Set 'time' as index for resampling
    data.set_index('time', inplace=True)
    
    # Define aggregation rules for OHLCV data
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample the data to the target frequency
    resampled = data.resample(t).agg(agg_dict)
    
    # Remove intervals with no data
    resampled.dropna(subset=['open'], inplace=True)
    
    # Reset the index to bring 'time' back as a column
    resampled.reset_index(inplace=True)
    
    return resampled

def rsi(prices, period=14):
    """Calculate the Relative Strength Index (RSI) for a given price series.
    
    Parameters:
        prices (array-like): Sequence of price values.
        period (int): The period over which to calculate the RSI (default is 14).
        
    Returns:
        numpy.ndarray: Array of RSI values.
    """
    prices = np.asarray(prices)
    if len(prices) < period + 1:
        raise ValueError('Not enough data to compute RSI')
    
    # Calculate price differences
    delta = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Compute initial average gain and loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Prepare RSI array with NaN values for the first period
    rsi_values = np.empty(len(prices))
    rsi_values[:period] = np.nan
    
    # Compute the first RSI value at index 'period'
    if avg_loss == 0:
        rsi_values[period] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100 - (100 / (1 + rs))
    
    # Compute subsequent RSI values using Wilder's smoothing method
    for i in range(period + 1, len(prices)):
        current_gain = gains[i - 1]  # because delta index is offset by 1
        current_loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + current_gain) / period
        avg_loss = (avg_loss * (period - 1) + current_loss) / period
        if avg_loss == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    return rsi_values

def is_extreme(data, targets):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (targets < lower_bound) | (targets > upper_bound)

def show_stats(code_block: types.FunctionType, *args, **kwargs):
    """
    Run a block of code and show profiling stats.

    Parameters:
        code_block (FunctionType): A function or method containing the code block to be profiled.
        *args: Positional arguments to pass to the code block.
        **kwargs: Keyword arguments to pass to the code block.
    """
    snakeviz = kwargs.pop('snakeviz', True)
    print_stats = kwargs.pop('print_stats', True)
    
    with cProfile.Profile() as pr:
        res = code_block(*args, **kwargs)

    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    
    if print_stats:
        stats.print_stats(5)
    
    if snakeviz:
        stats.dump_stats('profile_results.prof')
        process = subprocess.Popen(['snakeviz', 'profile_results.prof'])
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
        os.remove('profile_results.prof')
        
    return res

import heapq

def top_k_heap(arr: np.ndarray, k: int):
    """Return k largest elements from numpy array using a min-heap (pure Python)."""
    if k <= 0:
        return np.array([], dtype=arr.dtype)
    n = arr.size
    if k >= n:
        return np.sort(arr)[::-1]
    heap = list(arr[:k])
    heapq.heapify(heap)
    for x in arr[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)
    return np.array(sorted(heap, reverse=True), dtype=arr.dtype)

def top_k_numpy(arr, k: int):
    """Return k largest elements from numpy array using argpartition + sort in C."""
    if k <= 0:
        return np.array([], dtype=arr.dtype)
    n = arr.size
    if k >= n:
        return np.sort(arr)[::-1]
    idx = np.argpartition(arr, -k)[-k:]
    topk = arr[idx]
    return np.sort(topk)[::-1]

def top_k(arr: np.ndarray, k: int):
    """
    Dispatch to the fastest top-k implementation based on array size.
    Uses pure-Python heap for n < crossover, and NumPy argpartition for n ≥ crossover.
    """
    arr = np.asarray(arr)
    arr = arr[~np.isnan(arr)]
    crossover = 200
    if arr.size < crossover:
        return top_k_heap(arr, k)
    else:
        return top_k_numpy(arr, k)
    
def next_weekday_start(current_time: datetime) -> datetime:
    """
    Returns the start time of the next weekday (06:00 Mon. UTC+8)
    """
    start = datetime(1970, 1, 4, 22)
    return (start + timedelta(weeks=int((current_time - start) / timedelta(weeks=1))+1))
    
def get_dst_start_end(year):
    """
    Returns the start and end dates of Daylight Saving Time (DST) for a given year.
    """
    # March = 3
    sundays = [day for day in range(1, 15)  # first two weeks
                if calendar.weekday(year, 3, day) == calendar.SUNDAY]
    dst_start = datetime(year, 3, sundays[1])  # second Sunday in March
    
    # November = 11
    sundays = [day for day in range(1, 8)  # first week
                if calendar.weekday(year, 11, day) == calendar.SUNDAY]
    dst_end = datetime(year, 11, sundays[0])  # first Sunday in November
    
    return dst_start, dst_end

def split_week(data):
    '''
    Weekend starts at 20:00 Fri (US stock close). and ends at 22:00 Sun (CME open).
    '''
    
    unique_years = data['time'].dt.year.unique()

    # Create DST lookup
    dst_lookup = {}
    for year in unique_years:
        dst_start, dst_end = get_dst_start_end(year)
        dst_lookup[year] = (dst_start, dst_end)
        
    def is_dst(date):
        date = pd.Timestamp(date)
        year = date.year
        dst_start, dst_end = dst_lookup[year]
        return dst_start <= date < dst_end
        
    start_time = data['time'].iloc[0]
    end_time = data['time'].iloc[-1]
    weekday_start = next_weekday_start(start_time)
    weekday_starts = np.arange(weekday_start, end_time, np.timedelta64(1, 'W'), dtype='datetime64[ns]')
    
    # Vectorized DST adjustment
    dst_adjustments = np.fromiter((np.timedelta64(not is_dst(date), 'h') for date in weekday_starts), dtype='timedelta64[h]')
    weekday_starts = weekday_starts + dst_adjustments
    weekend_starts = weekday_starts + np.timedelta64(118, 'h')

    # Use searchsorted to assign each timestamp to weekend/weekday periods
    weekday_indices = np.searchsorted(data['time'].values, weekday_starts)
    weekend_indices = np.searchsorted(data['time'].values, weekend_starts)

    boundaries = np.empty((weekday_indices.size + weekend_indices.size - 1,), dtype=weekday_indices.dtype)
    boundaries[0::2] = weekday_indices
    boundaries[1::2] = weekend_indices[:-1]

    weekdays = [data.iloc[boundaries[i]:boundaries[i+1]].reset_index(drop=True) for i in range(0, len(boundaries)-2, 2)]
    weekends = [data.iloc[boundaries[i]:boundaries[i+1]].reset_index(drop=True) for i in range(1, len(boundaries)-1, 2)]
        
    return weekdays, weekends
    

if __name__ == '__main__':
    data = pd.read_csv('data/klines/BTCUSDT_1m.csv', parse_dates=['time'])
    data = data[data['time'] > datetime(2025, 5, 25)]
    print(convert_interval(data, '1h').head())
import requests, asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from scipy.stats import linregress
from timeit import default_timer as timer
from tqdm import tqdm
from typing import List, Optional

import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

import warnings
from pathlib import Path
import platform
import subprocess
warnings.filterwarnings('ignore')

def interval_to_timedelta(interval: str) -> timedelta:
    t = int(interval[0:-1])
    unit_to_timedelta = {
        's': timedelta(seconds=t), 
        'm': timedelta(minutes=t), 
        'h': timedelta(hours=t), 
        'd': timedelta(days=t),
        'w': timedelta(weeks=t),
    }
    return unit_to_timedelta[interval[-1]]
            
            
async def fetch_chunk(session: aiohttp.ClientSession, url: str, exchange: str, params: dict, semaphore: asyncio.Semaphore):
    """
    Fetches a single chunk of data asynchronously with retries.
    """
    retries = 3
    delay = 2
    for attempt in range(retries):
        async with semaphore:
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if exchange == 'binance':
                            return [d[:6] for d in data]
                        elif exchange == 'coinbase':
                            return [[d['start'], d['open'], d['high'], d['low'], d['close'], d['volume']] for d in data['candles']]
                        
                    if resp.status == 429:
                        print(f'Error {resp.status}: {await resp.text()}')
                        raise aiohttp.ClientError("Rate limit exceeded")
                    # If not successful, prepare for retry
                    print(f"Attempt {attempt + 1}/{retries}: Error {resp.status}: {await resp.text()}. Retrying in {delay}s...")
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"Attempt {attempt + 1}/{retries}: Request failed ({type(e).__name__}) for params {params}. Retrying in {delay}s...")

        if attempt < retries - 1:
            await asyncio.sleep(delay)

    print(f"Failed to fetch chunk for params {params} after {retries} attempts.")
    return []

async def fetch_exchange_klines(
    exchange: str, 
    base_coin: str, 
    quote_coin: str,
    interval: int, 
    start: datetime | str, 
    end: Optional[datetime | str]=None
) -> pd.DataFrame:
    exchange = exchange.lower()
    supported_exchanges = ['binance', 'coinbase']
    if exchange not in supported_exchanges:
        raise ValueError(f"Invalid exchange. Choose from {supported_exchanges}")
    
    supported_intervals = {
        'binance': {'1s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'},
        'coinbase': {'1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '1d'}
    }
    if interval not in supported_intervals[exchange]:
        raise ValueError(f"Invalid interval. Choose from {supported_intervals[exchange]}")
    
    symbol = f'{base_coin}{quote_coin}'
    
    if exchange == 'binance':
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000
        }
        timestamp_mul = 1000
        unit = 'ms'
        start_name = 'startTime'
        end_name = 'endTime'
        
    elif exchange == 'coinbase':
        product_id = f'{base_coin}-{quote_coin}'
        url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
        params = {
            'granularity': {
                '1m': 'ONE_MINUTE', '5m': 'FIVE_MINUTE', '15m': 'FIFTEEN_MINUTE',
                '30m': 'THIRTY_MINUTE', '1h': 'ONE_HOUR', '2h': 'TWO_HOUR',
                '4h': 'FOUR_HOUR', '6h': 'SIX_HOUR', '1d': 'ONE_DAY'
            }[interval],
            'limit': 350
        }
        timestamp_mul = 1
        unit = 's'
        start_name = 'start'
        end_name = 'end'
        
    interval_timedelta = interval_to_timedelta(interval)
    interval_seconds = int(interval_timedelta.total_seconds())
    if end is None:
        end = datetime.now(tz=timezone.utc) - interval_timedelta
    delta = interval_timedelta * params['limit']
    
    time_chunks: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        chunk_end = min(current + delta, end)
        time_chunks.append((current, chunk_end))
        current = chunk_end

    all_data = []
    
    def data_to_df(data, unit):
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume"], dtype=float)
        df["time"] = pd.to_datetime(df["time"], unit=unit, utc=True)
        df = df.sort_values("time").drop_duplicates().reset_index(drop=True)
        return df
    
    try:
        semaphore = asyncio.Semaphore(4)
        timeout = aiohttp.ClientTimeout(total=30)  # 30-second timeout per request
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for start_chunk, end_chunk in time_chunks:
                params[start_name] = int(start_chunk.timestamp() * timestamp_mul)
                params[end_name] = int((end_chunk.timestamp() - interval_seconds) * timestamp_mul)
                tasks.append(fetch_chunk(session, url, exchange, params.copy(), semaphore))
            
            results = await tqdm_asyncio.gather(*tasks, desc="Fetching data")
            for result in results:
                all_data.extend(result)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"Fetched a total of {len(all_data)} records.")
    df = data_to_df(all_data, unit)

    return df
            
            
def update_klines(exchange: str, base_coin: str, quote_coin: str, interval: str):
    path = Path(f'data/klines/{exchange}/{base_coin}{quote_coin}_{interval}.csv')
    
    # Try to efficiently get last timestamp
    print('Reading last timestamp...')
    last_time = None
    
    try:
        # Try platform-specific fast method first (Unix-like systems)
        
        if platform.system() != 'Windows':
            result = subprocess.run(['tail', '-n', '1', path], capture_output=True, text=True, check=True)
            last_line = result.stdout.strip()
            
            # Get header to find time column position
            with open(path, 'r') as f:
                header = f.readline().strip().split(',')
            
            time_index = header.index('time')
            values = last_line.split(',')
            last_time_str = values[time_index]
            last_time = pd.to_datetime(last_time_str) + interval_to_timedelta(interval)
    except Exception as e:
        print(f"Fast method failed: {e}")
    
    # Fall back to pandas if needed
    if last_time is None:
        try:
            # Only read the time column from the last row
            df = pd.read_csv(path, usecols=['time'], parse_dates=['time'])
            last_time = df['time'].iloc[-1] + interval_to_timedelta(interval)
        except Exception as e:
            raise e("Failed to read last timestamp from file.")
    
    print(f'Fetching new data after {last_time}...')
    new_data = asyncio.run(fetch_exchange_klines(exchange, base_coin, quote_coin, interval, last_time))
    
    if new_data.empty:
        print('No new data to add')
        return
        
    # Append to file without reading entire contents
    print(f'Appending {len(new_data)} rows to file...')
    new_data.to_csv(path, mode='a', header=False, index=False)
    print('Done!')
    
def main():
    # Update the existing klines data to current time
    # update_klines('binance', 'BTC', 'USDT', '1m')

    # To fetch klines data:
    start_dt = datetime(2018, 1, 1, tzinfo=timezone.utc)
    end_dt = datetime.now(tz=timezone.utc)
    exchange = 'binance'
    base_coin = 'BTC'
    quote_coin = 'USDT'
    interval = '1m'
    symbol = f'{base_coin}{quote_coin}'
    df = asyncio.run(fetch_exchange_klines(exchange, base_coin, quote_coin, interval, start=start_dt, end=end_dt))
    file_path = Path(f'data/klines/{exchange}')
    file_path.mkdir(parents=True, exist_ok=True)
    file_name = f'{symbol}_{interval}.csv'
    df.to_csv(file_path / file_name, index=False)
    

if __name__ == '__main__':
    main()

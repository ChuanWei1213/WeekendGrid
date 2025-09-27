from __future__ import annotations

# Data Processing
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from itertools import product, repeat
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import defaultdict


# Data Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import seaborn as sns
import mplfinance as mpf
from IPython.display import display

# Typing
from typing import List, Tuple, Dict, Literal


# Helper Functions
from helper import top_k, convert_interval, is_extreme, split_week
from Metric import (
    regressed_annual_return, sharpe_ratio, r_sharpe_ratio, robust_risk_reward_ratio, 
    drawdowns, drawdown_periods, sortino_ratio, calmar_ratio, profit_factor, max_drawdown, longest_drawdown,
    longest_streak, analyze_returns
)

# Time Series Analysis
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Numba to boost performance
import numba
from numba import njit, float64, int64

# Caching
import shelve
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import threading

from functools import lru_cache


RISK_PER_TRADE = 0.1
TOTAL_INVESTMENT = int(1e4)
MODE = 'Arithmetic' # 'Arithmetic' or 'Geometric'
ROUNDING_FACTOR = 1e3
LIQUIDATION_MARGIN = 0.9
GRIDBOT_INITIAL_MARGIN = 0.73 # 0.68-0.73 by manual testing on Bybit ([0.726, 0.705, 0.710, 0.702, 0.689, 0.727])
MARIGIN_BUFFER = 0.0006
INITIAL_MARGIN = 0.01 + MARIGIN_BUFFER
MAINTENANCE_MARGIN = 0.005 + MARIGIN_BUFFER
STOPLOSS_MUL = 1

NGRIDS = np.arange(2, 22)


class GridStrategy:
    maker_fee: float = 2e-4
    taker_fee: float = 5.5e-4 
    
    def __init__(self, data: pd.DataFrame, 
                 low: float, high: float, ngrid: int, 
                 low_sl: float, high_sl: float, 
                 leverage: int = 1, mode: str = 'Arithmetic', 
                 total_investment: float = TOTAL_INVESTMENT,
                 debug: bool = False):
        self.data = data
        self.low = low
        self.high = high
        self.ngrid = ngrid
        self.low_sl = low_sl
        self.high_sl = high_sl
        self.mode = mode
        self.total_investment = total_investment
        self.debug = debug
        
        self.valid = True
        
        # Bybit sets the following condition as 'not enough profit per grid'
        mean = (high + low) / 2
        half_spread = high / mean - 1
        if ngrid > half_spread / 0.01 * 8:
        # spg = (high - low) / ngrid # spread per grid
        # if spg / high < 2.5e-3:
            self.valid = False
            if debug:
                raise ValueError('Not enough profit per grid')
            return
        
        if high_sl < high or low_sl > low:
            raise ValueError('Stoplosses must be outside the grid levels.'
                             f'Stoplosses: {(low_sl, high_sl)}, Low-High: {(low, high)}')
        
        self.initial_price = data.at[0, 'open']
        self.leverage = leverage
        
        self._init_limit_orders(low, high, ngrid, self.initial_price, mode)
        
    @property
    def leverage(self):
        return self._leverage

    @leverage.setter
    def leverage(self, val):
        if val <= 0:
            raise ValueError("Leverage must be a positive number.")
        self._leverage = val
        _, lower_grids, higher_grids, _ = self.get_grids(self.low, self.high, self.ngrid, self.initial_price, self.mode)
        self.qty = self.get_qty(self.total_investment, val, lower_grids, higher_grids)
        
    @staticmethod
    def get_qty(total_investment, leverage, lower_grids, higher_grids, flooring=True, method=None):
        larger_portion = lower_grids if len(lower_grids) > len(higher_grids) else higher_grids
        if method == 'default':
            qty = total_investment * leverage / sum(larger_portion) * GRIDBOT_INITIAL_MARGIN
        else:
            qty = total_investment * leverage / ((sum(lower_grids) + sum(higher_grids)) * (1 + INITIAL_MARGIN) + sum(larger_portion) * MAINTENANCE_MARGIN)
        if flooring:
            return math.floor(qty * ROUNDING_FACTOR) / ROUNDING_FACTOR
        return qty
        
    @staticmethod
    def get_grids(low, high, ngrid, initial_price, mode):
        if ngrid < 2 or ngrid > 200:
            raise ValueError('Number of grids must be between 2 and 200')

        ## Grid levels based on mode
        if mode == 'Arithmetic':
            grid_levels = np.linspace(low, high, ngrid + 1).round(1)
        elif mode == 'Geometric':
            grid_levels = np.geomspace(low, high, ngrid + 1).round(1)
        else:
            raise ValueError('Invalid mode. Mode must be Arithmetic or Geometric')
        
        ## Limit orders
        lower_grids = grid_levels[grid_levels < initial_price].tolist() # e.g. [1, 2, 3]
        higher_grids = grid_levels[grid_levels >= initial_price][::-1].tolist() # e.g. [6, 5, 4]
        
        ### Determine the closest grid and remove it from the corresponding grid list
        if (len(lower_grids) > 0 and 
            (len(higher_grids) == 0 or 
             initial_price - lower_grids[-1] < higher_grids[-1] - initial_price)
            ):
            waiting = lower_grids.pop()
        else:
            waiting = higher_grids.pop()
            
        return grid_levels, lower_grids, higher_grids, waiting
        
    def _init_limit_orders(self, low, high, ngrid, initial_price, mode):
        grid_levels, lower_grids, higher_grids, waiting = self.get_grids(low, high, ngrid, initial_price, mode)
            
        limit_orders = {'long': [], 'short': [], 'waiting': waiting}
            
        ### Initialize limit orders
        for stack, prices in zip([limit_orders['long'], limit_orders['short']], [lower_grids, higher_grids]):
            for price in prices:
                stack.append({'price': price, 'type': 'open'})
                
        self.grid_levels = grid_levels
        self.lower_grids = lower_grids
        self.higher_grids = higher_grids
        self.limit_orders = limit_orders
        
    
    def run(self, signal_plot=True) -> Dict[str, float | int]:
        """
        Executes the grid trading strategy on the provided market data.
        This function simulates a grid trading strategy by placing limit orders at predefined grid levels
        and managing positions based on market price movements. It calculates realized and unrealized 
        profit and loss (PnL), tracks trade history, and optionally visualizes the strategy's performance 
        on a candlestick chart.
        Args:
            plot (bool, optional): If True, generates a plot of the trading signals, grid levels, 
                                   and cumulative returns. Defaults to False.
        Returns:
            Dict[str, float | int]: A dictionary containing the following keys:
                - 'realized_pnl' (float): The total realized profit or loss from the strategy.
                - 'n_profit_trades' (int): The number of profitable trades executed.
                - 'volume' (float): The total trading volume during the strategy.
                - 'return' (float): The return on the initial capital as a percentage.
        Raises:
            ValueError: If the grid strategy is invalid and debugging is enabled.
        Notes:
            - The strategy ends if the price crosses the stop-loss levels.
            - All open positions are closed at the end of the strategy.
            - The function supports visualization of trading signals and performance metrics 
              using matplotlib and mplfinance.
        """
        
        if not self.valid and self.debug:
            raise ValueError('Invalid grid strategy. Please check the parameters.')
        
        data = self.data
        low_sl = self.low_sl
        high_sl = self.high_sl
        maker_fee = self.maker_fee
        taker_fee = self.taker_fee
        total_investment = self.total_investment
        
        limit_orders = self.limit_orders
        qty = self.qty

        N = len(data)
        cum_pnls = np.zeros(N)
        
        history_trades = []
        active_trade_ids = []
        
        trade_data = dict(
            realized_pnl = 0.0,
            n_profit_trades = 0,
            volume = 0.0
        )

        # Arrays to store the position markers
        position_markers = dict(
            enter_long = np.full(N, np.nan),
            enter_short = np.full(N, np.nan),
            exit_long = np.full(N, np.nan),
            exit_short = np.full(N, np.nan)
        )
        
        # Flags to indicate the presence of a signal
        is_signal = dict(long = False, short = False)
        
        # The percentage difference between the position of the marker relative and the current price on the plot
        shift = 7.5e-3
        
        def long():
            while limit_orders['long'] and L < limit_orders['long'][-1]['price']:
                order = limit_orders['long'].pop()
                if order['type'] == 'open':
                    # Store signal marker's position
                    position_markers['enter_long'][i] = L * (1 - shift)
                    
                    is_signal['long'] = True
                    
                    # Open long position
                    active_trade_ids.append(len(history_trades))
                    history_trades.append({'short': None, 'long': {'price': order['price'], 'time': time}})

                    # Update limit orders
                    limit_orders['short'].append({'price': limit_orders['waiting'], 'type': 'close'})
                    
                    # Update trade data
                    volume = order['price'] * qty
                    trade_data['volume'] += volume
                    trade_data['realized_pnl'] -= maker_fee * volume
                else:
                    position_markers['exit_short'][i] = L * (1 - shift)
                    
                    trade_id = active_trade_ids.pop()
                    
                    # Close short position
                    history_trades[trade_id]['long'] = {'price': order['price'], 'time': time}
                    
                    # Update limit orders
                    limit_orders['short'].append({'price': limit_orders['waiting'], 'type': 'open'})

                    # Update trade data
                    volume = order['price'] * qty
                    trade_data['volume'] += volume
                    entry_price = history_trades[trade_id]['short']['price']
                    trade_data['realized_pnl'] += (entry_price - order['price']) * qty - maker_fee * volume
                    trade_data['n_profit_trades'] += 1
                    
                limit_orders['waiting'] = order['price']
                
        def short():
            while limit_orders['short'] and H > limit_orders['short'][-1]['price']:
                order = limit_orders['short'].pop()
                if order['type'] == 'open':
                    # Store signal marker's position
                    position_markers['enter_short'][i] = H * (1 + shift)
                    
                    is_signal['short'] = True
                    
                    # Open short position
                    active_trade_ids.append(len(history_trades))
                    history_trades.append({'short': {'price': order['price'], 'time': time}, 'long': None})

                    # Update limit orders
                    limit_orders['long'].append({'price': limit_orders['waiting'], 'type': 'close'})
                    
                    # Update trade data
                    volume = order['price'] * qty
                    trade_data['volume'] += volume
                    trade_data['realized_pnl'] -= maker_fee * volume
                else:
                    position_markers['exit_long'][i] = H * (1 + shift)
                    
                    trade_id = active_trade_ids.pop()
                    
                    # Close long position
                    history_trades[trade_id]['short'] = {'price': order['price'], 'time': time}
                    
                    # Update limit orders
                    limit_orders['long'].append({'price': limit_orders['waiting'], 'type': 'open'})
                    
                    # Update trade data
                    volume = order['price'] * qty
                    trade_data['volume'] += volume
                    entry_price = history_trades[trade_id]['long']['price']
                    trade_data['realized_pnl'] += (order['price'] - entry_price) * qty - maker_fee * volume
                    trade_data['n_profit_trades'] += 1
                    
                limit_orders['waiting'] = order['price']
                
        def close_all(price):
            for trade_id in active_trade_ids:
                volume = price * qty
                trade_data['volume'] += volume
                
                if history_trades[trade_id]['long']:
                    position_markers['exit_long'][i] = H * (1 + shift)
                    
                    entry_price = history_trades[trade_id]['long']['price']
                    trade_data['realized_pnl'] += (price - entry_price) * qty - taker_fee * volume
                else:
                    position_markers['exit_short'][i] = L * (1 - shift)
                    
                    entry_price = history_trades[trade_id]['short']['price']
                    trade_data['realized_pnl'] += (entry_price - price) * qty - taker_fee * volume
                    
            active_trade_ids.clear()

        for i, (time, O, H, L, C) in zip(data.index, data.values[:, :5]):
            # Due to the fact that a kline may intersect with multiple grid levels, 
            # if open > close, consider high appears before low, 
            # therefore we check short first,
            # vice versa.
            if O > C:
                short(); long()
            else:
                long(); short()

            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            for trade_id in active_trade_ids:
                if history_trades[trade_id]['long']:
                    unrealized_pnl += (C - history_trades[trade_id]['long']['price']) * qty
                else:
                    unrealized_pnl += (history_trades[trade_id]['short']['price'] - C) * qty

            # Update total PnL (realized + unrealized)
            pnl = trade_data['realized_pnl'] + unrealized_pnl
            cum_pnls[i] = pnl
            
            # End strategy if price crosses the stop loss levels
            if L < low_sl or H > high_sl:
                close_all(low_sl if L < low_sl else high_sl)
                break

        # Close all positions after ending the strategy
        close_all(C)
        
        trade_data['return'] = round(trade_data['realized_pnl'] / total_investment, 8)
        if not signal_plot:
            return trade_data
        
        pnl = trade_data['realized_pnl']
        cum_pnls[i:] = pnl
        cum_returns = cum_pnls / total_investment

        # Plot strategy
        apds = [mpf.make_addplot(np.full(N, grid_level), color='green', linestyle='--') for grid_level in self.grid_levels]

        apds.append(mpf.make_addplot(np.full(N, high_sl), color='red', linestyle='-'))
        apds.append(mpf.make_addplot(np.full(N, low_sl), color='red', linestyle='-'))

        apds.append(mpf.make_addplot(cum_returns, panel=1, ylabel='Return'))

        if is_signal['long']:
            apds.extend([
                mpf.make_addplot(position_markers['enter_long'], type='scatter', markersize=100, marker='^', color='green', panel=0),
                mpf.make_addplot(position_markers['enter_long']*(1-shift), type='scatter', markersize=2200, marker='$Enter\\ long$', color='green', panel=0),
                mpf.make_addplot(position_markers['exit_long'], type='scatter', markersize=100, marker='v', color='red', panel=0),
                mpf.make_addplot(position_markers['exit_long']*(1+shift), type='scatter', markersize=2000, marker='$Exit\\ long$', color='red', panel=0)
            ])
        if is_signal['short']:
            apds.extend([
                mpf.make_addplot(position_markers['enter_short'], type='scatter', markersize=100, marker='v', color='red', panel=0),
                mpf.make_addplot(position_markers['enter_short']*(1+shift), type='scatter', markersize=2600, marker='$Enter\\ short$', color='red', panel=0),
                mpf.make_addplot(position_markers['exit_short'], type='scatter', markersize=100, marker='^', color='green', panel=0),
                mpf.make_addplot(position_markers['exit_short']*(1-shift), type='scatter', markersize=2200, marker='$Exit\\ short$', color='green', panel=0)
            ])

        mpstyle = mpf.make_mpf_style(base_mpf_style='binance', rc={'font.family': 'DejaVu Sans'})

        tmp = data.set_index('time')
        fig, axlist = mpf.plot(tmp, type='candle', style=mpstyle, addplot=apds, panel_ratios=(2, 1), figsize=(10, 6), returnfig=True,
                                xlabel=f'Time ({tmp.index[0].year})', ylabel='Price ($)')
        fig.suptitle('Chart with Trading Signals and Grid Levels', y=1.1)
        
        # Add a new axes for the table above the main plot area.
        # [left, bottom, width, height] in figure coordinates (0-1).
        table_ax = fig.add_axes([0.36, 0.9, 0.3, 0.075])
        table_ax.axis('off')  # Hide the axes

        # Create the table.
        table = table_ax.table(cellText=[
                                    ('# of Profit Trades', trade_data['n_profit_trades']),
                                    ('Return', f'{cum_returns[-1]:.4f}'),
                                ],
                                bbox=[0, 0, 1, 2],
                                colWidths=[0.6, 0.4],
                                edges='closed')

        mpf.show()

        return trade_data
    
    def backtest(self) -> float:
        """
        `backtest` is a runtime optimized version of `run` method that only calculates the final return.
        """
        if not self.valid:
            raise ValueError('Invalid grid strategy. Please check the parameters.')
        
        arr = self.data[['open','high','low','close']].values.astype(np.float64)
        ngrid = np.int64(self.ngrid)
        low_sl = np.float64(self.low_sl)
        high_sl = np.float64(self.high_sl)
        maker_fee = np.float64(self.maker_fee)
        taker_fee = np.float64(self.taker_fee)
        grid_levels = self.grid_levels.astype(np.float64)
        lp = len(self.lower_grids) - 1
        
        return round(self._numba_backtest(arr, ngrid, low_sl, high_sl, maker_fee, taker_fee, grid_levels, lp) * self.qty / self.total_investment, 8)
    
    @staticmethod
    @njit(float64(float64[:, :], int64, float64, float64, float64, float64, float64[:], int64), cache=True, nogil=True)
    def _numba_backtest(arr, ngrid, low_sl, high_sl, maker_fee, taker_fee, grid_levels, lp):
        """
        Numba optimized backtest function for the grid strategy.
        """
        mp = lp + 1
        hp = mp + 1
        
        pnl = 0.0
        position = 0
        
        for i in range(arr.shape[0]):
            O, H, L, C = arr[i]
            # SHORT logic
            if O > C:
                while hp <= ngrid and H > grid_levels[hp]:
                    price = grid_levels[hp]
                    pnl -= price * maker_fee
                    if position > 0:
                        pnl += (price - grid_levels[mp])
                    position -= 1
                    lp += 1; mp += 1; hp += 1
                # LONG logic in same iteration
                while lp >= 0 and L < grid_levels[lp]:
                    price = grid_levels[lp]
                    pnl -= price * maker_fee
                    if position < 0:
                        pnl += (grid_levels[mp] - price)
                    position += 1
                    lp -= 1; mp -= 1; hp -= 1
            else:
                # LONG logic first
                while lp >= 0 and L < grid_levels[lp]:
                    price = grid_levels[lp]
                    pnl -= price * maker_fee
                    if position < 0:
                        pnl += (grid_levels[mp] - price)
                    position += 1
                    lp -= 1; mp -= 1; hp -= 1
                # SHORT logic
                while hp <= ngrid and H > grid_levels[hp]:
                    price = grid_levels[hp]
                    pnl -= price * maker_fee
                    if position > 0:
                        pnl += (price - grid_levels[mp])
                    position -= 1
                    lp += 1; mp += 1; hp += 1
            # STOPLOSS check
            if L < low_sl or H > high_sl:
                # CLOSE ALL logic
                price = low_sl if L < low_sl else high_sl
                pnl -= price * taker_fee * np.abs(position)
                while position > 0:
                    pnl += (price - grid_levels[mp])
                    mp += 1
                    position -= 1
                while position < 0:
                    pnl += (grid_levels[mp] - price)
                    mp -= 1
                    position += 1
                break
        # final CLOSE ALL at last close price
        price = C
        pnl -= price * taker_fee * np.abs(position)
        while position > 0:
            pnl += (price - grid_levels[mp])
            mp += 1
            position -= 1
        while position < 0:
            pnl += (grid_levels[mp] - price)
            mp -= 1
            position += 1
        return pnl
        

# Helper for parallel optimization
def _run_strategies_wrap(
    obj: WeekendGrid | WeekendGridV1, params, kwargs
):
    return obj._run_strategies(*params, **kwargs)

class WeekendGrid:
    metrics = {
        'Regressed Annual Return': regressed_annual_return, 
        'Sharpe Ratio': sharpe_ratio, 
        'R-Sharpe Ratio': r_sharpe_ratio, 
        'Robust Risk-Reward Ratio': robust_risk_reward_ratio
    }
    
    def __init__(
        self, 
        ticker = 'BTCUSDT', 
        interval: Literal['1m', '2m', '5m', '15m'] = '1m', 
    ):
        self.ticker = ticker
        self.interval = interval
        
        data_path = Path('/Users/chuanwei/Files/Projects/Quant_Trading/data')
        self.data_path = data_path
        self._data = pd.read_csv(data_path / Path(f'klines/binance/{ticker}_{interval}.csv'), parse_dates=['time'])
        self._data['time'] = pd.to_datetime(self._data['time'], utc=True).dt.tz_localize(None)
        self.data = self._data

        self.cache_file = data_path / Path('db/grid_cache.db')
        self._cache_lock = threading.Lock()
        
        np.random.seed(42)
        
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value.reset_index(drop=True)
        self.weekdays, self.weekends = split_week(self._data)
        self.times = np.array([weekend.at[0, 'time'] for weekend in self.weekends], dtype='datetime64[ns]')
        
    @property
    def montecarlo_weekdays(self):
        if not hasattr(self, '_montecarlo_weekdays'):
            self._montecarlo_weekdays, self._montecarlo_weekends = self._generate_montecarlo_week(len(self.weekends))
        return self._montecarlo_weekdays
    
    @property
    def montecarlo_weekends(self):
        if not hasattr(self, '_montecarlo_weekends'):
            self._montecarlo_weekdays, self._montecarlo_weekends = self._generate_montecarlo_week(len(self.weekends))
        return self._montecarlo_weekends


    def _get_default_ngrids(self):
        return NGRIDS.copy()
    
    
    def _get_default_thresholds(self):
        return np.arange(10, 31) / 1000
    
    
    
    @staticmethod
    def _adjust_leverage(initial_price, qty, 
                         lower_grids, higher_grids, 
                         low_sl, high_sl,
                         risk_per_trade=RISK_PER_TRADE, 
                         total_investment=TOTAL_INVESTMENT):
        def calculate_liquidation_price(grids, stoploss, direction):
            avg_price, position, fee = initial_price, 0, 0
            liquidation_price = np.inf * (-direction)
            for i, price in enumerate(grids[::-1] + [liquidation_price]):
                pnl = (price - avg_price) * position
                if pnl - fee < -total_investment * LIQUIDATION_MARGIN:
                    liquidation_price = -((-total_investment * LIQUIDATION_MARGIN + fee) / position + avg_price)
                    break
                position += qty * direction
                fee += maker_fee * price * qty
                avg_price = (avg_price * i + price) / (i + 1)
            fee += taker_fee * stoploss * abs(position)
            max_loss = -((stoploss - avg_price) * position - fee) / total_investment
            return max_loss

        maker_fee, taker_fee = GridStrategy.maker_fee, GridStrategy.taker_fee
        max_loss_long = calculate_liquidation_price(lower_grids, low_sl, direction=1)
        max_loss_short = calculate_liquidation_price(higher_grids, high_sl, direction=-1)
        max_loss = max(max_loss_long, max_loss_short)

        adjusted_leverage = risk_per_trade / max_loss
        if adjusted_leverage >= 1:
            return math.floor(adjusted_leverage)
        return math.floor(adjusted_leverage * ROUNDING_FACTOR) / ROUNDING_FACTOR
    
    @staticmethod
    def _grid_low_high_sl(initial_price, ngrid, threshold, sl_mul=STOPLOSS_MUL):
        low, high = initial_price * (1 - threshold), initial_price * (1 + threshold)
        space = (high - low) / ngrid
        low_sl, high_sl = low - space * sl_mul, high + space * sl_mul
        
        low = round(low, 1)
        high = round(high, 1)
        low_sl = round(low_sl, 1)
        high_sl = round(high_sl, 1)
        
        return low, high, low_sl, high_sl
    
    def _get_cache_key(self, weekend_idx, ngrid, threshold, mode):
        """Generate a unique cache key for the given parameters."""
        key_data = f"{weekend_idx}_{ngrid}_{threshold}_{mode}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # Python Shelve
    @contextmanager
    def _get_shelve_cache(self, flag='c'):
        """Context manager for shelve operations."""
        with self._cache_lock:
            cache = shelve.open(str(self.cache_file), flag=flag)
            try:
                yield cache
            finally:
                cache.close()
    
    def _shelve_get(self, cache_key):
        """Get value from shelve cache."""
        try:
            with self._get_shelve_cache('r') as cache:
                return cache.get(cache_key)
        except:
            return None
    
    def _shelve_set(self, cache_key, result):
        """Set value in shelve cache."""
        try:
            with self._get_shelve_cache('c') as cache:
                cache[cache_key] = result
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def grid_strategy(self, weekend_idx, ngrid, threshold, risk_per_trade=RISK_PER_TRADE, mode=MODE, 
                      total_investment=TOTAL_INVESTMENT, signal_plot=False, debug=False, use_cache=False, montecarlo=False):
        
        threshold = round(threshold, 8)
        
        # Generate cache key and try to load from cache
        if use_cache and not signal_plot and not debug and not montecarlo:
            cache_key = self._get_cache_key(weekend_idx, ngrid, threshold, mode)
            
            cached_result = self._shelve_get(cache_key)
            if cached_result is not None:
                return cached_result
        
        if not montecarlo:
            weekend = self.weekends[weekend_idx]
        else:
            weekend = self.montecarlo_weekends[weekend_idx]
        initial_price = weekend.at[0, 'open']
        low, high, low_sl, high_sl = self._grid_low_high_sl(initial_price, ngrid, threshold)
        _, lower_grids, higher_grids, _ = GridStrategy.get_grids(low, high, ngrid, initial_price, mode)
        qty = GridStrategy.get_qty(total_investment, 1, lower_grids, higher_grids, flooring=False)
        leverage = self._adjust_leverage(initial_price, qty,
                                         lower_grids, higher_grids,
                                         low_sl, high_sl, 
                                         risk_per_trade,
                                         total_investment)
        gs = GridStrategy(weekend, low, high, ngrid, low_sl, high_sl, 
                          leverage=leverage, mode=mode, total_investment=total_investment, debug=debug)
        
        if not gs.valid:
            result = 0.0
        elif debug:
            run_return = gs.run(False)['return']
            backtest_return = gs.backtest()
            assert np.allclose(run_return, backtest_return), "Backtest and run results do not match"
            assert backtest_return > -risk_per_trade, f"Loss ({backtest_return}) exceeds risk per trade ({-risk_per_trade})."
            result = backtest_return
        elif signal_plot:
            result = gs.run(signal_plot)['return']
        else:
            result = gs.backtest()
        
        # Save to cache
        if use_cache and not signal_plot and not debug and not montecarlo:
            self._shelve_set(cache_key, result)
        
        return result
    
    def get_cache_stats(self):
        """Get cache statistics."""
        try:
            with self._get_shelve_cache('r') as cache:
                count = len(cache)
        except:
            count = 0
        size = self.cache_file.stat().st_size if self.cache_file.exists() else 0
            
        return {
            'entries': count,
            'size_mb': size / (1024 * 1024),
        }

    def _run_strategies(self, ngrid, threshold, risk_per_trade, mode, total_investment,
                        signal_plot, debug, use_cache, montecarlo, start, end, multithread):
        
        def _task(i):
            return self.grid_strategy(
                i,
                ngrid,
                threshold,
                risk_per_trade=risk_per_trade,
                mode=mode,
                total_investment=total_investment,
                signal_plot=signal_plot,
                debug=debug,
                use_cache=use_cache,
                montecarlo=montecarlo
            )

        start = self._to_index(0 if start is None else start)
        end = self._to_index(len(self.weekends) if end is None else end)
        
        if multithread:
            with ThreadPoolExecutor() as executor:
                returns = list(executor.map(_task, range(start, end)))
        else:
            returns = [_task(i) for i in range(start, end)]

        return pd.Series(returns, index=range(start, end))
    
    def _to_index(self, val):
        # Support various types of time data for start/end (datetime, np.datetime64, str, Timestamp, etc.)
        if isinstance(val, (datetime, np.datetime64, pd.Timestamp, str)):
            return np.searchsorted(self.times, np.datetime64(val))
        elif isinstance(val, (int, np.integer, pd.Int64Dtype)) or val is None:
            return val
        raise ValueError(f"Invalid type for start/end: {type(val)}")
            
    @property
    def params(self):
        if hasattr(self, '_params'):
            return self._params
        
        self._params = {
            'ngrids': None,
            'thresholds': None,
            'risk_per_trade': None,
            'mode': None,
        }
        return self._params
            
    def _get_results(self, ngrids, thresholds, debug, use_cache, montecarlo, multithread, multiprocess, 
                    verbose, **kwargs):
        params = {'ngrids': ngrids, 'thresholds': thresholds}
        # Store additional parameters in params if they are defined
        params.update({k: v for k, v in kwargs.items() if k in self.params})
        
        # Check if the parameters are the same as the cached ones
        same_params = True
        for k in params:
            if self.params[k] is None:
                same_params = False
                break
            elif isinstance(params[k], (list, range, np.ndarray)):
                if not np.array_equal(params[k], self.params[k]):
                    same_params = False
                    break
            elif params[k] != self.params[k]:
                same_params = False
                break
            
        if same_params and not montecarlo:
            # Return cached results if parameters are the same
            return self.results
        else:
            kwargs.update(
                **{k: v for k, v in params.items() if k not in {'ngrids', 'thresholds'}},
                signal_plot=False,
                debug=debug,
                use_cache=use_cache,
                montecarlo=montecarlo,
                start=None,
                end=None,
                multithread=multithread,
            )

            pairs = list(product(ngrids, thresholds))
            items = (_run_strategies_wrap, repeat(self), pairs, repeat(kwargs))
            if multiprocess:
                with ProcessPoolExecutor() as executor:
                    iterator = executor.map(*items)
                    if verbose > 0:
                        desc = "Getting Results (multiprocessing, multithreaded)" if multithread \
                            else "Getting Results (multiprocessing)"
                        iterator = tqdm(iterator, total=len(pairs), desc=desc)
            else:
                iterator = map(*items)
                if verbose > 0:
                    desc = f"Getting Results (multithreaded)" if multithread else "Getting Results"
                    iterator = tqdm(iterator, total=len(pairs), desc=desc)

            data = list(iterator)
            
            index = data[0].index
            columns = pd.MultiIndex.from_product((ngrids, thresholds), names=('ngrid', 'threshold'))
            data = np.array([series.values for series in data]).T
            results = pd.DataFrame(data=data, index=index, columns=columns)

            self.params.update(params)
            self.results = results
            return results
    
    def rolling_optimize(self, optimize_window=104, update_period=13, ngrids=None, thresholds=None, 
                                metric='R-Sharpe Ratio', rank=1, 
                                risk_per_trade=RISK_PER_TRADE, mode=MODE,
                                total_investment=TOTAL_INVESTMENT, debug=False, use_cache=False, montecarlo=False,
                                start=None, end=None, multithread=False, multiprocess=False,
                                result_plot=True, signal_plot=False, verbose=1, **kwargs):
        if ngrids is None:
            ngrids = self._get_default_ngrids()
        if thresholds is None:
            thresholds = self._get_default_thresholds()
            
        results = self._get_results(
            ngrids, 
            thresholds, 
            risk_per_trade=risk_per_trade, 
            mode=mode, 
            total_investment=total_investment, 
            debug=debug, 
            use_cache=use_cache,
            montecarlo=montecarlo,
            multithread=multithread,
            multiprocess=multiprocess,
            verbose=verbose,
            **kwargs
        )
        
        start = max(self._to_index(0 if start is None else start), optimize_window)
        end = self._to_index(len(self.weekends) if end is None else end)

        returns = pd.Series(index=range(start, end), dtype=float)
        metric_fn = self.metrics[metric]

        iterator = range(start - optimize_window, end - optimize_window, update_period)
        if verbose > 0 and not signal_plot:
            desc = 'Collecting Results'
            iterator = tqdm(iterator, total=len(iterator), desc=desc)
            
        pr_values = pd.Series(index=range(start, end), dtype=float)
        median_returns = pd.Series(index=range(start, end), dtype=float)
        # Store returns for all ranks
        ranks = list(range(1, results.shape[1]+1))
        all_rank_returns = pd.DataFrame(index=range(start, end), columns=ranks, dtype=float)
        all_params = pd.Series(index=range(start, end), dtype=object)
        
        for train_start in iterator:
            train_end = train_start + optimize_window - 1
            test_start = train_end + 1
            test_end = min(test_start + update_period - 1, end - 1)
            train_data = results.loc[train_start:train_end]
            test_data = results.loc[test_start:test_end]

            train_results = train_data.apply(lambda col: metric_fn(col.values, 52))
            sorted_params = train_results.sort_values(ascending=False).index
            
            # Store returns for all ranks
            for r, params in enumerate(sorted_params, 1):
                all_rank_returns.loc[test_start:test_end, r] = test_data.loc[test_start:test_end, params]
            
            # Use specified rank for main strategy
            params = sorted_params[rank - 1]
            all_params.loc[test_start:test_end] = [params for _ in range(test_end - test_start + 1)]
            if signal_plot:
                test_returns = self._run_strategies(
                    *params, 
                    **{k: v for k, v in self.params.items() if k not in ['ngrids', 'thresholds']},
                    total_investment=total_investment, 
                    signal_plot=True,
                    debug=False, 
                    use_cache=False, 
                    montecarlo=montecarlo,
                    start=test_start, 
                    end=test_end+1, 
                    multithread=False, 
                ).values
            else:
                test_returns = test_data.loc[test_start:test_end, params].values

            returns.loc[test_start:test_end] = test_returns
            
            all_test_returns = test_data.loc[test_start:test_end].values
            pr_values.loc[test_start:test_end] = np.mean(all_test_returns <= test_returns[:, None], axis=1) * 100
            median_returns.loc[test_start:test_end] = np.nanpercentile(np.where(all_test_returns == 0.0, np.nan, all_test_returns), 50, axis=1)

        if test_end == len(self.weekends):
            train_start += update_period
            train_end = train_start + optimize_window - 1
            train_data = results.loc[train_start:train_end]
            
            train_results = train_data.apply(lambda col: metric_fn(col.values, 52))
            
            params = train_results.nlargest(rank).index[-1]

        self.all_params = all_params
        self.latest_params = (int(params[0]), float(params[1]))

        indices = slice(start, end)
        times = self.times[indices]
        
        if result_plot and len(returns) > 1:

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=False, gridspec_kw={'height_ratios': [1, 1, 1]})
            
            # Panel 1: Median Returns with Rolling Mean
            median_returns = np.array(median_returns)
            ax1.fill_between(times, 0, np.where(median_returns > 0, median_returns, 0), step='pre', color='green', alpha=0.5)
            ax1.fill_between(times, 0, np.where(median_returns <= 0, median_returns, 0), step='pre', color='red', alpha=0.5)
            ax1.plot(times, pd.Series(median_returns).rolling(13, min_periods=1, center=True).mean(), label=f'SMA {13}', color='orange', linewidth=2)

            ax1.set_ylabel('Return')
            ax1.set_title('Median Returns Through Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Panel 2: RAR for each rank
            rsr_values = [r_sharpe_ratio(np.array(all_rank_returns[r]), 52) for r in ranks]
            ax2.plot(ranks, rsr_values, label='R-Sharpe Ratio', color='lightgray')
            ax2.plot(ranks, pd.Series(rsr_values).rolling(10, min_periods=1, center=True).mean(), label=f'SMA {10}', color='orange', linewidth=2)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Rank')
            ax2.set_ylabel('R-Sharpe Ratio')
            ax2.set_title('R-Sharpe Ratio vs Rank')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Panel 3: PR values for chosen rank through time
            ax3.plot(times, pr_values, label=f'Percentile Rank (Rank {rank})', color='skyblue', linestyle='-', marker='o', markersize=3)
            ax3.plot(times, pd.Series(pr_values).rolling(13, min_periods=1, center=True).mean(), label=f'SMA {13}', color='orange', linewidth=2)
            ax3.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50th Percentile')
            ax3.set_ylabel('Percentile Rank (%)')
            ax3.set_xlabel('Time')
            ax3.set_title(f'Percentile Rank Through Time (Rank {rank})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        
        if not montecarlo:
            self.returns = returns
        else:
            self.montecarlo_returns = returns
            
        buy_hold = np.array([weekend['close'].iloc[-1] for weekend in self.weekends[start:end]]) / self.weekends[start]['open'].iloc[0]
        return analyze_returns(np.asarray(returns), times, 52, buy_hold, result_plot)
    
    def get_grid_params(self, initial_price, ngrid, threshold, risk_per_trade=RISK_PER_TRADE, mode=MODE,
                           total_investment=TOTAL_INVESTMENT):
        """
        Get the latest grid parameters for the given ngrid and threshold.
        """
        low, high, low_sl, high_sl = self._grid_low_high_sl(initial_price, ngrid, threshold)
        _, lower_grids, higher_grids, _ = GridStrategy.get_grids(low, high, ngrid, initial_price, mode)
        qty = GridStrategy.get_qty(total_investment, 1, lower_grids, higher_grids)
        leverage = self._adjust_leverage(initial_price, qty,
                                         lower_grids, higher_grids,
                                         low_sl, high_sl, 
                                         risk_per_trade,
                                         total_investment)
        qty = GridStrategy.get_qty(total_investment, leverage, lower_grids, higher_grids)
        params = pd.DataFrame([{
            'low': low,
            'high': high,
            'ngrid': ngrid,
            'mode': mode,
            'leverage': leverage,
            'total_investment': total_investment,
            'low_sl': low_sl,
            'high_sl': high_sl,
            'qty': qty,
            'start_price': initial_price,
        }])
        params = params.rename(index={0: 'Value'})
        params.columns.name = 'GridStrategy Parameters'
        return params
    
    def _generate_montecarlo_week(self, n):
        start_idices = np.random.choice(range(self.data.shape[0] - 10080), n)
        weekdays = [self.data.iloc[idx:idx+7080].reset_index(drop=True) for idx in start_idices]
        weekends = [self.data.iloc[idx+7080:idx+10080].reset_index(drop=True) for idx in start_idices]
        return weekdays, weekends
    
    def montecarlo_simulation(self):
        print('Strategy Rolling Optimize:')
        self.rolling_optimize(result_plot=False)
        equity_curve = (self.returns + 1).cumprod()
        
        n = 1000
        npy_path = self.data_path / Path(f'npy/{self.__class__.__name__}_montecarlo.npy')
        if npy_path.exists():
            all_montecarlo_equity_curve = np.load(npy_path)
            if all_montecarlo_equity_curve.shape[1] != equity_curve.shape[0]:
                new_all_montecarlo_equity_curve = np.empty((n, equity_curve.shape[0]), dtype=float)
                for i, mon_ec in tqdm(enumerate(all_montecarlo_equity_curve), desc='Monte Carlo Simulations'):
                    self.data = self.data # Update montecarlo data
                    start = len(self.weekends) + mon_ec.shape[0] - equity_curve.shape[0]
                    self.rolling_optimize(montecarlo=True, start=start, result_plot=False, verbose=0)
                    new_all_montecarlo_equity_curve[i] = np.r_[mon_ec, ((self.montecarlo_returns + 1) * mon_ec[-1]).cumprod()]
                    
                all_montecarlo_equity_curve = new_all_montecarlo_equity_curve
                np.save(npy_path, all_montecarlo_equity_curve)
        else:
            all_montecarlo_equity_curve = np.empty((n, equity_curve.shape[0]), dtype=float)
            for i in tqdm(range(n), desc='Monte Carlo Simulations'):
                self.data = self.data # Update montecarlo data
                self.rolling_optimize(montecarlo=True, result_plot=False, verbose=0)
                all_montecarlo_equity_curve[i] = (self.montecarlo_returns + 1).cumprod()
                
            np.save(npy_path, all_montecarlo_equity_curve)
                
        times = self.times[-equity_curve.shape[0]:]
        plt.plot(times, equity_curve, label='Strategy', color='green')
        for i, montecarlo_cum_returns in enumerate(all_montecarlo_equity_curve):
            if i == 0:
                plt.plot(times, montecarlo_cum_returns, label='Monte Carlo', alpha=0.5, color='lightgray')
            else:
                plt.plot(times, montecarlo_cum_returns, alpha=0.5, color='lightgray')

        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.title('Monte Carlo Simulation of Weekend Grid Strategy')
        plt.xlabel('Time')
        plt.ylabel('Equity Curve')
        plt.show()


class WeekendGridV1(WeekendGrid): 
    @WeekendGrid.data.setter
    def data(self, value):
        WeekendGrid.data.fset(self, value)
        self._preprocess()
        self._preprocess(montecarlo=True)
       
       
    @staticmethod
    def _get_default_thresholds(target):
        if target == 'tr':
            return np.arange(300, 1001, 25) / 1000
        elif target == 'atr':
            return np.arange(40, 181, 5) / 1000
        else:
            return None

       
    @staticmethod 
    def _get_atr(data):
        data = convert_interval(data)
        return (data['high'] - data['low']).mean() / data.at[0, 'open'] * data.shape[0]
    
    
    @staticmethod
    def _get_tr(data):
        return (data['high'].max() - data['low'].min()) / data.at[0, 'open']
    
    
    def _preprocess(self, montecarlo=False): 
        features = ['atr', 'tr']
        weekday_data = pd.DataFrame(np.zeros((len(self.weekdays), len(features))), columns=features)
        weekend_data = pd.DataFrame(np.zeros((len(self.weekends), len(features))), columns=features)
        
        if not montecarlo:
            weekdays, weekends = self.weekdays, self.weekends
        else:
            weekdays, weekends = self.montecarlo_weekdays, self.montecarlo_weekends
        
        for price_data, feature_data in zip(
            [weekdays, weekends], 
            [weekday_data, weekend_data], 
        ):
            for i, df in enumerate(price_data):
                feature_data.at[i, 'atr'] = self._get_atr(df)
                feature_data.at[i, 'tr'] = self._get_tr(df)
            
        if not montecarlo:
            self.weekday_data = weekday_data
            self.weekend_data = weekend_data
        else:
            self.montecarlo_weekday_data = weekday_data
            self.montecarlo_weekend_data = weekend_data
        
    def _predictor(self, feature, target, window, filtering, montecarlo=False):
        if not montecarlo:
            weekday_data = self.weekday_data
            weekend_data = self.weekend_data
        else:
            weekday_data = self.montecarlo_weekday_data
            weekend_data = self.montecarlo_weekend_data
        
        mask = np.zeros_like(weekend_data[target], dtype=bool)
        
        ratios: pd.Series = weekend_data[target] / weekday_data[feature]

        if filtering == 'raw':
            ratio_ema = ratios.ewm(span=window).mean()
        elif filtering == 'stable':
            mask[window:] = np.array([
                ~is_extreme(ratios[i-window:i], ratios[i])
                for i in range(window, len(ratios))
            ])

            ratio_ema = pd.Series(index=ratios.index, dtype=float)
            ratio_ema[window-1] = ratios[:window].mean()
            for i in range(window, len(ratio_ema)):
                if mask[i]:
                    ratio_ema[i] = (ratio_ema[i-1] * (window - 1) + ratios[i]) / window
                else:
                    ratio_ema[i] = ratio_ema[i-1]
        else:
            raise ValueError('Filtering must be either "raw" or "stable"')  
            
        return ratio_ema
        
    def _run_strategies(self, ngrid, threshold, feature, target, window, filtering, 
                        risk_per_trade, mode, signal_plot, total_investment, debug, use_cache, montecarlo,
                        start, end, multithread):
        def _task(i):
            dyn_threshold = (feature_data[i] * ratio_ema[i-1] * threshold)
            return self.grid_strategy(
                i,
                ngrid,
                dyn_threshold,
                risk_per_trade=risk_per_trade,
                mode=mode,
                total_investment=total_investment,
                signal_plot=signal_plot,
                debug=debug,
                use_cache=use_cache,
                montecarlo=montecarlo,
            )
        
        feature_data = self.weekday_data[feature].values
        ratio_ema = self._predictor(feature, target, window, filtering, montecarlo=montecarlo)
            
        start = max(self._to_index(window if start is None else start), window)
        end = self._to_index(len(self.weekends) if end is None else end)

        if multithread:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(_task, range(start, end)))
        else:
            results = [_task(i) for i in range(start, end)]
        
        return pd.Series(results, index=range(start, end))
  
    def get_grid_params(self, weekday, ngrid, threshold, feature='atr', target='atr', window=52, 
                           filtering='stable', risk_per_trade=RISK_PER_TRADE, mode=MODE,
                           total_investment=TOTAL_INVESTMENT):
        """
        Get the latest grid parameters for the given ngrid and threshold.
        """
        ratio = self._predictor(feature, target, window, filtering).iloc[-1]
        if feature == 'atr':
            x = self._get_atr(weekday)
        elif feature == 'tr':
            x = self._get_tr(weekday)

        dyn_threshold = x * ratio * threshold
        return super().get_grid_params(
            weekday.at[len(weekday)-1, 'close'], ngrid, dyn_threshold, 
            risk_per_trade=risk_per_trade, mode=mode, total_investment=total_investment
        )
    
    @property
    def params(self):
        if hasattr(self, '_params'):
            return self._params
        
        self._params = super().params
        self._params.update({
            'feature': None,
            'target': None,
            'window': None,
            'filtering': None,
        })
        return self._params
    
    def rolling_optimize(self, optimize_window=104, update_period=13, ngrids=None, thresholds=None, 
                                metric='R-Sharpe Ratio', rank=1,
                                feature='atr', target='atr', window=52, filtering='stable', 
                                risk_per_trade=RISK_PER_TRADE, mode=MODE, 
                                total_investment=TOTAL_INVESTMENT, debug=False, use_cache=False, montecarlo=False,
                                start=None, end=None, 
                                multithread=False, multiprocess=False, 
                                result_plot=True, signal_plot=False, verbose=1, **kwargs):
        mp = dict(
            feature=feature,
            target=target,
            window=window,
            filtering=filtering,
        )
        kwargs.update(mp)
        
        if ngrids is None:
            ngrids = self._get_default_ngrids()
        if thresholds is None:
            thresholds = self._get_default_thresholds(target)

        start = max(self._to_index(0 if start is None else start), window+optimize_window)
        return super().rolling_optimize(
            optimize_window, update_period, ngrids, thresholds,
            metric=metric, rank=rank,
            risk_per_trade=risk_per_trade, mode=mode,
            total_investment=total_investment, debug=debug, use_cache=use_cache, montecarlo=montecarlo,
            start=start, end=end,
            multithread=multithread, multiprocess=multiprocess,
            result_plot=result_plot, signal_plot=signal_plot, verbose=verbose,
            **kwargs
        )
    
if __name__ == '__main__':
    wg0 = WeekendGrid()
    wg0.rolling_optimize(start=datetime(2024, 1, 1))

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import Series, DataFrame
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def drawdowns(cum_returns: ndarray | List | Series) -> ndarray:
    """
    Calculate the drawdown series from cumulative returns.

    Parameters:
        cum_returns (array-like): Cumulative returns.
        
    Returns:
        numpy.ndarray: Array of drawdown values.
    """
    equity_curve = np.asarray(cum_returns) + 1
    running_max = np.maximum.accumulate(equity_curve)
    dds = equity_curve / running_max - 1
    return dds

def max_drawdown(cum_returns: ndarray | List | Series) -> float:
    """
    Calculate the maximum drawdown from cumulative returns.
    
    Parameters:
        cum_returns (array-like): Cumulative returns.
        
    Returns:
        float: Maximum drawdown value.
    """
    dds = drawdowns(cum_returns)
    return dds.min()

def compund_annual_growth_rate(returns: ndarray | List | Series, trading_periods: int) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) from a sequence of returns.
    
    Parameters:
        returns (array-like): Sequence of returns.
        trading_periods (int): Number of trading periods in a year.
        
    Returns:
        float: CAGR value.
    """
    returns = np.asarray(returns)
    total_return = (1 + returns).prod() - 1
    return (1 + total_return) ** (trading_periods / returns.size) - 1

def regressed_annual_return(returns: ndarray | List | Series, 
                           trading_periods: int, 
                           returnline: bool = False) -> float | Tuple[float, ndarray]:
    """
    Calculate the annualized return using linear regression on log returns.
    
    Parameters:
        returns (array-like): Sequence of returns.
        trading_periods (int): Number of trading periods in a year.
        returnline (bool): If True, return the regression line as well.
        
    Returns:
        float: Annualized return.
        numpy.ndarray: Regression line (if returnline is True).
    """
    returns = np.asarray(returns)
    if returns.size < 2:
        if returnline:
            raise ValueError('Not enough data to compute regressed annual return')
        return compund_annual_growth_rate(returns, trading_periods)
    
    t = np.arange(returns.size) / trading_periods  # time vector in years
    
    equity_curve = (1 + returns).cumprod()
    # Linearize multiplicative growth via log
    log_equity = np.log(equity_curve)
    slope, intercept = np.polyfit(t, log_equity, 1)
    cagr = np.exp(slope) - 1
    if returnline:
        return cagr, np.exp(slope * t + intercept)
    return cagr

def sharpe_ratio(returns: ndarray | List | Series, trading_periods: int) -> float:
    """
    Calculate the Sharpe Ratio from a sequence of returns.
    
    Parameters:
        returns (array-like): Sequence of returns.
        trading_periods (int): Number of trading periods in a year.
        
    Returns:
        float: Sharpe Ratio value.
    """
    returns = np.asarray(returns)
    std = returns.std()
    mean = returns.mean()
    sr = mean / std if std != 0 else mean * np.inf
    return sr * np.sqrt(trading_periods)

def r_sharpe_ratio(returns: ndarray | List | Series,
                   trading_periods: int) -> float:
    """
    Regressed Sharpe Ratio (RSR).

    - Expected return is estimated by a log-equity regression (RAR),
      then converted back to the period horizon.
    - Risk is the period standard deviation of raw returns.
    - The whole ratio is annualised with √trading_periods.

    Parameters
    ----------
    returns : array-like
        Sequence of period returns (e.g. weekly if trading_periods=52).
    trading_periods : int
        Number of such periods in a year.

    Returns
    -------
    float
        Regressed Sharpe Ratio.
    """
    returns = np.asarray(returns)

    # 1. Expected annual return from the regression (geometric)
    rar = regressed_annual_return(returns, trading_periods)      # annual net %

    # 2. Convert that to an *arithmetic* mean for one period
    mean = (1 + rar) ** (1 / trading_periods) - 1           # same horizon as `returns`

    # 3. Risk on the same horizon
    std = returns.std()

    # 4. Period Sharpe, then annualise
    if std == 0:
        return mean * np.inf
    rsr = mean / std
    return rsr * np.sqrt(trading_periods)

def _streaks(mask: ndarray) -> ndarray:
    """
    Helper function to calculate streaks in a boolean array.
    
    Parameters:
        mask (numpy.ndarray): Boolean array.
        
    Returns:
        numpy.ndarray: Array of streak lengths.
    """
    # Pad with 0's to detect boundaries
    padded = np.r_[0, mask.astype(int), 0]
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return ends - starts if starts.size else np.empty(0)

def longest_streak(mask: ndarray) -> int:
    """
    Calculate the longest streak of consecutive True values in a boolean mask.
    
    Parameters:
        mask (array-like): Boolean array.
        
    Returns:
        int: Length of the longest streak.
    """
    streaks = _streaks(mask)
    return streaks.max() if streaks.size else 0

def robust_risk_reward_ratio(returns: ndarray | List | Series, trading_periods: int) -> float:
    """
    Calculate the Robust Risk-Reward Ratio from a sequence of returns.  
    The ratio is calculated by regressed annual return divided by the product of 
    top 5 maximum drawdown and top 5 longest drawdown period.

    Parameters:
        returns (array-like): Sequence of returns.
        trading_periods (int): Number of trading periods in a year.
        
    Returns:
        float: Robust Risk-Reward Ratio value.
    """
    rar = regressed_annual_return(returns, trading_periods)

    cum_returns = (np.asarray(returns) + 1).cumprod()
    dds = drawdowns(cum_returns)
    partitions = np.split(dds, np.where(dds == 0)[0])
    mdds = np.sort(np.array([part.min() for part in partitions if part.size]))
    mdd_bar = mdds[:5].mean() if mdds.size > 5 else mdds.mean()

    dd_streaks = np.sort(_streaks(dds < 0))
    ldd_bar = dd_streaks[-5:].mean() if dd_streaks.size > 5 else dd_streaks.mean()
    ldd_bar /= trading_periods
    if mdd_bar == 0 or ldd_bar == 0:
        return rar * np.inf
    return rar / (-mdd_bar * ldd_bar)

def sortino_ratio(returns: ndarray | List | Series, trading_periods: int) -> float:
    """
    Calculate the Sortino Ratio from a sequence of returns.
    
    Parameters:
        returns (array-like): Sequence of returns.
        trading_periods (int): Number of trading periods in a year.
        
    Returns:
        float: Sortino Ratio value.
    """
    returns = np.asarray(returns)
    downside_returns = np.where(returns < 0, returns, 0)
    downside_std = downside_returns.std()
    mean = returns.mean()
    return mean / downside_std * np.sqrt(trading_periods) if downside_std != 0 else 0

def calmar_ratio(returns: ndarray | List | Series, trading_periods: int) -> float:
    """
    Calculate the Calmar Ratio from a sequence of returns.
    
    Parameters:
        returns (array-like): Sequence of returns.
        trading_periods (int): Number of trading periods in a year.
        
    Returns:
        float: Calmar Ratio value.
    """
    returns = np.asarray(returns)
    mdd = max_drawdown((returns + 1).cumprod())
    cagr = compund_annual_growth_rate(returns, trading_periods)
    return cagr / -mdd if mdd != 0 else cagr * np.inf

def profit_factor(returns: ndarray | List | Series) -> float:
    """
    Calculate the Profit Factor from a sequence of returns.
    
    Parameters:
        returns (array-like): Sequence of returns.
        
    Returns:
        float: Profit Factor value.
    """
    returns = np.asarray(returns)
    gain = returns[returns > 0].sum()
    loss = -returns[returns < 0].sum()
    return gain / loss if loss != 0 else gain * np.inf

def drawdown_periods(cum_returns: ndarray | List | Series) -> ndarray:
    """
    Calculate the current drawdown period count at each time point.  
    A drawdown period increases by 1 when price is below running max, else resets to 0.
    
    Parameters:
        cum_returns (array-like): Cumulative returns.
        
    Returns:
        numpy.ndarray: Array of integers representing drawdown period at each index.
    """
    cum_returns = np.asarray(cum_returns)
    running_max = np.maximum.accumulate(cum_returns)
    periods = np.zeros_like(cum_returns, dtype=int)
    count = 0
    for i, price in enumerate(cum_returns):
        if price < running_max[i]:
            count += 1
        else:
            count = 0
        periods[i] = count
    return periods

def longest_drawdown(cum_returns: ndarray | List | Series) -> int:
    """
    Calculate the longest drawdown period from cumulative returns.
    
    Parameters:
        cum_returns (array-like): Cumulative returns.
        
    Returns:
        int: Length of the longest drawdown period.
    """
    return drawdown_periods(cum_returns).max()

def analyze_returns(returns: ndarray, times: ndarray, trading_period: int, buy_hold: Optional[ndarray] = None, result_plot=True) -> DataFrame:
        if returns.size < 2:
            return
    
        equity_curve = (returns + 1).cumprod()
        cum_returns = equity_curve - 1
        
        # Compute regressed annual returns (RAR)
        rar, reg_line = regressed_annual_return(returns, trading_period, returnline=True)
        
        if result_plot:
            # Compute drawdowns
            dd = drawdowns(cum_returns)
            
            # Compute drawdown periods
            dd_periods = drawdown_periods(cum_returns)
            
            plt.style.use('default')
            fig = plt.figure(figsize=(16, 12))
            gs = GridSpec(nrows=5, ncols=1, height_ratios=[2, 1, 1, 1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
            ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
            ax5 = fig.add_subplot(gs[4, 0])
            
            ax1.plot(times, equity_curve, label='Equity Curve')
            ax1.plot(times, reg_line, label='Regression Line', linestyle='--', color='red')
            if buy_hold is not None:
                ax1.plot(times, buy_hold, label='Buy & Hold', linestyle='-', color='blue')
            
            ax1.set_title(f'RAR={rar:.2%}')
            ax1.set_ylabel('Equity Curve')
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True)
            
            pos_returns = np.where(returns > 0, returns, 0)
            neg_returns = np.where(returns <= 0, returns, 0)
            ax2.fill_between(times, pos_returns, step='pre', alpha=0.5, color='green')
            ax2.fill_between(times, neg_returns, step='pre', alpha=0.5, color='red')
            ax2.set_ylabel('Returns')
            ax2.grid(True)
            
            ax3.fill_between(times, dd, 0, step='pre', alpha=0.7, color='red')
            ax3.set_ylabel('Drawdown')
            ax3.grid(True)
            
            ax4.fill_between(times, dd_periods, 0, step='pre', alpha=0.7, color='orange')
            ax4.set_ylabel('Drawdown Periods')
            ax4.set_xlabel('Time')
            ax4.grid(True)
            
            ax5.hist(returns, bins=200, edgecolor='black', alpha=0.7)
            ax5.set_ylabel('Frequency')
            ax5.set_xlabel('Returns')
            ax5.grid(True)
            ax5.axvline(x=0, color='red', linestyle='--', label='Zero Line')

            plt.suptitle(f'Equity Curves, {"Buy & Hold Comparison, " if buy_hold is not None else ""}Drawdowns, Drawdown Periods and Return Distribution', fontsize=16)
            plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
            plt.show()
            
        sr = sharpe_ratio(returns, trading_period)
        rsr = r_sharpe_ratio(returns, trading_period)
        rrr = robust_risk_reward_ratio(returns, trading_period)
        sortino = sortino_ratio(returns, trading_period)
        calmar = calmar_ratio(returns, trading_period)
        
        mdd = max_drawdown(cum_returns)
        ldd = longest_drawdown(cum_returns)
        vol = np.std(returns)

        cagr = compund_annual_growth_rate(returns, trading_period)
        pf = profit_factor(returns)
        nonzero_returns = returns[returns != 0]
        wr = np.sum(nonzero_returns > 0) / nonzero_returns.size
        trade_freq = nonzero_returns.size / returns.size
        
        longest_win_streak = longest_streak(returns > 0)
        longest_loss_streak = longest_streak(returns < 0)
        
        res = pd.DataFrame([{
            # Risk-Adjusted Metrics
            'Sharpe Ratio': f'{sr:.2f}',
            'R-Sharpe Ratio': f'{rsr:.2f}',
            'Robust Risk-Reward Ratio': f'{rrr:.2f}',
            'Sortino Ratio': f'{sortino:.2f}',
            'Calmar Ratio': f'{calmar:.2f}',
            
            # Risk Metrics
            'Maximum Drawdown': f'{mdd:.2%}',
            'Longest Drawdown': ldd,
            'Volatility': f'{vol:.2%}',

            # Performance Metrics
            'Compounded Annual Growth Rate': f'{cagr:.2%}',
            'Regressed Annual Return': f'{rar:.2%}',
            'Profit Factor': f'{pf:.2f}',
            'Win Rate': f'{wr:.2%}',
            'Trade Frequency': f'{trade_freq:.2%}',
            'Average Return': f'{np.mean(returns):.2%}',
            'Median Return': f'{np.median(returns):.2%}',
            'Maximum Return': f'{max(returns):.2%}',
            'Minimum Return': f'{min(returns):.2%}',
            'Average Win Return': f'{np.mean(returns[returns > 0]):.2%}',
            'Average Loss Return': f'{np.mean(returns[returns < 0]):.2%}',
            'Maximum Win Streak': longest_win_streak,
            'Maximum Loss Streak': longest_loss_streak,
        }])
        
        return res.T.rename(columns={0: 'Value'}).rename_axis('Metric').style.format(na_rep='-').data
# Weekend Grid Strategy
Runs neutral grid strategy on BTCUSDT in every weekend.  
## Neutral Grid Strategy
- Splits prices from `low` to `high` into `ngrid` spreads, `ngrid + 1` grid levels.
- Long when the grid level is above `start_price`; Short when the grid level is below `start_price`.
- Ignore the grid level closest to `start_price`.
- Close position and cancel all orders if:
    - `low_sl` or `high_sl` is triggered.
    - `end_time` is reached.

## Weekend
- Starts when US stock market closes (04:00 Sat. UTC+8)
- Ends when BTC1! opens (06:00 Mon. UTC+8)
- `start_time` and `end_time` need to adjust for daylight saving time (05:00 Sat. ~ 07:00 Mon. UTC+8)

# Grid Parameters  
`predictor`:
- The exponential moving average of the ratio: weekend `target` / weekday `feature`, where `'atr'` are the default values for `target` and `feature`.
- `atr = Mean(high - low) / first_price`
- By setting `filtering` to `'stable'`, it uses the previous value of EMA to update the EMA if the computed value is extreme (val < Q1 - 1.5IQR or val > Q3 + 1.5IQR, where Q1, Q3 are the 25th and 75th percentile, IQR = Q3 - Q1). This removes the effect by extremes.
- EMA times `threshold` will be `dyn_threshold`. It decides the upper bound and lower bound of grid strategy.

**Stop Losses and Leverage**:
- Stop losses are set at one grid level above `high` and one grid level below `low`.
- Dynamically adjust the leverage such that the maximum loss per strategy is fixed. Default -10%.

`rolling_optimize`:
- Backtest all the combination of `ngrids` and `thresholds` in the `optimize_window`, choose the `rank`-th best performing parameters (`ngrid`, `threshold`).
- The performance is judged by `metric`. Default: `'R-Sharpe Ratio'`, Regressed Sharpe Ratio.
- The parameters are updated once for `update_period` period of time.

# Returns
## Plot
- **Overall Grid Strategy the Chosen Parameters Performance**:
  1. Median returns (among all the parameter combinations) box plot through time. Shows whether the market is suitable for grid strategy.
  2. The `Metric` value across all `rank`. Shows the overall performances of all `rank`. This gives an insight of choosing the optimal `rank`.
  3. The percentile rank of `rank` through time. Shows the detailed performance of the chosen `rank`.
 
- `result_plot`
  1. Equity curve, trend line, and buy and hold curve, with Regressed Annual Return (RAR).
  2. Return box plot through time.
  3. Drawdown box plot through time.
  4. Cumulative drawdown period box plot through time.
  5. Returns histogram.

## Return Value
- **Dataframe consist of multiple metrics**
  - Sharpe Ratio
  - Maximum Drawdown
  - CAGR
  - etc.
  

# Run Backtest
## Check Python installation
```
python3 --version
```

## Create a virtual environment
```
python3 -m venv venv
```

## Activate the environment
- macOS/Linux:
  ```
  source venv/bin/activate
  ```
- Windows (PowerShell):
  ```
  venv\Scripts\Activate.ps1
  ```

- Windows (cmd.exe):
  ```
  venv\Scripts\activate.bat
  ```

## Install packages
```
pip install -r requirements.txt
```

## Fetch klines data
```
python fetching.py
```

## Backtest results
Run the cells in `WeekendGrid.ipynb`

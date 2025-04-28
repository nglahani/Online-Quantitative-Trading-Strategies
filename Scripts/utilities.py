#Library Imports
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

pd.set_option('future.no_silent_downcasting', True)

##############################################################################
# PROJECT / UTILITY CODE
##############################################################################
def get_all_tickers(folder_path):
    """
    Scans subdirectories for CSV files named table_<TICKER>.csv
    and returns a list of all unique tickers found.
    """
    folder_names = [
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name)) and name.startswith('allstocks')
    ]

    ticker_set = set()

    for date_folder in folder_names:
        date_path = os.path.join(folder_path, date_folder)
        for fname in os.listdir(date_path):
            # Check for files named "table_<TICKER>.csv"
            if fname.startswith("table_") and fname.endswith(".csv"):
                ticker = fname[len("table_"):-len(".csv")]
                ticker_set.add(ticker)

    return sorted(ticker_set)

def initialize_portfolio(m):
    return np.ones(m) / m


def calculate_price_relative_vectors(folder_path, tickers):
    """
    Create a DataFrame of price relative vectors for each day and each ticker.
    If a ticker did not exist (missing file) on a certain day, the close price is NaN.
    Then we forward-fill the close prices so that short gaps become continuous.
    Finally, the ratio x_{t} = Close[t]/Close[t-1]. If no prior close is available
    (e.g. brand-new ticker), set ratio = 1.0 on that day.
    """

    # 1) Gather all folder names that start with "allstocks_"
    folder_names = [
        name for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name)) and name.startswith("allstocks_")
    ]
    folder_names.sort()  # Sort chronologically by folder name (YYYYMMDD)

    # 2) Parse out dates from folder names and build an index
    date_list = []
    for folder_name in folder_names:
        date_str = folder_name.replace("allstocks_", "")
        date_obj = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
        date_list.append(date_obj)

    # Prepare a DataFrame of shape (num_days, num_tickers) with NaN placeholders
    price_df = pd.DataFrame(index=date_list, columns=tickers, dtype=float)
    
    column_names = [
        'Date', 'Time', 'Open', 'High', 'Low', 
        'Close', 'Volume', 'Split Factor', 'Earnings', 'Dividends'
    ]

    folder_counter = 0
    # 3) Loop over each folder/day and fill in close price if file is found
    for folder_name in folder_names:
        folder_counter +=1
        date_str = folder_name.replace("allstocks_", "")
        date_obj = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")

        day_path = os.path.join(folder_path, folder_name)

        for ticker in tickers:
            file_name = f"table_{ticker}.csv"
            file_path = os.path.join(day_path, file_name)

            # If file doesn’t exist => skip (no data for that day/ticker)
            if not os.path.exists(file_path):
                continue

            # Read CSV
            df = pd.read_csv(file_path, header=None)
            df.columns = column_names

            # Take the last row's close as "the close" for that day
            last_close = df['Close'].iloc[-1]

            # Store it in price_df
            price_df.at[date_obj, ticker] = last_close
                # Print progress every 100 folders processed
        if folder_counter % 100 == 0:
            print(f"Processed {folder_counter}/{len(folder_names)} folders")

    # 4) Forward-fill the close prices across days
    price_df.sort_index(inplace=True)  # Make sure it's sorted by date
    price_df.fillna(method="ffill", inplace=True)

    # 5) Compute price relatives: ratio[t] = Close[t] / Close[t-1]
    price_relative_df = price_df / price_df.shift(1)
    # The very first day is NaN => set to 1.0
    price_relative_df.iloc[0, :] = 1.0

    # If a ticker only appears mid-period, previous day remains NaN => set to 1.0
    price_relative_df.fillna(1.0, inplace=True)

    # Prevent zero or negative (shouldn't happen, but just in case)
    price_relative_df[price_relative_df <= 0] = 1e-10

    return price_relative_df


def calculate_period_return(b_t, x_t):
    """ Calculate the return of the portfolio in a single period. """
    return np.dot(b_t, x_t)

def calculate_cumulative_wealth(b_n_1, price_relative_vectors, S0=1.0):
    """
    Calculate the final cumulative wealth after investing over len(price_relative_vectors) periods.
    """
    cumulative_wealth = S0
    for t, x_t in enumerate(price_relative_vectors):
        period_return = calculate_period_return(b_n_1[t], x_t)
        cumulative_wealth *= period_return
    return cumulative_wealth

def calculate_exponential_growth_rate(Sn, n, S0=1.0):
    """ Calculate the exponential growth rate (average log growth). """
    return (1 / n) * np.log(Sn / S0)


def calculate_cumulative_wealth_over_time(b_n_1, price_relative_vectors, S0=1.0):
    """
    Calculate the cumulative wealth path (array) over time.
    """
    T = len(price_relative_vectors)
    cumulative_wealth = np.zeros(T)
    wealth = S0
    for t in range(T):
        r_t = calculate_period_return(b_n_1[t], price_relative_vectors[t])
        wealth = wealth * r_t
        cumulative_wealth[t] = wealth
    return cumulative_wealth

def compute_periodic_returns(cumulative_wealth):
    """
    Given a time series of portfolio wealth, compute the *periodic* returns array.
    If W_t is the wealth at time t, then the return for time t is (W_t / W_{t-1} - 1).
    The first return is NaN or 0 by convention, so we drop it.
    """
    if len(cumulative_wealth) < 2:
        return np.array([])
        
    w = pd.Series(cumulative_wealth)
    # Add small constant to prevent division by zero
    w = w.clip(lower=1e-10)
    
    # Calculate returns with explicit fill_method=None to address deprecation warning
    returns = w.pct_change(fill_method=None).dropna()
    return returns.values

def compute_sharpe_ratio(returns, freq=252, risk_free_rate=0.05):
    """
    Compute the (annualized) Sharpe ratio given a series of periodic returns.
    :param returns: A NumPy or pandas array of *periodic* returns. 
                    (For daily data, each element could be the single-day return.)
    :param freq: Number of periods in a year (default 252 for daily data)
    :param risk_free_rate: Risk-free return for one *year*. If 0, then no adjustment.
    :return: Sharpe ratio (float)
    """
    if len(returns) == 0:
        return np.nan
        
    # Convert annual risk-free rate to a single-period risk-free return
    rf_periodic = (1 + risk_free_rate)**(1/freq) - 1
    
    # Replace any infinite values with NaN
    returns = np.where(np.isfinite(returns), returns, np.nan)
    
    # Excess returns over each period
    excess_returns = returns - rf_periodic
    
    # Handle edge-cases
    if len(excess_returns) == 0 or np.all(np.isnan(excess_returns)):
        return np.nan
    
    # Calculate std with ddof=1 for sample standard deviation
    std_excess = np.nanstd(excess_returns, ddof=1)
    if std_excess < 1e-10:
        return np.nan
    
    # Mean of excess returns (excluding NaN values)
    mean_excess = np.nanmean(excess_returns)
    
    # Annualize the Sharpe ratio
    sharpe = (mean_excess / std_excess) * np.sqrt(freq)
    return sharpe if np.isfinite(sharpe) else np.nan

def calculate_maximum_drawdown(cumulative_wealth):
    """
    Calculate the maximum drawdown for a given cumulative wealth series.
    
    :param cumulative_wealth: A NumPy array or pandas Series of cumulative wealth values
    :return: Maximum drawdown as a percentage (a negative value)
    """
    # Convert to numpy array if it's a pandas Series
    wealth = np.array(cumulative_wealth)
    
    # Handle edge cases
    if len(wealth) == 0 or np.any(np.isnan(wealth)):
        return np.nan
        
    # Calculate the running maximum with numeric stability check
    running_max = np.maximum.accumulate(wealth)
    running_max = np.maximum(running_max, 1e-10)  # Prevent division by zero
    
    # Calculate drawdown percentage at each point with numeric stability
    drawdown = np.where(running_max > 0, 
                       (wealth - running_max) / running_max,
                       0.0)  # Fallback for zero running max
    
    # Find the maximum drawdown
    max_drawdown = np.min(drawdown) if not np.all(np.isnan(drawdown)) else np.nan
    
    return max_drawdown

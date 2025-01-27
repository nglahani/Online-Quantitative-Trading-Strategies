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
    Function to calculate the price relative vector for multiple stocks and return them side by side.
    """
    folder_names = [name for name in os.listdir(folder_path) 
                    if os.path.isdir(os.path.join(folder_path, name)) and name.startswith('allstocks')]

    price_relative_df = pd.DataFrame()
    column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Split Factor', 'Earnings', 'Dividends']

    for ticker in tickers:
        close_price_df = pd.DataFrame(columns=['Ticker', 'Date', 'Close'])

        for date in folder_names: 
            stock_file = 'table_' + ticker + '.csv'
            file_path = os.path.join(folder_path, date, stock_file)

            iter_df = pd.read_csv(file_path)
            iter_df.columns = column_names

            last_row = iter_df[['Date', 'Close']].iloc[-1]
            last_row['Ticker'] = ticker
            close_price_df = pd.concat([close_price_df, last_row.to_frame().T], ignore_index=True)

        close_price_df['Date'] = close_price_df['Date'].astype(int)
        close_price_df['Date'] = pd.to_datetime(close_price_df['Date'], format='%Y%m%d')
        close_price_df.set_index('Date', inplace=True)

        close_price_df[ticker] = (close_price_df['Close'] / close_price_df['Close'].shift(1)).fillna(1)

        if price_relative_df.empty:
            price_relative_df = close_price_df[[ticker]]
        else:
            price_relative_df = price_relative_df.join(close_price_df[[ticker]], how='outer')

        price_relative_df = price_relative_df.astype(float)
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
    # shift(1) to get the previous wealth
    # daily_return_t = (W_t / W_{t-1}) - 1
    # We skip the very first period since there's no "previous" wealth in the array.
    
    w = pd.Series(cumulative_wealth)  # convert to pandas for convenience
    returns = w.pct_change().dropna()  # drop the first NaN
    return returns.values  # convert back to a NumPy array

def compute_sharpe_ratio(returns, freq=252, risk_free_rate=0.0):
    """
    Compute the (annualized) Sharpe ratio given a series of periodic returns.
    :param returns: A NumPy or pandas array of *periodic* returns. 
                    (For daily data, each element could be the single-day return.)
    :param freq: Number of periods in a year (default 252 for daily data)
    :param risk_free_rate: Risk-free return for one *year*. If 0, then no adjustment.
    :return: Sharpe ratio (float)
    """
    # Convert annual risk-free rate to a single-period risk-free return
    # For daily: (1 + Rf_annual)^(1/252) - 1
    rf_periodic = (1 + risk_free_rate)**(1/freq) - 1
    
    # Excess returns over each period
    excess_returns = returns - rf_periodic

    # Handle edge-cases (e.g., zero or near-zero standard deviation)
    std_excess = excess_returns.std()
    if std_excess < 1e-14:
        return np.nan
    
    # Mean of excess returns
    mean_excess = excess_returns.mean()
    
    # Annualize the Sharpe ratio
    sharpe = (mean_excess / std_excess) * np.sqrt(freq)
    return sharpe

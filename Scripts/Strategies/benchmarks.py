##############################################################################
# BENCHMARK ALGORITHMS
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

# Strategy 1: Buy and Hold
def buy_and_hold(b, price_relative_vectors):
    """
    Buy-and-hold: invests initial weights 'b' in each asset and never rebalances.
    The returned b_n is purely for reference; the actual wealth grows un-rebalanced.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        # The nominal holdings scale by price_relative_vectors[t-1].
        # Then (optionally) we can track the fraction of total. But by definition,
        # buy-and-hold doesn't rebalance, so the fraction can be seen as
        # (old holdings * x_{t-1}) / sum(...).
        new_portfolio = b_n[t-1] * price_relative_vectors[t-1]
        new_portfolio /= np.sum(new_portfolio)
        b_n[t] = new_portfolio

    return b_n

# Strategy 2: Best Stock
def best_stock(b, price_relative_vectors):
    """
    At each period, invests entirely in whichever stock performed best in the previous period.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    # Start with uniform
    b_n[0] = b

    for t in range(1, T):
        best_idx = np.argmax(price_relative_vectors[t-1])
        new_portfolio = np.zeros(N)
        new_portfolio[best_idx] = 1.0
        b_n[t] = new_portfolio

    return b_n

# Strategy 3: Constant Rebalancing
def constant_rebalancing(b, price_relative_vectors):
    """
    Always hold the same proportion 'b' each period (rebalance to b each time).
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    for t in range(T):
        b_n[t] = b
    return b_n

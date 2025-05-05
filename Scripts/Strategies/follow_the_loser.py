##############################################################################
# FOLLOW-THE-LOSER ALGORITHMS (Optimized)
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from Strategies.helper import *


# Strategy 9: Anti-Correlation (Anticor) - Simplified and Vectorized
def anticor(b, price_relative_vectors, window_size=3, alpha=2.5, corr_threshold=0.5):
    """
    Implements a simplified anticorrelation strategy with additional parameters.
    
    Parameters:
        b: Initial portfolio weight vector.
        price_relative_vectors: A T x N numpy array where each row is a price relative vector.
        window_size: The window size to compute log-price relatives for correlation estimation.
        alpha: Transfer scaling factor to control aggressiveness.
        corr_threshold: Only correlations above this threshold are used in computing transfers.
        
    Returns:
        b_n: A T x N array representing the portfolio weights over time.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        if t >= 2 * window_size:
            # Compute log-price relatives for two successive windows
            y1 = np.log(price_relative_vectors[t - 2*window_size : t - window_size])
            y2 = np.log(price_relative_vectors[t - window_size : t])
            
            if y1.shape[0] > 0 and y2.shape[0] > 0:
                # Compute means (available if needed)
                mean_y1 = np.mean(y1, axis=0)
                mean_y2 = np.mean(y2, axis=0)
                
                # Compute cross-covariance between the two windows.
                # np.cov returns a 2N x 2N matrix, so we take the upper-right quadrant.
                Mcov = np.cov(y1.T, y2.T)[:N, N:]
                std_y1 = np.std(y1, axis=0)
                std_y2 = np.std(y2, axis=0)
                
                # Avoid division by zero
                std_y1[std_y1 == 0] = 1e-10
                std_y2[std_y2 == 0] = 1e-10
                Mcor = Mcov / np.outer(std_y1, std_y2)
                
                # Apply the correlation threshold: only correlations above the threshold are retained.
                pos_corr = np.where(Mcor > corr_threshold, Mcor, 0)
                
                # Compute transfer amounts:
                # For each asset, sum the incoming and outgoing transfer claims from correlations.
                transfer_amounts = np.sum(pos_corr, axis=0) - np.sum(pos_corr, axis=1)
                # Scale the transfer amounts by alpha.
                transfer_amounts *= alpha
                
                # Update the portfolio by applying the transfer amounts,
                # ensuring non-negativity and then renormalizing to sum to one.
                b_star = b_n[t-1] + transfer_amounts
                b_star = np.maximum(b_star, 0)
                if np.sum(b_star) > 0:
                    b_star /= np.sum(b_star)
                b_n[t] = b_star
            else:
                b_n[t] = b_n[t-1]
        else:
            b_n[t] = b_n[t-1]

    return b_n


# Strategy 10: PAMR (Passive Aggressive Mean Reversion)
def pamr(b, price_relative_vectors, epsilon=0.9, C=10.0):
    """
    Implements the PAMR strategy with an additional aggressiveness cap parameter C.
    Uses previous period’s price relatives to compute an adjustment factor.

    Parameters:
      b : numpy array, initial portfolio weights (shape: [N])
      price_relative_vectors : numpy array (shape: [T, N])
      epsilon : sensitivity threshold (default 0.5)
      C : cap for the update step (default 1.0)

    Returns:
      b_n : numpy array of shape (T, N) representing the portfolio weights over time.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        x_t = price_relative_vectors[t-1]
        portfolio_return = np.dot(b_n[t-1], x_t)

        x_t_mean = np.mean(x_t)
        x_t_diff = x_t - x_t_mean
        denom = np.linalg.norm(x_t_diff) ** 2

        if denom > 0:
            tau_t = (portfolio_return - epsilon) / (denom + 1e-15)
            tau_t = max(0, tau_t)         # Ensure non-negative tau
            tau_t = min(C, tau_t)         # Cap the update using C
        else:
            tau_t = 0

        b_t1 = b_n[t-1] - tau_t * x_t_diff
        b_t1 = np.maximum(b_t1, 0)
        if np.sum(b_t1) > 0:
            b_t1 /= np.sum(b_t1)
        b_n[t] = b_t1

    return b_n


#Strategy 11: CWMR (Confidence-Weighted Mean Reversion)
def cwmr(b, price_relative_vectors, epsilon=0.89, theta=0.92, eta=.93):
    """
    Implements a simplified version of CWMR with an additional learning rate factor (eta).
    Maintains a mean vector (mu_t) and covariance matrix (Sigma_t) to update the portfolio.

    Parameters:
        b: Initial portfolio (numpy array)
        price_relative_vectors: numpy array with shape (T, N)
        epsilon: Sensitivity parameter for mean reversion (default 0.5)
        theta: Confidence threshold parameter (default 0.95)
        eta: Learning rate factor to scale the update (default 1.0)

    Returns:
        b_n: numpy array of shape (T, N) with portfolio weights over time.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()

    mu_t = b.copy().astype(np.float64)
    Sigma_t = np.eye(N) * 1.0

    for t in range(1, T):
        x_t = price_relative_vectors[t-1]
        x_t_mean = np.dot(mu_t, x_t)

        # Compute denominator for lambda_t calculation
        denominator = x_t @ (Sigma_t @ x_t)
        if denominator > 0:
            # Incorporate the learning rate factor (eta)
            lambda_t = eta * max(0, (x_t_mean - epsilon) / (denominator + 1e-15))
        else:
            lambda_t = 0

        # Update the mean vector
        mu_t -= lambda_t * (Sigma_t @ x_t)
        # Update the covariance matrix with a small ridge for numerical stability
        Sigma_t_inv = np.linalg.inv(Sigma_t + np.eye(N) * 1e-12)
        Sigma_t_inv += 2 * lambda_t * theta * np.outer(x_t, x_t)
        Sigma_t = np.linalg.inv(Sigma_t_inv)
        # Project the updated mean to the simplex (ensure portfolio constraints)
        mu_t = project_to_simplex(mu_t)
        b_n[t] = mu_t

    return b_n


#Strategy 12: OLMAR 
def olmar(b, price_relative_vectors, window_size=2, epsilon=.8, eta=20):
    """
    Implements a simplified OLMAR strategy with a learning rate multiplier.
    Uses a moving average of past price relatives as a prediction.

    Parameters:
        b: Initial portfolio (numpy array)
        price_relative_vectors: numpy array of price relative vectors (shape: [T, N])
        window_size: Window length for computing the moving average (default 10)
        epsilon: Threshold for triggering the update (default 0.5)
        eta: Learning rate multiplier to scale the update step (default 1.0)

    Returns:
        b_n: numpy array of shape (T, N) representing the portfolio weights over time.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        if t < window_size:
            ma_t = np.mean(price_relative_vectors[:t], axis=0)
        else:
            ma_t = np.mean(price_relative_vectors[t-window_size : t], axis=0)

        # Basic prediction (x_t_tilde) using the moving average
        x_t_tilde = ma_t
        b_t = b_n[t-1]
        x_t_tilde_mean = np.dot(b_t, x_t_tilde)
        if x_t_tilde_mean < epsilon:
            tau = (epsilon - x_t_tilde_mean) / (np.dot(x_t_tilde, x_t_tilde) + 1e-15)
            b_t1 = b_t + eta * tau * (x_t_tilde - b_t)
            b_t1 = project_to_simplex(b_t1)
        else:
            b_t1 = b_t
        b_n[t] = b_t1

    return b_n

#Strategy 13: Robust Median Reversion
def rmr(b, price_relative_vectors, window_size=8, epsilon=1.1, eta=30):
    """
    Implements a robust median reversion (RMR) strategy with an additional learning rate multiplier.
    Uses an L1-median computed over a sliding window of price relatives to form predictions.

    Parameters:
        b: Initial portfolio (numpy array)
        price_relative_vectors: numpy array of price relative vectors (shape: [T, N])
        window_size: Window length for computing the L1-median (default 10)
        epsilon: Threshold for triggering the update (default 0.8)
        eta: Learning rate multiplier to scale the update step (default 1.0)

    Returns:
        b_n: numpy array of shape (T, N) representing the portfolio weights over time.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()

    for t in range(1, T):
        if t < window_size:
            window_data = price_relative_vectors[:t]
        else:
            window_data = price_relative_vectors[t-window_size : t]

        # Compute the L1-median of the window data
        mu_t_plus_1 = calculate_l1_median(np.array(window_data, dtype=np.float64))
        # Predict next price relatives by comparing the median with last observed prices
        x_t_tilde = mu_t_plus_1 / (price_relative_vectors[t-1] + 1e-15)

        b_t = b_n[t-1]
        x_t_tilde_mean = np.dot(b_t, x_t_tilde)
        if x_t_tilde_mean < epsilon:
            tau = (epsilon - x_t_tilde_mean) / (np.dot(x_t_tilde, x_t_tilde) + 1e-15)
            # Scale the update with eta
            b_t1 = b_t + eta * tau * (x_t_tilde - b_t)
            b_t1 = project_to_simplex(b_t1)
        else:
            b_t1 = b_t
        b_n[t] = b_t1

    return b_n


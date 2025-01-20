##############################################################################
# FOLLOW-THE-LOSER ALGORITHMS
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from .. import utilities

# Strategy 9: Anti-Correlation (Anticor) - Simplified
def anticor(b, price_relative_vectors, window_size=5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        if t >= 2 * window_size:
            y1 = np.log(price_relative_vectors[t - 2*window_size : t - window_size])
            y2 = np.log(price_relative_vectors[t - window_size : t])

            if y1.shape[0] > 0 and y2.shape[0] > 0:
                mean_y1 = np.mean(y1, axis=0)
                mean_y2 = np.mean(y2, axis=0)

                Mcov = np.cov(y1.T, y2.T)[:N, N:]
                std_y1 = np.std(y1, axis=0)
                std_y2 = np.std(y2, axis=0)
                std_y1[std_y1 == 0] = 1e-10
                std_y2[std_y2 == 0] = 1e-10
                Mcor = Mcov / np.outer(std_y1, std_y2)

                transfer_amounts = np.zeros(N)
                for i in range(N):
                    for j in range(N):
                        if Mcor[i, j] > 0:
                            transfer_amount = Mcor[i, j]
                            transfer_amounts[i] -= transfer_amount
                            transfer_amounts[j] += transfer_amount

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

# Strategy 10: PAMR
def pamr(b, price_relative_vectors, epsilon=0.5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        x_t = price_relative_vectors[t-1]
        portfolio_return = np.dot(b_n[t-1], x_t)

        x_t_mean = np.mean(x_t)
        x_t_diff = x_t - x_t_mean
        denom = np.linalg.norm(x_t_diff)**2
        if denom > 0:
            tau_t = max(0, (portfolio_return - epsilon) / denom)
        else:
            tau_t = 0

        b_t1 = b_n[t-1] - tau_t * x_t_diff
        b_t1 = np.maximum(b_t1, 0)
        if np.sum(b_t1) > 0:
            b_t1 /= np.sum(b_t1)
        b_n[t] = b_t1

    return b_n

# Strategy 11: CWMR (simplified)
def cwmr(b, price_relative_vectors, epsilon=0.5, theta=0.95):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()

    mu_t = b.copy().astype(np.float64)
    Sigma_t = np.eye(N)*1.0

    for t in range(1, T):
        x_t = price_relative_vectors[t-1]
        x_t_mean = np.dot(mu_t, x_t)

        # x_t_tilde not strictly needed in this simplified version.
        # We'll keep it minimal.
        denominator = x_t @ (Sigma_t @ x_t)
        if denominator <= 0:
            lambda_t = 0
        else:
            lambda_t = max(0, (x_t_mean - epsilon)/denominator)

        mu_t -= lambda_t * Sigma_t @ x_t
        # Update Sigma
        Sigma_t_inv = np.linalg.inv(Sigma_t + np.eye(N)*1e-12)
        Sigma_t_inv += 2 * lambda_t * theta * np.outer(x_t, x_t)
        Sigma_t = np.linalg.inv(Sigma_t_inv)

        mu_t = utilities.project_to_simplex(mu_t)
        b_n[t] = mu_t

    return b_n

# Strategy 12: OLMAR
def olmar(b, price_relative_vectors, window_size=10, epsilon=0.5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        # Weighted moving average
        if t < window_size:
            ma_t = np.mean(price_relative_vectors[:t], axis=0)
        else:
            ma_t = np.mean(price_relative_vectors[t-window_size : t], axis=0)

        # Basic prediction x_t_tilde
        # Just a placeholder. The real OLMAR logic is more nuanced.
        x_t_tilde = ma_t  # very simplified

        b_t = b_n[t-1]
        x_t_tilde_mean = np.dot(b_t, x_t_tilde)
        if x_t_tilde_mean < epsilon:
            tau = (epsilon - x_t_tilde_mean)/ (np.dot(x_t_tilde, x_t_tilde)+1e-15)
            b_t1 = b_t + tau*(x_t_tilde - b_t)
            b_t1 = utilities.project_to_simplex(b_t1)
        else:
            b_t1 = b_t
        b_n[t] = b_t1

    return b_n

# Strategy 13: RMR

def rmr(b, price_relative_vectors, window_size=10, epsilon=0.8):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()

    for t in range(1, T):
        if t < window_size:
            window_data = price_relative_vectors[:t]
        else:
            window_data = price_relative_vectors[t-window_size : t]

        mu_t_plus_1 = calculate_l1_median(np.array(window_data, dtype=np.float64))
        # Predict next
        x_t_tilde = mu_t_plus_1 / (price_relative_vectors[t-1]+1e-15)

        b_t = b_n[t-1]
        x_t_tilde_mean = np.dot(b_t, x_t_tilde)
        if x_t_tilde_mean < epsilon:
            tau = (epsilon - x_t_tilde_mean)/ (np.dot(x_t_tilde, x_t_tilde)+1e-15)
            b_t1 = b_t + tau*(x_t_tilde - b_t)
            b_t1 = utilities.project_to_simplex(b_t1)
        else:
            b_t1 = b_t
        b_n[t] = b_t1

    return b_n
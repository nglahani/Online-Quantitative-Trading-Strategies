##############################################################################
# PATTERN-MATCHING FRAMEWORK
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from Strategies.helper import *


def histogram_based_selection(price_relative_vectors, w=5, bins=[(0, 0.5), (0.5, 1), (1, 1.5)]):
    n = len(price_relative_vectors)
    if n < w:
        return []
    # Compute overall daily means and then rolling means using convolution
    daily_means = price_relative_vectors.mean(axis=1)
    kernel = np.ones(w) / w
    rolling_means = np.convolve(daily_means, kernel, mode='valid')
    latest_mean = rolling_means[-1]

    latest_bin = -1
    for idx, (low, high) in enumerate(bins):
        if low <= latest_mean < high:
            latest_bin = idx
            break

    C = []
    for i, hist_mean in enumerate(rolling_means[:-1]):  # Exclude the latest window
        hist_bin = -1
        for idx, (low, high) in enumerate(bins):
            if low <= hist_mean < high:
                hist_bin = idx
                break
        if hist_bin == latest_bin:
            C.append(i + w)  # Map window index to corresponding day index
    return C


def kernel_based_selection(price_relative_vectors, w=5, threshold=0.2):
    n = len(price_relative_vectors)
    if n < w:
        return []
    latest_window = np.mean(price_relative_vectors[-w:], axis=0)
    # Use sliding_window_view to get all windows of length w along axis 0
    rolling_windows = np.lib.stride_tricks.sliding_window_view(price_relative_vectors, window_shape=w, axis=0)
    rolling_means = rolling_windows.mean(axis=1)  # shape: (n - w + 1, N)
    distances = np.linalg.norm(rolling_means - latest_window, axis=1)
    valid = np.where(distances <= threshold / np.sqrt(w))[0]
    # Map valid window indices to day indices (window index 0 corresponds to day w)
    C = [int(i) + w for i in valid]
    return C


def nearest_neighbor_selection(price_relative_vectors, w=5, num_neighbors=3):
    n = len(price_relative_vectors)
    if n < w:
        return []
    latest_window = np.mean(price_relative_vectors[-w:], axis=0)
    rolling_windows = np.lib.stride_tricks.sliding_window_view(price_relative_vectors, window_shape=w, axis=0)
    rolling_means = rolling_windows.mean(axis=1)
    distances = np.linalg.norm(rolling_means - latest_window, axis=1)
    sorted_idx = np.argsort(distances)
    C = [int(idx) + w for idx in sorted_idx[:num_neighbors]]
    return C


def correlation_based_selection(price_relative_vectors, w=5, rho=0.7):
    n = len(price_relative_vectors)
    if n < w:
        return []
    latest_window = np.mean(price_relative_vectors[-w:], axis=0)
    latest_mean = latest_window.mean()
    latest_std = latest_window.std() + 1e-15  # Avoid division by zero
    rolling_windows = np.lib.stride_tricks.sliding_window_view(price_relative_vectors, window_shape=w, axis=0)
    rolling_means = rolling_windows.mean(axis=1)
    rolling_mean_means = rolling_means.mean(axis=1)
    rolling_std = rolling_means.std(axis=1) + 1e-15
    numerator = np.sum((rolling_means - rolling_means.mean(axis=1, keepdims=True)) * (latest_window - latest_mean), axis=1)
    correlation = numerator / (rolling_std * latest_std)
    valid = np.where(correlation >= rho)[0]
    C = [int(i) + w for i in valid]
    return C


def log_optimal_portfolio(C, price_relative_vectors):
    if not C:
        m = price_relative_vectors.shape[1]
        return np.ones(m) / m
    X_C = price_relative_vectors[C]

    def objective(b):
        return -np.sum(np.log(np.dot(X_C, b) + 1e-15)) / len(C)

    def grad(b):
        y = np.dot(X_C, b) + 1e-15
        return - (X_C.T @ (1 / y)) / len(C)

    cons = ({'type': 'eq', 'fun': lambda b: np.sum(b) - 1})
    bounds = [(0, 1)] * price_relative_vectors.shape[1]
    b_init = np.ones(price_relative_vectors.shape[1]) / price_relative_vectors.shape[1]
    result = minimize(objective, b_init, method='SLSQP', jac=grad, bounds=bounds, constraints=cons)
    return result.x


def semi_log_optimal_portfolio(C, price_relative_vectors):
    if not C:
        m = price_relative_vectors.shape[1]
        return np.ones(m) / m
    X_C = price_relative_vectors[C]

    def f_z(z):
        return z - 0.5 * (z - 1) ** 2

    def objective(b):
        return -np.sum(f_z(np.dot(X_C, b))) / len(C)

    def grad(b):
        z = np.dot(X_C, b)
        return - (X_C.T @ (2 - z)) / len(C)

    cons = ({'type': 'eq', 'fun': lambda b: np.sum(b) - 1})
    bounds = [(0, 1)] * price_relative_vectors.shape[1]
    b_init = np.ones(price_relative_vectors.shape[1]) / price_relative_vectors.shape[1]
    result = minimize(objective, b_init, method='SLSQP', jac=grad, bounds=bounds, constraints=cons)
    return result.x


def markowitz_portfolio(C, price_relative_vectors, lambda_=0.5):
    if not C:
        m = price_relative_vectors.shape[1]
        return np.ones(m) / m
    X_C = price_relative_vectors[C]
    mean_returns = np.mean(X_C, axis=0)
    cov_matrix = np.cov(X_C, rowvar=False)

    def objective(b):
        return -np.dot(b, mean_returns) + lambda_ * b.T.dot(cov_matrix).dot(b)

    def grad(b):
        return -mean_returns + 2 * lambda_ * cov_matrix.dot(b)

    cons = ({'type': 'eq', 'fun': lambda b: np.sum(b) - 1})
    bounds = [(0, 1)] * price_relative_vectors.shape[1]
    b_init = np.ones(price_relative_vectors.shape[1]) / price_relative_vectors.shape[1]
    result = minimize(objective, b_init, method='SLSQP', jac=grad, bounds=bounds, constraints=cons)
    return result.x


def pattern_matching_portfolio_master(b, price_relative_vectors, methods, w=5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        if t >= w:
            C_t = methods['sample_selection'](price_relative_vectors[:t], w=w)
        else:
            C_t = []
        b_t = methods['portfolio_optimization'](C_t, price_relative_vectors[:t])
        b_n[t] = b_t
    return b_n
##############################################################################
# PATTERN-MATCHING FRAMEWORK
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from Strategies.helper import *

def histogram_based_selection(price_relative_vectors, w=5, bins=[(0,0.5),(0.5,1),(1,1.5)]):
    n = len(price_relative_vectors)
    if n < w:
        return []
    rolling_means = np.array([np.mean(price_relative_vectors[i-w:i], axis=0) for i in range(w,n+1)])
    latest_mean = np.mean(rolling_means[-1])
    latest_bin = next((idx for idx,(low,high) in enumerate(bins) if low <= latest_mean < high), -1)

    C = []
    for i in range(len(rolling_means)-1):
        hist_mean = np.mean(rolling_means[i])
        hist_bin = next((idx for idx,(low,high) in enumerate(bins) if low <= hist_mean < high), -1)
        if hist_bin == latest_bin:
            C.append(i + w)
    return C

def kernel_based_selection(price_relative_vectors, w=5, threshold=0.2):
    n = len(price_relative_vectors)
    if n < w:
        return []
    C = []
    latest_window = np.mean(price_relative_vectors[-w:], axis=0)

    for i in range(w, n):
        historical_window = np.mean(price_relative_vectors[i-w:i], axis=0)
        distance = np.linalg.norm(latest_window - historical_window)
        if distance <= threshold / np.sqrt(w):
            C.append(i)
    return C

def nearest_neighbor_selection(price_relative_vectors, w=5, num_neighbors=3):
    n = len(price_relative_vectors)
    if n < w:
        return []
    latest_window = np.mean(price_relative_vectors[-w:], axis=0)
    distances = []
    for i in range(w, n):
        historical_window = np.mean(price_relative_vectors[i-w:i], axis=0)
        dist = np.linalg.norm(latest_window - historical_window)
        distances.append((i, dist))
    distances.sort(key=lambda x: x[1])
    C = [idx for idx,_ in distances[:num_neighbors]]
    return C

def correlation_based_selection(price_relative_vectors, w=5, rho=0.7):
    n = len(price_relative_vectors)
    if n < w:
        return []
    C = []
    latest_window = np.mean(price_relative_vectors[-w:], axis=0)
    for i in range(w,n):
        historical_window = np.mean(price_relative_vectors[i-w:i], axis=0)
        corr = np.corrcoef(latest_window, historical_window)[0,1]
        if corr >= rho:
            C.append(i)
    return C

def log_optimal_portfolio(C, price_relative_vectors):
    if not C:
        m = price_relative_vectors.shape[1]
        return np.ones(m)/m
    X_C = price_relative_vectors[C]

    def objective(b):
        return -np.sum(np.log(np.dot(X_C,b)+1e-15))/len(C)

    cons = ({'type':'eq','fun':lambda b: np.sum(b)-1})
    bounds = [(0,1)]*price_relative_vectors.shape[1]
    b_init = np.ones(price_relative_vectors.shape[1])/price_relative_vectors.shape[1]
    result = minimize(objective, b_init, method='SLSQP', bounds=bounds, constraints=cons)
    return result.x

def semi_log_optimal_portfolio(C, price_relative_vectors):
    if not C:
        m = price_relative_vectors.shape[1]
        return np.ones(m)/m
    X_C = price_relative_vectors[C]

    def f_z(z):
        return z - 0.5*(z-1)**2

    def objective(b):
        return -np.sum(f_z(np.dot(X_C,b)))/len(C)

    cons = ({'type':'eq','fun':lambda b: np.sum(b)-1})
    bounds = [(0,1)] * price_relative_vectors.shape[1]
    b_init = np.ones(price_relative_vectors.shape[1])/price_relative_vectors.shape[1]
    result = minimize(objective, b_init, method='SLSQP', bounds=bounds, constraints=cons)
    return result.x

def markowitz_portfolio(C, price_relative_vectors, lambda_=0.5):
    if not C:
        m = price_relative_vectors.shape[1]
        return np.ones(m)/m
    X_C = price_relative_vectors[C]
    mean_returns = np.mean(X_C, axis=0)
    cov_matrix = np.cov(X_C, rowvar=False)

    def objective(b):
        return -np.dot(b, mean_returns) + lambda_* b.T.dot(cov_matrix).dot(b)

    cons = ({'type':'eq','fun':lambda b: np.sum(b)-1})
    bounds = [(0,1)] * price_relative_vectors.shape[1]
    b_init = np.ones(price_relative_vectors.shape[1])/price_relative_vectors.shape[1]
    result = minimize(objective, b_init, method='SLSQP', bounds=bounds, constraints=cons)
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

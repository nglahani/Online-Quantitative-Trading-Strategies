import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

def project_to_simplex(v):
    """ Project the vector v onto the probability simplex (sum to 1 and all entries >= 0). """
    n = len(v)
    # Sort v in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u > cssv / np.arange(1, n+1))[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0)

def calculate_l1_median(data, max_iter=100, tol=1e-5):
    mu = np.mean(data, axis=0)  # Initial guess
    for _ in range(max_iter):
        distances = np.linalg.norm(data - mu, axis=1)
        distances[distances < 1e-10] = 1e-10  # Avoid divide-by-zero
        weights = 1.0 / distances[:, np.newaxis]
        mu_new = np.sum(weights * data, axis=0) / np.sum(weights, axis=0)
        if np.linalg.norm(mu_new - mu) < tol:
            break
        mu = mu_new
    return mu

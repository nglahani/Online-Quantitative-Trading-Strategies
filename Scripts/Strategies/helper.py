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

def calculate_l1_median(data):
    def objective_function(mu):
        return np.sum(np.linalg.norm(data - mu, axis=1))
    initial_guess = np.mean(data, axis=0)
    result = minimize(objective_function, initial_guess)
    return result.x
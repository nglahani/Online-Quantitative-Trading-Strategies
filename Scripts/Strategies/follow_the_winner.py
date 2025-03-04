##############################################################################
# FOLLOW-THE-WINNER ALGORITHMS (Optimized)
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from Strategies.helper import *


# Strategy 4: Universal Portfolios (Approximation)
def universal_portfolios(b, price_relative_vectors, num_portfolios=500):
    """
    Cover's Universal Portfolios, approximated by sampling 'num_portfolios' random points
    on the simplex. We track the wealth of each sampled portfolio over time and then 
    blend them by their (normalized) wealth to form a final "universal" portfolio each step.
    """
    T, N = price_relative_vectors.shape
    # Sample many random portfolios on the simplex
    portfolios = np.random.dirichlet(np.ones(N), size=num_portfolios)  # shape (num_portfolios, N)
    
    # Each portfolio starts with wealth = 1.0
    wealth = np.ones(num_portfolios)
    b_n = np.zeros((T, N))
    
    for t in range(T):
        # Compute weighted average of sampled portfolios (weights = current wealth)
        w_t = np.average(portfolios, axis=0, weights=wealth)
        w_t /= w_t.sum()
        b_n[t] = w_t

        # Update wealth based on the observed price relatives x_t
        x_t = price_relative_vectors[t]
        portfolio_returns = portfolios.dot(x_t)
        wealth *= portfolio_returns

    return b_n


# Strategy 5: Exponential Gradient
def exponential_gradient(b, price_relative_vectors, learning_rate=0.1):
    """
    Implements the exponential gradient update in a vectorized manner.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        # Use the previous period's return to update portfolio
        x_t = price_relative_vectors[t-1]
        portfolio_return = np.dot(b_n[t-1], x_t)
        # Vectorized update factor for each asset
        update_factor = learning_rate * (x_t / (portfolio_return + 1e-15) - 1) + 1
        new_b = b_n[t-1] * update_factor
        new_b /= np.sum(new_b)
        b_n[t] = new_b

    return b_n


# Strategy 6: Follow-The-Leader
def follow_the_leader(b, price_relative_vectors, gamma=0.5):
    """
    Uses cumulative log returns to identify the best-performing portfolio 
    and then blends it with the previous portfolio using gamma-smoothing.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    # Precompute cumulative log returns for efficiency
    log_returns = np.log(price_relative_vectors + 1e-15)
    cum_log_returns = np.cumsum(log_returns, axis=0)

    for t in range(1, T):
        # Use cumulative log returns up to time t-1 to form the leader portfolio
        b_star = np.exp(cum_log_returns[t-1])
        b_star /= np.sum(b_star)
        # Blend with the previous portfolio using smoothing parameter gamma
        b_n[t] = (1 - gamma) * b_star + gamma * b_n[t-1]
        b_n[t] /= np.sum(b_n[t])
    return b_n


# Strategy 7: Follow the Regularized Leader (simplified ONS style)
def follow_the_regularized_leader(b, price_relative_vectors, beta=0.1, delta=0.5):
    """
    Implements a simplified ONS-style Follow-The-Regularized Leader.
    Here we accumulate the gradient incrementally and update A_t accordingly.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    A_t = np.eye(N)
    grad_sum = np.zeros(N)

    for t in range(1, T):
        # Get the price relative at previous time step
        x_t = price_relative_vectors[t-1]
        port_return = np.dot(b_n[t-1], x_t)
        
        # Update the matrix A_t with a ridge term for numerical stability
        A_t += np.outer(x_t, x_t) / (port_return + 1e-15)**2 + np.eye(N) * 1e-3

        # Incrementally update the gradient sum
        grad_t = x_t / (port_return + 1e-15)
        grad_sum += grad_t

        p_t = (1 + (1 / beta)) * grad_sum
        new_b = np.linalg.inv(A_t).dot(p_t) * delta
        new_b = project_to_simplex(new_b)
        b_n[t] = new_b

    return b_n


# Strategy 8: Aggregation-Based (Simple)
def aggregation_based_simple(b, price_relative_vectors, learning_rate=0.5):
    """
    Aggregates a set of base portfolios using their cumulative performance.
    This version updates the portfolio performance iteratively to avoid
    recomputing the full product over time at each iteration.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    # Generate base portfolios (one per asset)
    base_portfolios = np.random.dirichlet(np.ones(N), N)
    prior_weights = np.ones(N) / N

    # Initialize cumulative performance for each base portfolio
    portfolio_performance = np.ones(N)
    for t in range(T):
        if t > 0:
            # Update cumulative performance iteratively
            x_t = price_relative_vectors[t-1]
            portfolio_performance *= base_portfolios.dot(x_t)
        adjusted_weights = prior_weights * (portfolio_performance ** learning_rate)
        adjusted_weights /= np.sum(adjusted_weights)
        b_n[t] = adjusted_weights.dot(base_portfolios)

    return b_n
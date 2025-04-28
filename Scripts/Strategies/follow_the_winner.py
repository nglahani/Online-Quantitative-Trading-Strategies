##############################################################################
# FOLLOW-THE-WINNER ALGORITHMS (Optimized)
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from Strategies.helper import *

# Strategy 4: Universal Portfolios (Approximation)
def universal_portfolios(b, price_relative_vectors, num_portfolios=3, tau=.4):
    """
    Cover's Universal Portfolios, approximated by sampling 'num_portfolios' random points
    on the simplex. We track the wealth of each sampled portfolio over time and then 
    blend them by their (normalized) wealth to form a final "universal" portfolio each step.
    
    Hyperparameters:
    - num_portfolios: number of random portfolios to sample.
    - tau: wealth weighting temperature. When tau=1, weights are the raw wealth.
           When tau > 1, higher performing portfolios get more emphasis.
           When tau < 1, the influence of wealth differences is softened.
           
    Parameters:
    - price_relative_vectors: A 2D NumPy array of shape (T, N) where T is the number of time periods
      and N is the number of assets.
    
    Returns:
    - b_n: A 2D NumPy array of shape (T, N) representing the blended portfolio at each period.
    """
    T, N = price_relative_vectors.shape
    
    # Sample many random portfolios on the simplex
    portfolios = np.random.dirichlet(np.ones(N), size=num_portfolios)  # shape (num_portfolios, N)
    
    # Each portfolio starts with wealth = 1.0
    wealth = np.ones(num_portfolios)
    b_n = np.zeros((T, N))
    
    for t in range(T):
        # Compute weighted average of sampled portfolios using wealth**tau as weights
        weights = wealth ** tau
        w_t = np.average(portfolios, axis=0, weights=weights)
        w_t /= w_t.sum()  # Ensure the portfolio sums to 1
        b_n[t] = w_t

        # Update wealth based on the observed price relatives x_t
        x_t = price_relative_vectors[t]
        portfolio_returns = portfolios.dot(x_t)
        wealth *= portfolio_returns

    return b_n


# Strategy 5: Exponential Gradient
def exponential_gradient(b, price_relative_vectors, learning_rate=0.1, smoothing=0.0):
    """
    Implements the exponential gradient update in a vectorized manner with a smoothing parameter.
    
    Parameters:
        b: Initial portfolio vector (numpy array).
        price_relative_vectors: 2D numpy array of price relatives (shape: T x N).
        learning_rate: Learning rate (η) for the exponential gradient update.
        smoothing: Smoothing parameter (α) in [0, 1]. 
                   With α = 1, the update is fully applied; with α < 1, the update is a convex
                   combination of the previous portfolio and the computed update.
                   
    Returns:
        b_n: 2D numpy array where each row is the portfolio vector at a given time step.
    """
    # Debug output to verify parameter values
    # print(f"[DEBUG] exponential_gradient called with learning_rate={learning_rate}, smoothing={smoothing}")
    # print(f"[DEBUG] price_relative_vectors shape: {price_relative_vectors.shape}")
    
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        x_t = price_relative_vectors[t-1]
        portfolio_return = np.dot(b_n[t-1], x_t)
        # Compute update factor using the exponential gradient formulation
        update_factor = learning_rate * (x_t / (portfolio_return + 1e-15) - 1) + 1
        
        # Compute the updated portfolio weights (before smoothing)
        computed_b = b_n[t-1] * update_factor
        computed_b /= np.sum(computed_b)
        
        # Apply smoothing: interpolate between the old portfolio and the computed update
        new_b = (1 - smoothing) * b_n[t-1] + smoothing * computed_b
        new_b /= np.sum(new_b)
        b_n[t] = new_b

    return b_n



# Strategy 6: Follow-The-Leader
def follow_the_leader(b_init, price_relative_vectors, gamma=1.0, alpha=2.0):
    """Implementation of Follow-the-Leader with stability improvements"""
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b_init.copy()
    
    for t in range(1, T):
        # Calculate historical returns with stability threshold
        historical_returns = np.maximum(np.cumprod(price_relative_vectors[:t], axis=0), 1e-10)
        
        # Calculate weights with power function and stability constant
        weights = np.power(historical_returns[-1], alpha) + 1e-10
        
        # Normalize with numeric stability
        sum_weights = np.sum(weights)
        if sum_weights > 1e-10:
            b_n[t] = weights / sum_weights
        else:
            b_n[t] = np.ones(N) / N  # Fallback to uniform allocation
            
        # Apply momentum factor
        if gamma != 1.0:
            b_n[t] = gamma * b_n[t] + (1 - gamma) * b_n[t-1]
            # Ensure final normalization
            sum_b = np.sum(b_n[t])
            if sum_b > 1e-10:
                b_n[t] /= sum_b
            else:
                b_n[t] = np.ones(N) / N
    
    return b_n

#Strategy 7: Follow the Regularized Leader
def follow_the_regularized_leader(b, price_relative_vectors, beta=0.09, delta=0.925, ridge_const=.0075):
    """
    Implements a simplified ONS-style Follow-The-Regularized Leader with an adjustable ridge regularization constant.
    
    Parameters:
    - b: Initial portfolio vector.
    - price_relative_vectors: 2D numpy array with price relative vectors.
    - beta: Hyperparameter controlling the weight on the gradient sum.
    - delta: Scaling factor for the new portfolio update.
    - ridge_const: Ridge regularization constant added to the update of A_t for numerical stability.
    
    Returns:
    - b_n: Updated portfolio trajectory over time.
    """
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    A_t = np.eye(N)
    grad_sum = np.zeros(N)

    for t in range(1, T):
        # Get the price relative at the previous time step
        x_t = price_relative_vectors[t-1]
        port_return = np.dot(b_n[t-1], x_t)
        
        # Update the matrix A_t with the ridge term for numerical stability
        A_t += np.outer(x_t, x_t) / (port_return + 1e-15)**2 + np.eye(N) * ridge_const

        # Incrementally update the gradient sum
        grad_t = x_t / (port_return + 1e-15)
        grad_sum += grad_t

        p_t = (1 + (1 / beta)) * grad_sum
        new_b = np.linalg.inv(A_t).dot(p_t) * delta
        new_b = project_to_simplex(new_b)
        b_n[t] = new_b

    return b_n

#Strategy 8: Aggregation-Based Simple
def aggregation_based_simple(b, price_relative_vectors, learning_rate=0.4, num_base_portfolios=1):
    """
    Aggregates a set of base portfolios using their cumulative performance.
    This version updates the portfolio performance iteratively to avoid
    recomputing the full product over time at each iteration.
    
    Parameters:
        b : numpy array
            The initial portfolio vector.
        price_relative_vectors : numpy array of shape (T, N)
            Matrix where each row represents a period and each column an asset.
        learning_rate : float, default 0.5
            Hyperparameter controlling the influence of cumulative performance.
        num_base_portfolios : int, optional
            The number of base portfolios to generate. If None, defaults to the number of assets (N).
    
    Returns:
        b_n : numpy array of shape (T, N)
            Matrix containing the aggregated portfolio weights over time.
    """
    T, N = price_relative_vectors.shape
    if num_base_portfolios is None:
        num_base_portfolios = N

    b_n = np.zeros((T, N))
    b_n[0] = b

    # Generate the base portfolios (tunable number) using a Dirichlet distribution.
    base_portfolios = np.random.dirichlet(np.ones(N), num_base_portfolios)
    prior_weights = np.ones(num_base_portfolios) / num_base_portfolios

    # Initialize cumulative performance for each base portfolio.
    portfolio_performance = np.ones(num_base_portfolios)
    for t in range(T):
        if t > 0:
            # Update cumulative performance iteratively.
            x_t = price_relative_vectors[t-1]
            portfolio_performance *= base_portfolios.dot(x_t)
        adjusted_weights = prior_weights * (portfolio_performance ** learning_rate)
        adjusted_weights /= np.sum(adjusted_weights)
        b_n[t] = adjusted_weights.dot(base_portfolios)

    return b_n

##############################################################################
# FOLLOW-THE-WINNER ALGORITHMS
##############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from .. import utilities

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
    
    # Each portfolio has a wealth, starting at 1.0
    wealth = np.ones(num_portfolios)
    
    # The universal mixture portfolio at each time step
    b_n = np.zeros((T, N))
    
    for t in range(T):
        # Weighted average of the sampled portfolios (weights = their current wealth)
        w_t = np.average(portfolios, axis=0, weights=wealth)
        # Project or re-normalize to ensure sum to 1 (should already be near 1).
        w_t /= w_t.sum()
        b_n[t] = w_t

        # Next, we observe the price relative x_t and update each portfolio's wealth
        x_t = price_relative_vectors[t]
        # The return for each sampled portfolio is dot(portfolio, x_t)
        portfolio_returns = portfolios.dot(x_t)
        wealth *= portfolio_returns

    return b_n

# Strategy 5: Exponential Gradient
def exponential_gradient(b, price_relative_vectors, learning_rate=0.1):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        x_t = price_relative_vectors[t-1]  # The return from previous step
        portfolio_return = np.dot(b_n[t-1], x_t)

        # Helmbold's exponential update
        new_b = b_n[t-1].copy()
        for i in range(N):
            new_b[i] *= (learning_rate * (x_t[i]/(portfolio_return+1e-15) - 1) + 1)

        new_b = new_b / np.sum(new_b)
        b_n[t] = new_b
    
    return b_n

# Strategy 6: Follow-The-Leader
def follow_the_leader(b, price_relative_vectors, gamma=0.5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    for t in range(1, T):
        # Up to time t-1
        cumulative_log_returns = np.sum(np.log(price_relative_vectors[:t]), axis=0)
        b_star = np.exp(cumulative_log_returns)
        b_star /= np.sum(b_star)

        b_n[t] = (1 - gamma) * b_star + gamma * b_n[t-1]
        b_n[t] /= np.sum(b_n[t])
    return b_n

# Strategy 7: Follow the Regularized Leader (simplified ONS style)
def follow_the_regularized_leader(b, price_relative_vectors, beta=0.1, delta=0.5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    A_t = np.eye(N)

    for t in range(1, T):
        # price relatives at time t-1
        x_t = price_relative_vectors[t-1]
        port_return = np.dot(b_n[t-1], x_t)

        A_t += np.outer(x_t, x_t)/(port_return+1e-15)**2 + np.eye(N)*1e-3  # small ridge

        # Summation-like term p_t
        # For a purely incremental approach, we'd store partial sums. Here is a naive approach:
        # sum over tau in [0..t-1]
        grad_sum = np.zeros(N)
        for tau in range(t):
            x_tau = price_relative_vectors[tau]
            b_tau = b_n[tau]
            grad_sum += x_tau / (np.dot(b_tau, x_tau) + 1e-15)
        p_t = (1 + (1 / beta)) * grad_sum

        new_b = np.linalg.inv(A_t).dot(p_t) * delta
        new_b = utilities.project_to_simplex(new_b)
        b_n[t] = new_b

    return b_n

# Strategy 8: Aggregation-Based (Simple)
def aggregation_based_simple(b, price_relative_vectors, learning_rate=0.5):
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b

    base_portfolios = np.random.dirichlet(np.ones(N), N)
    prior_weights = np.ones(N) / N

    for t in range(T):
        # Compute performance of each base portfolio up to time t
        if t == 0:
            portfolio_performance = np.ones(N)
        else:
            # product of returns up to time t-1
            # shape: base_portfolios is (N, N?), might need to do an iterative approach
            portfolio_performance = np.ones(len(base_portfolios))
            for i in range(t):
                x_i = price_relative_vectors[i]
                # each row in base_portfolios is a portfolio
                portfolio_performance *= base_portfolios.dot(x_i)

        adjusted_weights = prior_weights * (portfolio_performance ** learning_rate)
        adjusted_weights /= np.sum(adjusted_weights)

        b_n[t] = adjusted_weights.dot(base_portfolios)

    return b_n
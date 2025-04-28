#############################################################################
# META-LEARNING ALGORITHMS
#############################################################################

import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from Strategies.benchmarks import *
from Strategies.follow_the_loser import *
from Strategies.follow_the_winner import *
from Strategies.pattern_matching import *
from Strategies.helper import *

import numpy as np

import numpy as np

def aggregation_algorithm_generalized(b, price_relative_vectors, learning_rate=0.001, gamma=0.3):
    epsilon = 1e-15  # constant for numerical stability
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))
    b_n[0] = b
    expert_weights = np.ones(N) / N

    for t in range(T):
        expert_weights /= np.sum(expert_weights)
        b_n[t] = expert_weights
        if t < T - 1:
            x_t = price_relative_vectors[t]
            losses = -np.log(x_t + epsilon)
            expert_weights *= np.exp(-learning_rate * losses)
            # Mixing step: blend updated weights with a uniform distribution
            expert_weights = (1 - gamma) * expert_weights + gamma * (np.ones(N) / N)
    return b_n


def fast_universalization(b, price_relative_vectors, learning_rate=0.1, base_experts = [cwmr, follow_the_regularized_leader, pamr]):
    T, N = price_relative_vectors.shape
    M = len(base_experts)

    expert_weights = np.ones(M) / M
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()

    expert_portfolios = np.array([b.copy() for _ in range(M)])

    for t in range(T):
        meta_portfolio = np.dot(expert_weights, expert_portfolios)
        b_n[t] = meta_portfolio

        if t < T - 1:
            x_t = price_relative_vectors[t]
            # Vectorized dot product over experts: expert_portfolios has shape (M, N)
            expert_performance = np.dot(expert_portfolios, x_t)
            losses = -np.log(expert_performance + 1e-15)
            expert_weights *= np.exp(-learning_rate * losses)
            expert_weights /= np.sum(expert_weights)

            # Update each expert portfolio using its own strategy
            for i, expert in enumerate(base_experts):
                full_b = expert(expert_portfolios[i], price_relative_vectors[:t+1])
                expert_portfolios[i] = full_b[-1]

    return b_n

def ons_single_step(current_portfolio, x_t, A, eta=0.1):
    """Helper function for ONS updates with improved numeric stability"""
    port_return = np.dot(current_portfolio, x_t)
    # Add stability constant to prevent division by zero
    port_return = np.maximum(port_return, 1e-10)
    
    grad = -x_t / port_return
    A += np.outer(grad, grad)
    # Add stability constant to diagonal before inversion
    A_inv = np.linalg.inv(A + np.eye(len(x_t)) * 1e-8)
    b_next = current_portfolio - (1.0 / eta) * A_inv.dot(grad)
    b_next = project_to_simplex(b_next)
    return b_next, A


class ONSExpert:
    def __init__(self, b_init, start_time, N, delta=1e-2):
        self.b_current = b_init.copy()
        self.start_time = start_time
        self.N = N
        self.A = np.eye(N) * delta

    def update(self, t, price_relative_vectors, eta=0.1):
        """
        Update the expert from time t to t+1 using a single-step ONS logic.
        """
        x_t = price_relative_vectors[t]
        self.b_current, self.A = ons_single_step(self.b_current, x_t, self.A, eta=eta)

    def get_portfolio(self):
        return self.b_current


def meta_weighted_majority(expert_portfolios, meta_weights, x_t, learning_rate=0.5):
    """Weighted majority with numeric stability improvements"""
    M, N = expert_portfolios.shape
    # Add small constant for numeric stability
    expert_returns = np.maximum(np.einsum('mn,n->m', expert_portfolios, x_t), 1e-10)
    losses = -np.log(expert_returns)
    
    # Compute weights with numeric stability
    weights = meta_weights * np.exp(-learning_rate * losses)
    sum_weights = np.sum(weights)
    if sum_weights > 1e-10:
        new_weights = weights / sum_weights
    else:
        new_weights = np.ones_like(weights) / len(weights)
    return new_weights


def follow_the_leading_history(b_init, price_relative_vectors, eta=0.4, learning_rate=0.05, drop_threshold=0.5):
    """
    Spawns a new ONS expert at each time step t, uses Weighted Majority to 
    combine them, and drops underperforming experts. Uses parallel processing
    for expert updates.
    """
    from joblib import Parallel, delayed
    
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))

    experts = []
    meta_weights = np.array([])

    def update_expert(expert, t, data, eta):
        expert.update(t, data, eta=eta)
        return expert

    for t in range(T):
        # Spawn a new expert at time t
        new_expert = ONSExpert(b_init, start_time=t, N=N, delta=1e-2)
        experts.append(new_expert)

        # Expand meta_weights to include the new expert
        if len(meta_weights) == 0:
            meta_weights = np.array([1.0])
        else:
            meta_weights = np.append(meta_weights, [1.0])
            meta_weights /= np.sum(meta_weights)

        # Combine the current portfolios of all experts
        expert_portfolios = np.array([exp.get_portfolio() for exp in experts])
        meta_portfolio = np.dot(meta_weights, expert_portfolios)
        meta_portfolio /= np.sum(meta_portfolio)
        b_n[t] = meta_portfolio

        # After we've decided on b_n[t], we see the actual price relative x_t and update
        if t < T - 1:
            x_t = price_relative_vectors[t]
            meta_weights = meta_weighted_majority(expert_portfolios, meta_weights, x_t, learning_rate)
            
            # Parallel update of experts
            updated_experts = Parallel(n_jobs=-1)(
                delayed(update_expert)(
                    expert, 
                    t, 
                    price_relative_vectors, 
                    eta
                ) for expert in experts
            )
            experts = updated_experts
            
            # Drop experts below threshold
            active_idxs = np.where(meta_weights >= drop_threshold)[0]
            if len(active_idxs) == 0 and len(meta_weights) > 0:
                active_idxs = np.array([np.argmax(meta_weights)])
            experts = [experts[i] for i in active_idxs]
            meta_weights = meta_weights[active_idxs]
            if meta_weights.sum() > 0:
                meta_weights /= meta_weights.sum()
            else:
                meta_weights = np.ones(len(experts)) / len(experts)

    return b_n


def online_gradient_update_meta(b, price_relative_vectors, learning_rate=0.01, base_experts=[cwmr, follow_the_regularized_leader, pamr]):
    """Add numeric stability to meta updates"""
    from joblib import Parallel, delayed
    
    T, N = price_relative_vectors.shape
    M = len(base_experts)
    expert_weights = np.ones(M) / M
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    expert_portfolios = np.array([b.copy() for _ in range(M)])
    
    def update_expert_portfolio(expert, portfolio, data):
        full_b = expert(portfolio, data)
        return full_b[-1]
    
    for t in range(T):
        meta_portfolio = np.dot(expert_weights, expert_portfolios)
        # Ensure portfolio sums to 1
        sum_portfolio = np.sum(meta_portfolio)
        if sum_portfolio > 1e-10:
            meta_portfolio /= sum_portfolio
        else:
            meta_portfolio = np.ones(N) / N
        b_n[t] = meta_portfolio
        
        if t < T - 1:
            x_t = price_relative_vectors[t]
            # Add stability constant for returns calculation
            expert_returns = np.maximum(np.dot(expert_portfolios, x_t), 1e-10)
            losses = -np.log(expert_returns)
            
            # Update weights with stability check
            expert_weights = expert_weights * np.exp(-learning_rate * losses)
            sum_weights = np.sum(expert_weights)
            if sum_weights > 1e-10:
                expert_weights /= sum_weights
            else:
                expert_weights = np.ones(M) / M
            
            # Parallel update of expert portfolios
            updated_portfolios = Parallel(n_jobs=-1)(
                delayed(update_expert_portfolio)(
                    expert, 
                    expert_portfolios[i], 
                    price_relative_vectors[:t+1]
                ) for i, expert in enumerate(base_experts)
            )
            expert_portfolios = np.array(updated_portfolios)
    
    return b_n

def online_newton_update_meta(b, price_relative_vectors, learning_rate=0.01, base_experts=[cwmr, follow_the_regularized_leader, pamr]):
    """Add numeric stability to Newton updates"""
    from joblib import Parallel, delayed
    
    delta = 1e-2  # Constant for numeric stability
    
    T, N = price_relative_vectors.shape
    M = len(base_experts)
    w_experts = np.ones(M) / M
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    expert_portfolios = np.array([b.copy() for _ in range(M)])
    A_inv = np.eye(M) / delta
    
    def update_expert_portfolio(expert, portfolio, data):
        full_b = expert(portfolio, data)
        return full_b[-1]

    for t in range(T):
        # Compute aggregated portfolio with stability check
        aggregated_portfolio = np.dot(w_experts, expert_portfolios)
        sum_portfolio = np.sum(aggregated_portfolio)
        if sum_portfolio > 1e-10:
            aggregated_portfolio /= sum_portfolio
        else:
            aggregated_portfolio = np.ones(N) / N
        b_n[t] = aggregated_portfolio

        if t < T - 1:
            x_t = price_relative_vectors[t]
            # Add stability constant for returns calculation
            expert_returns = np.maximum(np.dot(expert_portfolios, x_t), 1e-10)
            losses = -np.log(expert_returns)

            # Update A_inv with stability checks
            u = losses.reshape(-1, 1)
            Au = A_inv @ u
            uAu = (u.T @ Au)[0, 0]
            # Add small constant to denominator
            denom = 1.0 + uAu + 1e-10
            A_inv = A_inv - (Au @ Au.T) / denom

            # Compute new weights with stability
            w_next = w_experts - (1.0 / learning_rate) * (A_inv @ losses)
            w_experts = project_to_simplex(w_next)

            # Parallel update of expert portfolios
            updated_portfolios = Parallel(n_jobs=-1)(
                delayed(update_expert_portfolio)(
                    expert, 
                    expert_portfolios[i], 
                    price_relative_vectors[:t+1]
                ) for i, expert in enumerate(base_experts)
            )
            expert_portfolios = np.array(updated_portfolios)

    return b_n



# FOLLOW-THE-LEADING-HISTORY IMPLEMENTATION

def ons_single_step(current_portfolio, x_t, A, eta=0.1):
    port_return = np.dot(current_portfolio, x_t)
    # Add stability constant to prevent division by zero
    port_return = np.maximum(port_return, 1e-10)
    
    grad = -x_t / port_return
    A += np.outer(grad, grad)
    # Add stability constant to diagonal before inversion
    A_inv = np.linalg.inv(A + np.eye(len(x_t)) * 1e-8)
    b_next = current_portfolio - (1.0 / eta) * A_inv.dot(grad)
    b_next = project_to_simplex(b_next)
    return b_next, A


class ONSExpert:
    def __init__(self, b_init, start_time, N, delta=1e-2):
        self.b_current = b_init.copy()
        self.start_time = start_time
        self.N = N
        self.A = np.eye(N) * delta

    def update(self, t, price_relative_vectors, eta=0.1):
        """
        Update the expert from time t to t+1 using a single-step ONS logic.
        """
        x_t = price_relative_vectors[t]
        self.b_current, self.A = ons_single_step(self.b_current, x_t, self.A, eta=eta)

    def get_portfolio(self):
        return self.b_current


def meta_weighted_majority(expert_portfolios, meta_weights, x_t, learning_rate=0.5):
    """Weighted majority with numeric stability improvements"""
    M, N = expert_portfolios.shape
    # Add small constant for numeric stability
    expert_returns = np.maximum(np.einsum('mn,n->m', expert_portfolios, x_t), 1e-10)
    losses = -np.log(expert_returns)
    
    # Compute weights with numeric stability
    weights = meta_weights * np.exp(-learning_rate * losses)
    sum_weights = np.sum(weights)
    if sum_weights > 1e-10:
        new_weights = weights / sum_weights
    else:
        new_weights = np.ones_like(weights) / len(weights)
    return new_weights


def follow_the_leading_history(b_init, price_relative_vectors, eta=0.4, learning_rate=0.05, drop_threshold=0.5):
    """
    Spawns a new ONS expert at each time step t, uses Weighted Majority to 
    combine them, and drops underperforming experts. Uses parallel processing
    for expert updates.
    """
    from joblib import Parallel, delayed
    
    T, N = price_relative_vectors.shape
    b_n = np.zeros((T, N))

    experts = []
    meta_weights = np.array([])

    def update_expert(expert, t, data, eta):
        expert.update(t, data, eta=eta)
        return expert

    for t in range(T):
        # Spawn a new expert at time t
        new_expert = ONSExpert(b_init, start_time=t, N=N, delta=1e-2)
        experts.append(new_expert)

        # Expand meta_weights to include the new expert
        if len(meta_weights) == 0:
            meta_weights = np.array([1.0])
        else:
            meta_weights = np.append(meta_weights, [1.0])
            meta_weights /= np.sum(meta_weights)

        # Combine the current portfolios of all experts
        expert_portfolios = np.array([exp.get_portfolio() for exp in experts])
        meta_portfolio = np.dot(meta_weights, expert_portfolios)
        meta_portfolio /= np.sum(meta_portfolio)
        b_n[t] = meta_portfolio

        # After we've decided on b_n[t], we see the actual price relative x_t and update
        if t < T - 1:
            x_t = price_relative_vectors[t]
            meta_weights = meta_weighted_majority(expert_portfolios, meta_weights, x_t, learning_rate)
            
            # Parallel update of experts
            updated_experts = Parallel(n_jobs=-1)(
                delayed(update_expert)(
                    expert, 
                    t, 
                    price_relative_vectors, 
                    eta
                ) for expert in experts
            )
            experts = updated_experts
            
            # Drop experts below threshold
            active_idxs = np.where(meta_weights >= drop_threshold)[0]
            if len(active_idxs) == 0 and len(meta_weights) > 0:
                active_idxs = np.array([np.argmax(meta_weights)])
            experts = [experts[i] for i in active_idxs]
            meta_weights = meta_weights[active_idxs]
            if meta_weights.sum() > 0:
                meta_weights /= meta_weights.sum()
            else:
                meta_weights = np.ones(len(experts)) / len(experts)

    return b_n


def online_gradient_update_meta(b, price_relative_vectors, learning_rate=0.01, base_experts=[cwmr, follow_the_regularized_leader, pamr]):
    """Add numeric stability to meta updates"""
    from joblib import Parallel, delayed
    
    T, N = price_relative_vectors.shape
    M = len(base_experts)
    expert_weights = np.ones(M) / M
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    expert_portfolios = np.array([b.copy() for _ in range(M)])
    
    def update_expert_portfolio(expert, portfolio, data):
        full_b = expert(portfolio, data)
        return full_b[-1]
    
    for t in range(T):
        meta_portfolio = np.dot(expert_weights, expert_portfolios)
        # Ensure portfolio sums to 1
        sum_portfolio = np.sum(meta_portfolio)
        if sum_portfolio > 1e-10:
            meta_portfolio /= sum_portfolio
        else:
            meta_portfolio = np.ones(N) / N
        b_n[t] = meta_portfolio
        
        if t < T - 1:
            x_t = price_relative_vectors[t]
            # Add stability constant for returns calculation
            expert_returns = np.maximum(np.dot(expert_portfolios, x_t), 1e-10)
            losses = -np.log(expert_returns)
            
            # Update weights with stability check
            expert_weights = expert_weights * np.exp(-learning_rate * losses)
            sum_weights = np.sum(expert_weights)
            if sum_weights > 1e-10:
                expert_weights /= sum_weights
            else:
                expert_weights = np.ones(M) / M
            
            # Parallel update of expert portfolios
            updated_portfolios = Parallel(n_jobs=-1)(
                delayed(update_expert_portfolio)(
                    expert, 
                    expert_portfolios[i], 
                    price_relative_vectors[:t+1]
                ) for i, expert in enumerate(base_experts)
            )
            expert_portfolios = np.array(updated_portfolios)
    
    return b_n

def online_newton_update_meta(b, price_relative_vectors, learning_rate=0.01, base_experts=[cwmr, follow_the_regularized_leader, pamr]):
    """Add numeric stability to Newton updates"""
    from joblib import Parallel, delayed
    
    delta = 1e-2  # Constant for numeric stability
    
    T, N = price_relative_vectors.shape
    M = len(base_experts)
    w_experts = np.ones(M) / M
    b_n = np.zeros((T, N))
    b_n[0] = b.copy()
    expert_portfolios = np.array([b.copy() for _ in range(M)])
    A_inv = np.eye(M) / delta
    
    def update_expert_portfolio(expert, portfolio, data):
        full_b = expert(portfolio, data)
        return full_b[-1]

    for t in range(T):
        # Compute aggregated portfolio with stability check
        aggregated_portfolio = np.dot(w_experts, expert_portfolios)
        sum_portfolio = np.sum(aggregated_portfolio)
        if sum_portfolio > 1e-10:
            aggregated_portfolio /= sum_portfolio
        else:
            aggregated_portfolio = np.ones(N) / N
        b_n[t] = aggregated_portfolio

        if t < T - 1:
            x_t = price_relative_vectors[t]
            # Add stability constant for returns calculation
            expert_returns = np.maximum(np.dot(expert_portfolios, x_t), 1e-10)
            losses = -np.log(expert_returns)

            # Update A_inv with stability checks
            u = losses.reshape(-1, 1)
            Au = A_inv @ u
            uAu = (u.T @ Au)[0, 0]
            # Add small constant to denominator
            denom = 1.0 + uAu + 1e-10
            A_inv = A_inv - (Au @ Au.T) / denom

            # Compute new weights with stability
            w_next = w_experts - (1.0 / learning_rate) * (A_inv @ losses)
            w_experts = project_to_simplex(w_next)

            # Parallel update of expert portfolios
            updated_portfolios = Parallel(n_jobs=-1)(
                delayed(update_expert_portfolio)(
                    expert, 
                    expert_portfolios[i], 
                    price_relative_vectors[:t+1]
                ) for i, expert in enumerate(base_experts)
            )
            expert_portfolios = np.array(updated_portfolios)

    return b_n
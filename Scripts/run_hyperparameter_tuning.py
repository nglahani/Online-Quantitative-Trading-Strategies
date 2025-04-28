"""
Main script for running hyperparameter tuning on portfolio optimization strategies.

This script provides a unified interface for tuning various strategies,
with support for walk-forward validation to prevent overfitting.
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
import importlib
import itertools
from tqdm import tqdm

from tuning_framework import StrategyTuner
from strategy_tuners import StrategyFactory
from utilities import *

# Import strategies
from Strategies.benchmarks import *
from Strategies.follow_the_loser import *
from Strategies.follow_the_winner import *
from Strategies.pattern_matching import *
from Strategies.meta_learning import *
from Strategies.pattern_matching import histogram_based_selection

default_sample_selection_function = histogram_based_selection


def get_strategy_func(strategy_name):
    """
    Get the strategy function based on its name.
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy
        
    Returns:
    --------
    callable
        Function implementing the strategy
    """
    # Dictionary mapping strategy names to their functions
    strategy_map = {
        # Follow-the-Winner strategies
        'follow_the_leader': follow_the_leader,
        'exponential_gradient': exponential_gradient,
        'follow_the_regularized_leader': follow_the_regularized_leader,
        'aggregation_based_simple': aggregation_based_simple,
        'universal_portfolios': universal_portfolios,
        
        # Follow-the-Loser strategies
        'anticor': anticor,
        'pamr': pamr,
        'cwmr': cwmr,
        'olmar': olmar,
        'rmr': rmr,
        
        # Pattern Matching strategies
        'pattern_matching': pattern_matching_portfolio_master,
        
        # Meta-Learning strategies
        'online_gradient_update_meta': online_gradient_update_meta,
        'online_newton_update_meta': online_newton_update_meta,
        'fast_universalization': fast_universalization,
        'follow_the_leading_history': follow_the_leading_history,
        'aggregation_algorithm_generalized': aggregation_algorithm_generalized
    }
    
    if strategy_name.lower() not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {', '.join(strategy_map.keys())}")
    
    return strategy_map[strategy_name.lower()]


def get_default_param_grid(strategy_name):
    """
    Get a default parameter grid for the specified strategy.
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy
        
    Returns:
    --------
    dict
        Dictionary of parameter names and lists of values to try
    """
    # Dictionary mapping strategy names to their default parameter grids
    default_grids = {
        # Follow-the-Winner strategies (expand parameters where momentum effects might be stronger)
        'follow_the_leader': {
            'gamma': [0.8, 0.9, 1.0, 1.1],  # Added higher value to test stronger trend following
            'alpha': [1.5, 2.0, 2.5, 3.0]    # Added higher alpha for more aggressive momentum
        },
        'exponential_gradient': {
            'learning_rate': [0.05, 0.1, 0.15, 0.2],  # Added higher learning rate
            'smoothing': [0.0, 0.1, 0.2]  # Added more smoothing options
        },
        'follow_the_regularized_leader': {
            'beta': [0.07, 0.09, 0.11, 0.13],        # Added higher regularization option
            'delta': [0.85, 0.9, 0.925, 0.95],       # Expanded range slightly
            'ridge_const': [0.005, 0.0075, 0.01, 0.015]  # Added higher ridge constant
        },
        'aggregation_based_simple': {
            'learning_rate': [0.3, 0.4, 0.5, 0.6],  # Added higher learning rate
            'num_base_portfolios': [1, 2, 3, 5]    # Added option for more portfolios
        },
        'universal_portfolios': {
            'num_portfolios': [3, 5, 7, 10],     # Added more portfolios option
            'tau': [0.3, 0.4, 0.5, 0.6]         # Added higher tau value
        },
        
        # Follow-the-Loser strategies (expand for market scenarios with higher volatility)
        'anticor': {
            'window_size': [2, 3, 4, 5],         # Added larger window
            'alpha': [1.8, 2.0, 2.2, 2.5],       # Added more aggressive reversion
            'corr_threshold': [0.4, 0.5, 0.6, 0.7]  # Added higher threshold
        },
        'pamr': {
            'epsilon': [0.7, 0.8, 0.9, 1.0],    # Added lower epsilon for more conservative updates
            'C': [5.0, 8.0, 10.0, 12.0]         # Added lower C for more stability
        },
        'cwmr': {
            'epsilon': [0.89, 0.91, 0.93, 0.95],      # Added higher epsilon
            'theta': [0.92, 0.94, 0.96, 0.98],        # Expanded range
            'eta': [0.93, 0.95, 0.97, 0.99]           # Added higher eta
        },
        'olmar': {
            'window_size': [2, 3, 4],              # Added larger window
            'epsilon': [0.8, 0.9, 1.0, 1.1],       # Added lower epsilon
            'eta': [20, 25, 30, 35]                # Added higher eta
        },
        'rmr': {
            'window_size': [5, 6, 7, 8],           # Expanded range
            'epsilon': [0.8, 0.9, 1.0, 1.1],       # Added lower epsilon
            'eta': [15, 20, 25, 30]                # Balanced range
        },
        
        # Pattern Matching strategies
        'pattern_matching': {
            'w': [3, 4, 5, 6]  # Added larger window option
        },
        
        # Meta-Learning strategies with consistent expert combinations
        'online_gradient_update_meta': {
            'learning_rate': [0.005, 0.01, 0.015, 0.02],
            'base_experts': [
                # Core Combo: Best individual performers from research
                ['cwmr', 'pamr', 'follow_the_regularized_leader'],
                
                # Pure Mean Reversion: Test strong mean reversion hypothesis
                ['cwmr', 'pamr', 'rmr', 'olmar'],
                
                # Pure Momentum: Test trend-following hypothesis
                ['follow_the_regularized_leader', 'exponential_gradient', 'universal_portfolios'],
                
                # Adaptive Hybrid: Test market regime adaptation
                ['cwmr', 'follow_the_regularized_leader', 'exponential_gradient', 'universal_portfolios']
            ]
        },
        'online_newton_update_meta': {
            'learning_rate': [0.005, 0.01, 0.015, 0.02],
            'base_experts': [
                # Core Combo: Best individual performers from research
                ['cwmr', 'pamr', 'follow_the_regularized_leader'],
                
                # Pure Mean Reversion: Test strong mean reversion hypothesis
                ['cwmr', 'pamr', 'rmr', 'olmar'],
                
                # Pure Momentum: Test trend-following hypothesis
                ['follow_the_regularized_leader', 'exponential_gradient', 'universal_portfolios'],
                
                # Adaptive Hybrid: Test market regime adaptation
                ['cwmr', 'follow_the_regularized_leader', 'exponential_gradient', 'universal_portfolios']
            ]
        },
        'fast_universalization': {
            'learning_rate': [0.05, 0.08, 0.1, 0.12],
            'base_experts': [
                # Core Combo: Best individual performers from research
                ['cwmr', 'pamr', 'follow_the_regularized_leader'],
                
                # Pure Mean Reversion: Test strong mean reversion hypothesis
                ['cwmr', 'pamr', 'rmr', 'olmar'],
                
                # Pure Momentum: Test trend-following hypothesis
                ['follow_the_regularized_leader', 'exponential_gradient', 'universal_portfolios'],
                
                # Adaptive Hybrid: Test market regime adaptation
                ['cwmr', 'follow_the_regularized_leader', 'exponential_gradient', 'universal_portfolios']
            ]
        },
        'follow_the_leading_history': {
            'eta': [0.3, 0.35, 0.4, 0.45],          
            'learning_rate': [0.03, 0.05, 0.07, 0.1],
            'drop_threshold': [0.4, 0.45, 0.5, 0.55]   
        },
        'aggregation_algorithm_generalized': {
            'learning_rate': [0.0005, 0.001, 0.002, 0.003],
            'gamma': [0.2, 0.25, 0.3, 0.35]
        }
    }
    
    if strategy_name.lower() not in default_grids:
        raise ValueError(f"No default parameter grid for strategy: {strategy_name}")
    
    return default_grids[strategy_name.lower()]


def setup_pattern_matching_grid(param_grid=None):
    """Set up parameter grid for pattern matching strategies."""
    from Strategies.pattern_matching import (
        histogram_based_selection, kernel_based_selection, 
        nearest_neighbor_selection, correlation_based_selection,
        log_optimal_portfolio, semi_log_optimal_portfolio, markowitz_portfolio
    )

    param_combinations = []
    
    # Define focused parameter ranges based on research results
    # Previous research showed better performance with these specific windows
    window_sizes = [3, 4, 5, 6]  # Added 6 for longer-term patterns
    # Lambda values were most effective at higher ranges
    lambda_values = [0.6, 0.7, 0.8, 0.9]
    
    # 1. Histogram-based selection combinations (consistently best performing)
    for w in window_sizes:
        # With markowitz portfolio optimizer (highest Sharpe ratios)
        for lambda_ in lambda_values:
            param_combinations.append({
                'methods': {
                    'sample_selection': histogram_based_selection,
                    'portfolio_optimization': markowitz_portfolio
                },
                'w': w,
                'lambda_': lambda_
            })
        
        # Semi-log performed better than pure log-optimal
        param_combinations.append({
            'methods': {
                'sample_selection': histogram_based_selection,
                'portfolio_optimization': semi_log_optimal_portfolio
            },
            'w': w
        })
    
    # 2. Kernel-based selection combinations (good for volatile markets)
    for w in window_sizes[:3]:  # Shorter windows worked better for kernel
        for threshold in [0.1, 0.15, 0.2]:  # Added middle value for finer control
            # Focus on Markowitz and semi-log which performed better
            for lambda_ in lambda_values[-3:]:  # Higher lambda values only
                param_combinations.append({
                    'methods': {
                        'sample_selection': kernel_based_selection,
                        'portfolio_optimization': markowitz_portfolio
                    },
                    'w': w,
                    'threshold': threshold,
                    'lambda_': lambda_
                })
            
            # Semi-log showed good stability
            param_combinations.append({
                'methods': {
                    'sample_selection': kernel_based_selection,
                    'portfolio_optimization': semi_log_optimal_portfolio
                },
                'w': w,
                'threshold': threshold
            })
    
    # 3. Nearest-neighbor selection combinations (useful for trending markets)
    for w in window_sizes[:3]:  # Shorter windows more effective
        for num_neighbors in [3, 4, 5]:  # Added middle value
            # Focus on higher lambda values which showed better performance
            for lambda_ in lambda_values[-2:]:
                param_combinations.append({
                    'methods': {
                        'sample_selection': nearest_neighbor_selection,
                        'portfolio_optimization': markowitz_portfolio
                    },
                    'w': w,
                    'num_neighbors': num_neighbors,
                    'lambda_': lambda_
                })
                
            # Semi-log performed well with nearest neighbors
            param_combinations.append({
                'methods': {
                    'sample_selection': nearest_neighbor_selection,
                    'portfolio_optimization': semi_log_optimal_portfolio
                },
                'w': w,
                'num_neighbors': num_neighbors
            })
    
    # 4. Correlation-based selection combinations (useful as part of ensemble)
    for w in window_sizes[:3]:  # Shorter windows more effective
        for rho in [0.6, 0.65, 0.7]:  # Added middle value, higher correlations worked better
            # Only use higher lambda values based on performance
            for lambda_ in lambda_values[-2:]:
                param_combinations.append({
                    'methods': {
                        'sample_selection': correlation_based_selection,
                        'portfolio_optimization': markowitz_portfolio
                    },
                    'w': w,
                    'rho': rho,
                    'lambda_': lambda_
                })
                
            # Semi-log provided good stability with correlation
            param_combinations.append({
                'methods': {
                    'sample_selection': correlation_based_selection,
                    'portfolio_optimization': semi_log_optimal_portfolio
                },
                'w': w,
                'rho': rho
            })
    
    print(f"Created {len(param_combinations)} strategically focused parameter combinations for pattern matching")
    return param_combinations


def run_tuning(args):
    """Run hyperparameter tuning for the specified strategy."""
    # Load price relative vectors
    print(f"Loading price relative vectors from {args.data_file}")
    price_relative_df = pd.read_csv(args.data_file, index_col=0)
    
    # Get strategy function
    strategy_func = get_strategy_func(args.strategy)
    
    # Determine parameter grid
    if args.param_file:
        param_module = importlib.import_module(args.param_file.replace('.py', '').replace('/', '.').replace('\\', '.'))
        param_grid = getattr(param_module, 'param_grid')
    else:
        param_grid = get_default_param_grid(args.strategy)
        
        # Special handling for pattern matching
        if args.strategy.lower() == 'pattern_matching':
            param_grid = setup_pattern_matching_grid(param_grid)
    
    # Create tuner using the factory
    tuner = StrategyFactory.create_tuner(
        args.strategy,
        strategy_func,
        price_relative_df,
        use_walk_forward=args.walk_forward,
        validation_windows=args.val_windows,
        validation_window_size=args.val_size,
        parallel=not args.no_parallel,
        n_jobs=args.n_jobs
    )
    
    # Set parameter grid
    tuner.set_param_grid(param_grid)
    
    # Run tuning
    print(f"\nRunning hyperparameter tuning for {args.strategy}...")
    start_time = time.time()
    results_df, best_params = tuner.tune_strategy()
    elapsed_time = time.time() - start_time
    
    # For pattern matching, convert function objects to their names
    if args.strategy.lower() == 'pattern_matching':
        results_df['sample_selection'] = results_df['sample_selection'].apply(lambda x: x.__name__ if hasattr(x, '__name__') else str(x))
        results_df['portfolio_optimization'] = results_df['portfolio_optimization'].apply(lambda x: x.__name__ if hasattr(x, '__name__') else str(x))
    
    # Save results
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"{args.strategy.lower().replace(' ', '_')}_tuning_results.csv"
    
    output_path = os.path.join(args.output_dir, output_file)
    print(f"\nSaving results to {output_path}")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\nTuning completed in {elapsed_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    return results_df, best_params


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for portfolio optimization strategies')
    
    parser.add_argument('strategy', type=str, help='Name of the strategy to tune')
    parser.add_argument('--data-file', type=str, default='../Data/Price Relative Vectors/price_relative_vectors.csv',
                        help='Path to the price relative vectors CSV file')
    parser.add_argument('--param-file', type=str, help='Python file with parameter grid')
    parser.add_argument('--output-dir', type=str, default='../Data/Tuning Data',
                        help='Directory to save tuning results')
    parser.add_argument('--output-file', type=str, help='Output file name')
    
    # Walk-forward validation parameters
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward validation')
    parser.add_argument('--val-windows', type=int, default=5, 
                        help='Number of validation windows for walk-forward validation')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Size of validation window as fraction of data')
    
    # Parallel processing parameters
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run tuning
    run_tuning(args)


if __name__ == '__main__':
    main()

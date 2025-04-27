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
        # Follow-the-Winner strategies
        'follow_the_leader': {
            'gamma': [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            'alpha': [0.1, 0.5, 1.0, 2.0, 3.0]
        },
        'exponential_gradient': {
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            'smoothing': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        },
        'follow_the_regularized_leader': {
            'beta': [0.05, 0.07, 0.09, 0.1, 0.15],
            'delta': [0.8, 0.85, 0.9, 0.95, 0.99],
            'ridge_const': [0.005, 0.0075, 0.01, 0.015, 0.02]
        },
        'aggregation_based_simple': {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0],
            'num_base_portfolios': [1, 2, 3, 5, 10, 20, 50]
        },
        'universal_portfolios': {
            'num_portfolios': [5, 10, 20, 50, 100],
            'tau': [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
        },
        
        # Follow-the-Loser strategies
        'anticor': {
            'window_size': [2, 3, 5, 7, 10],
            'alpha': [0.5, 1.0, 1.5, 2.0, 2.5],
            'corr_threshold': [0.0, 0.1, 0.3, 0.5, 0.7]
        },
        'pamr': {
            'epsilon': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'C': [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        },
        'cwmr': {
            'epsilon': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'theta': [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            'eta': [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
        },
        'olmar': {
            'window_size': [2, 3, 5, 7, 10],
            'epsilon': [0.5, 0.75, 1.0, 1.25, 1.5],
            'eta': [5, 10, 15, 20, 25, 30]
        },
        'rmr': {
            'window_size': [2, 3, 5, 7, 10],
            'epsilon': [0.5, 0.75, 1.0, 1.25, 1.5],
            'eta': [5, 10, 15, 20, 25, 30]
        },
        
        # Pattern Matching strategies (simplified)
        'pattern_matching': {
            'w': [3, 5, 7],  # Window size
        },
        
        # Meta-Learning strategies (minimal configuration for proof of concept)
        'online_gradient_update_meta': {
            'learning_rate': [0.1, 1.0],  # Reduced to just two learning rates
            'base_experts': [
                ['cwmr', 'pamr', 'follow_the_regularized_leader']  # Single basic expert combination
            ]
        },
        'online_newton_update_meta': {
            'learning_rate': [0.01, 0.1],  # Reduced to just two learning rates
            'base_experts': [
                ['cwmr', 'pamr', 'follow_the_regularized_leader']  # Single basic expert combination
            ]
        },
        'fast_universalization': {
            'learning_rate': [0.1, 1.0],  # Reduced to just two learning rates
            'base_experts': [
                ['cwmr', 'pamr', 'follow_the_regularized_leader']  # Single basic expert combination
            ]
        },
        'follow_the_leading_history': {
            'eta': [0.3],  # Single middle value
            'learning_rate': [0.1],  # Single reasonable learning rate
            'drop_threshold': [0.3]  # Single middle value
        },
        'aggregation_algorithm_generalized': {
            'learning_rate': [0.1, 0.5],  # Reduced to just two learning rates
            'gamma': [0.3, 0.7]  # Reduced to just two gamma values
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
    
    # Define all parameter ranges once
    window_sizes = [3, 5, 7, 10, 15]
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # 1. Histogram-based selection combinations
    for w in window_sizes:
        # With markowitz portfolio optimizer
        for lambda_ in lambda_values:
            param_combinations.append({
                'methods': {
                    'sample_selection': histogram_based_selection,
                    'portfolio_optimization': markowitz_portfolio
                },
                'w': w,
                'lambda_': lambda_
            })
        
        # With log optimal portfolio optimizer
        param_combinations.append({
            'methods': {
                'sample_selection': histogram_based_selection,
                'portfolio_optimization': log_optimal_portfolio
            },
            'w': w
        })
        
        # With semi-log optimal portfolio optimizer
        param_combinations.append({
            'methods': {
                'sample_selection': histogram_based_selection,
                'portfolio_optimization': semi_log_optimal_portfolio
            },
            'w': w
        })
    
    # 2. Kernel-based selection combinations
    for w in window_sizes:
        for threshold in [0.05, 0.1, 0.2, 0.3, 0.5]:
            # With markowitz portfolio optimizer
            for lambda_ in lambda_values:
                param_combinations.append({
                    'methods': {
                        'sample_selection': kernel_based_selection,
                        'portfolio_optimization': markowitz_portfolio
                    },
                    'w': w,
                    'threshold': threshold,
                    'lambda_': lambda_
                })
            
            # With log optimal portfolio optimizer
            param_combinations.append({
                'methods': {
                    'sample_selection': kernel_based_selection,
                    'portfolio_optimization': log_optimal_portfolio
                },
                'w': w,
                'threshold': threshold
            })
            
            # With semi-log optimal portfolio optimizer
            param_combinations.append({
                'methods': {
                    'sample_selection': kernel_based_selection,
                    'portfolio_optimization': semi_log_optimal_portfolio
                },
                'w': w,
                'threshold': threshold
            })
    
    # 3. Nearest-neighbor selection combinations
    for w in window_sizes:
        for num_neighbors in [3, 5, 7, 10, 15]:
            # With markowitz portfolio optimizer
            for lambda_ in lambda_values:
                param_combinations.append({
                    'methods': {
                        'sample_selection': nearest_neighbor_selection,
                        'portfolio_optimization': markowitz_portfolio
                    },
                    'w': w,
                    'num_neighbors': num_neighbors,
                    'lambda_': lambda_
                })
            
            # With log optimal portfolio optimizer
            param_combinations.append({
                'methods': {
                    'sample_selection': nearest_neighbor_selection,
                    'portfolio_optimization': log_optimal_portfolio
                },
                'w': w,
                'num_neighbors': num_neighbors
            })
            
            # With semi-log optimal portfolio optimizer
            param_combinations.append({
                'methods': {
                    'sample_selection': nearest_neighbor_selection,
                    'portfolio_optimization': semi_log_optimal_portfolio
                },
                'w': w,
                'num_neighbors': num_neighbors
            })
    
    # 4. Correlation-based selection combinations
    for w in window_sizes:
        for rho in [0.5, 0.6, 0.7, 0.8, 0.9]:
            # With markowitz portfolio optimizer
            for lambda_ in lambda_values:
                param_combinations.append({
                    'methods': {
                        'sample_selection': correlation_based_selection,
                        'portfolio_optimization': markowitz_portfolio
                    },
                    'w': w,
                    'rho': rho,
                    'lambda_': lambda_
                })
            
            # With log optimal portfolio optimizer
            param_combinations.append({
                'methods': {
                    'sample_selection': correlation_based_selection,
                    'portfolio_optimization': log_optimal_portfolio
                },
                'w': w,
                'rho': rho
            })
            
            # With semi-log optimal portfolio optimizer
            param_combinations.append({
                'methods': {
                    'sample_selection': correlation_based_selection,
                    'portfolio_optimization': semi_log_optimal_portfolio
                },
                'w': w,
                'rho': rho
            })
    
    print(f"Created {len(param_combinations)} comprehensive parameter combinations for pattern matching")
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

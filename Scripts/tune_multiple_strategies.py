"""
Example script for tuning multiple strategies sequentially or in parallel.

This script demonstrates how to use the tuning framework to tune multiple
strategies, with the option to use walk-forward validation.
"""

import pandas as pd
import numpy as np
import os
import time
import argparse
from joblib import Parallel, delayed
import multiprocessing

import time

from tuning_framework import StrategyTuner
from strategy_tuners import StrategyFactory
from run_hyperparameter_tuning import get_strategy_func, get_default_param_grid, setup_pattern_matching_grid

# Import strategies
from Strategies.benchmarks import *
from Strategies.follow_the_loser import *
from Strategies.follow_the_winner import *
from Strategies.pattern_matching import *
from Strategies.meta_learning import *

# Import utilities
from utilities import *


def tune_strategy(strategy_name, price_relative_df, use_walk_forward=False, output_dir='../Data/Tuning Data',
                  val_windows=5, val_size=0.2, parallel=True, n_jobs=None):
    """
    Tune a single strategy and return the best parameters.
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy to tune
        
    price_relative_df : pandas.DataFrame
        DataFrame containing price relative vectors
        
    use_walk_forward : bool, optional
        Whether to use walk-forward validation
        
    output_dir : str, optional
        Directory to save tuning results
        
    val_windows : int, optional
        Number of validation windows for walk-forward validation
        
    val_size : float, optional
        Size of each validation window as a fraction of the data
        
    parallel : bool, optional
        Whether to use parallel processing for tuning
        
    n_jobs : int, optional
        Number of parallel jobs for tuning. If None, uses CPU count - 2
        
    Returns:
    --------
    dict
        Dictionary containing best parameters and performance metrics
    """
    # Get strategy function
    strategy_func = get_strategy_func(strategy_name)
    
    # Get default parameter grid
    param_grid = get_default_param_grid(strategy_name)

    # Create tuner without the invalid output_path parameter
    tuner = StrategyFactory.create_tuner(
        strategy_name,
        strategy_func,
        price_relative_df,
        use_walk_forward=use_walk_forward,
        validation_windows=val_windows,
        validation_window_size=val_size,
        parallel=parallel,
        n_jobs=n_jobs
    )

    # Special handling for pattern matching
    if strategy_name.lower() == 'pattern_matching':
        param_grid = setup_pattern_matching_grid(param_grid)

    # Set parameter grid
    tuner.set_param_grid(param_grid)
    
    # Run tuning
    print(f"\nRunning hyperparameter tuning for {strategy_name}...")
    start_time = time.time()
    results_df, best_params = tuner.tune_strategy()
    elapsed_time = time.time() - start_time
    
    # Save results
    output_file = f"{strategy_name.lower().replace(' ', '_')}_tuning_results.csv"
    
    # Save results manually since we don't have output_path in tuner
    output_path = os.path.join(output_dir, output_file)
    # Sort results by average validation sharpe ratio if present before saving
    if 'avg_val_sharpe' in results_df.columns:
        results_df = results_df.sort_values('avg_val_sharpe', ascending=False)
    elif 'sharpe' in results_df.columns:
        results_df = results_df.sort_values('sharpe', ascending=False)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nTuning completed in {elapsed_time:.2f} seconds")
    print("\nBest parameter combination:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    # Add additional information to best parameters
    best_params['strategy_name'] = strategy_name
    best_params['runtime_seconds'] = elapsed_time
    
    return best_params


def run_multiple_tunings(args):
    """
    Run hyperparameter tuning for multiple strategies.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    # Determine which strategies to tune
    strategies_to_tune = []
    
    if args.all_strategies:
        # Tune all strategies
        strategies_to_tune = [
            # Follow-the-Winner strategies
            'follow_the_leader',
            'exponential_gradient',
            'follow_the_regularized_leader',
            'aggregation_based_simple',
            'universal_portfolios',
            
            # Follow-the-Loser strategies
            'anticor',
            'pamr',
            'cwmr',
            'olmar',
            'rmr',
            
            # Pattern Matching strategy
            'pattern_matching',
            
            # Meta-Learning strategies
            'online_gradient_update_meta',
            'online_newton_update_meta',
            'fast_universalization',
            'follow_the_leading_history',
            'aggregation_algorithm_generalized'
        ]
    elif args.category:
        # Tune strategies in the specified category
        category_map = {
            'ftw': [
                'follow_the_leader',
                'exponential_gradient',
                'follow_the_regularized_leader',
                'aggregation_based_simple',
                'universal_portfolios'
            ],
            'ftl': [
                'anticor',
                'pamr',
                'cwmr',
                'olmar',
                'rmr'
            ],
            'pm': [
                'pattern_matching'
            ],
            'meta': [
                'online_gradient_update_meta',
                'online_newton_update_meta',
                'fast_universalization',
                'follow_the_leading_history',
                'aggregation_algorithm_generalized'
            ]
        }
        
        if args.category.lower() not in category_map:
            raise ValueError(f"Unknown category: {args.category}. Available categories: {', '.join(category_map.keys())}")
        
        strategies_to_tune = category_map[args.category.lower()]
    else:
        # Tune specified strategies
        strategies_to_tune = args.strategies
    
    # Check that we have at least one strategy to tune
    if not strategies_to_tune:
        raise ValueError("No strategies specified for tuning")
    
    # Load price relative vectors
    print(f"Loading price relative vectors from {args.data_file}")
    price_relative_df = pd.read_csv(args.data_file, index_col=0)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run tuning for each strategy
    all_results = []
    
    if args.parallel_strategies and not args.no_parallel:
        # Tune strategies in parallel
        print(f"Tuning {len(strategies_to_tune)} strategies in parallel")
        
        # Determine number of jobs for strategy tuning (1 per strategy)
        n_strategy_jobs = min(len(strategies_to_tune), args.n_jobs if args.n_jobs else multiprocessing.cpu_count() - 1)
        
        # Determine number of jobs for parameter tuning (distribute remaining jobs)
        n_param_jobs = 1
        if args.n_jobs:
            n_param_jobs = max(1, args.n_jobs // n_strategy_jobs)
        else:
            n_param_jobs = max(1, (multiprocessing.cpu_count() - 1) // n_strategy_jobs)
        
        results = Parallel(n_jobs=n_strategy_jobs)(
            delayed(tune_strategy)(
                strategy_name=strategy,
                price_relative_df=price_relative_df,
                use_walk_forward=args.walk_forward,
                output_dir=args.output_dir,
                val_windows=args.val_windows,
                val_size=args.val_size,
                parallel=not args.no_parallel,
                n_jobs=n_param_jobs
            )
            for strategy in strategies_to_tune
        )
        
        all_results.extend(results)
    else:
        # Tune strategies sequentially
        for strategy in strategies_to_tune:
            result = tune_strategy(
                strategy_name=strategy,
                price_relative_df=price_relative_df,
                use_walk_forward=args.walk_forward,
                output_dir=args.output_dir,
                val_windows=args.val_windows,
                val_size=args.val_size,
                parallel=not args.no_parallel,
                n_jobs=args.n_jobs
            )
            all_results.append(result)
    
    return all_results


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for multiple portfolio optimization strategies')
    
    strategy_group = parser.add_mutually_exclusive_group(required=True)
    strategy_group.add_argument('--strategies', type=str, nargs='+', help='List of strategies to tune')
    strategy_group.add_argument('--category', type=str, choices=['ftw', 'ftl', 'pm', 'meta'],
                              help='Category of strategies to tune')
    strategy_group.add_argument('--all-strategies', action='store_true', help='Tune all available strategies')
    
    parser.add_argument('--data-file', type=str, default='../Data/Price Relative Vectors/price_relative_vectors.csv',
                        help='Path to the price relative vectors CSV file')
    parser.add_argument('--output-dir', type=str, default='../Data/Tuning Data',
                        help='Directory to save tuning results')
    
    # Walk-forward validation parameters
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward validation')
    parser.add_argument('--val-windows', type=int, default=5, 
                        help='Number of validation windows for walk-forward validation')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Size of validation window as fraction of data')
    
    # Parallel processing parameters
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--parallel-strategies', action='store_true', 
                        help='Tune strategies in parallel (in addition to parameter tuning)')
    parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run tuning
    run_multiple_tunings(args)

if __name__ == '__main__':
    main()

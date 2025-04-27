"""
Unified Hyperparameter Tuning Framework

This module implements a flexible, object-oriented framework for hyperparameter tuning
of portfolio optimization strategies with walk-forward validation support.
"""

import pandas as pd
import numpy as np
import itertools
import os
import time
from tqdm import tqdm

# Import strategies
from Strategies.benchmarks import *
from Strategies.follow_the_loser import *
from Strategies.follow_the_winner import *
from Strategies.pattern_matching import *
from Strategies.meta_learning import *
from utilities import *

class StrategyTuner:
    """
    Base class for tuning portfolio optimization strategies.
    
    This class provides the fundamental functionality for hyperparameter tuning
    with optional walk-forward validation.
    """
    
    def __init__(self, strategy_name, strategy_func, price_relative_df, use_walk_forward=False,
                 validation_windows=5, validation_window_size=0.2, parallel=True, n_jobs=None):
        """
        Initialize the tuner.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy being tuned
        
        strategy_func : callable
            Function implementing the strategy
            
        price_relative_df : pandas.DataFrame
            DataFrame containing price relative vectors
            
        output_path : str, optional
            Path where tuning results will be saved
            
        use_walk_forward : bool, optional
            Whether to use walk-forward validation
            
        validation_windows : int, optional
            Number of validation windows for walk-forward validation
            
        validation_window_size : float, optional
            Size of each validation window as a fraction of the total data
            
        parallel : bool, optional
            Whether to use parallel processing for tuning
            
        n_jobs : int, optional
            Number of parallel jobs for tuning. If None, uses CPU count - 2
        """
        self.strategy_name = strategy_name
        self.strategy_func = strategy_func
        self.price_relative_df = price_relative_df
        self.param_grid = None
        self.use_walk_forward = use_walk_forward
        self.validation_windows = validation_windows
        self.validation_window_size = validation_window_size
        self.parallel = parallel
        self.n_jobs = n_jobs

    def set_param_grid(self, param_grid):
        """
        Set the parameter grid for hyperparameter tuning.
        
        Parameters:
        -----------
        param_grid : dict or list
            Either:
            - Dictionary of parameter names and lists of values to try (standard grid)
            - List of parameter dictionaries (custom combinations for pattern matching)
        """
        self.param_grid = param_grid

    def evaluate_strategy(self, b, price_relative_vectors, **params):
        """
        Evaluate the strategy with given parameters and return performance metrics.
        
        Parameters:
        -----------
        b : numpy.ndarray
            Initial portfolio
            
        price_relative_vectors : numpy.ndarray
            Price relative vectors
            
        **params : dict
            Strategy hyperparameters
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        try:
            # Call the strategy function with the given parameters
            start_time = time.time()
            b_n = self.strategy_func(b, price_relative_vectors, **params)
            runtime = time.time() - start_time
            
            # Calculate performance metrics
            final_wealth = calculate_cumulative_wealth(b_n, price_relative_vectors)
            n_periods = len(price_relative_vectors)
            exp_growth = calculate_exponential_growth_rate(final_wealth, n_periods)
            
            # Calculate Sharpe ratio
            cum_wealth = calculate_cumulative_wealth_over_time(b_n, price_relative_vectors)
            daily_returns = compute_periodic_returns(cum_wealth)
            sharpe = compute_sharpe_ratio(daily_returns)
            
            # Calculate maximum drawdown
            max_drawdown = calculate_maximum_drawdown(cum_wealth)
            
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': final_wealth,
                'exp_growth': exp_growth,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'runtime_seconds': runtime
            }
            
            # Add all parameters to the result
            for key, value in params.items():
                result[key] = value
                
            return result
        
        except Exception as e:
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': np.nan,
                'exp_growth': np.nan,
                'sharpe': np.nan,
                'max_drawdown': np.nan,
                'runtime_seconds': np.nan,
                'error': str(e)
            }
            for key, value in params.items():
                result[key] = value
                
            return result
            
    def _generate_param_combinations(self):
        """
        Generate all combinations of parameters from the parameter grid.
        
        Returns:
        --------
        list
            List of parameter dictionaries
        """
        # If we have custom parameter combinations, return those directly
        if hasattr(self, 'custom_param_combinations') and self.custom_param_combinations is not None:
            return self.custom_param_combinations
        
        # Otherwise generate combinations from the parameter grid
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        param_dicts = []
        for params in itertools.product(*param_values):
            param_dict = dict(zip(param_names, params))
            param_dicts.append(param_dict)
            
        return param_dicts
        
    def _function_name_to_callable(self, func_name_or_obj):
        """
        Convert a function name string to its callable object if needed.
        
        Parameters:
        -----------
        func_name_or_obj : str or callable
            Function name or function object
            
        Returns:
        --------
        callable
            The callable function object
        """
        if callable(func_name_or_obj):
            return func_name_or_obj
            
        # Import specific functions that might be needed
        from Strategies.pattern_matching import (
            histogram_based_selection, kernel_based_selection, 
            nearest_neighbor_selection, correlation_based_selection,
            log_optimal_portfolio, semi_log_optimal_portfolio, markowitz_portfolio
        )
        
        # Mapping of function names to actual functions
        function_map = {
            'histogram_based_selection': histogram_based_selection,
            'kernel_based_selection': kernel_based_selection,
            'nearest_neighbor_selection': nearest_neighbor_selection,
            'correlation_based_selection': correlation_based_selection,
            'log_optimal_portfolio': log_optimal_portfolio,
            'semi_log_optimal_portfolio': semi_log_optimal_portfolio,
            'markowitz_portfolio': markowitz_portfolio
        }
        
        # Check if it's a string that exists in our mapping
        if isinstance(func_name_or_obj, str) and func_name_or_obj in function_map:
            return function_map[func_name_or_obj]
        
        # If all else fails, return the original
        return func_name_or_obj
    
    def _callable_to_function_name(self, func):
        """
        Convert a callable function to its string name for storage in results.
        
        Parameters:
        -----------
        func : callable
            Function to convert to string name
            
        Returns:
        --------
        str
            String name representing the function
        """
        from Strategies.pattern_matching import (
            histogram_based_selection, kernel_based_selection, 
            nearest_neighbor_selection, correlation_based_selection,
            log_optimal_portfolio, semi_log_optimal_portfolio, markowitz_portfolio
        )
        
        # Create reverse mapping of functions to names
        function_map = {
            histogram_based_selection: 'histogram_based_selection',
            kernel_based_selection: 'kernel_based_selection',
            nearest_neighbor_selection: 'nearest_neighbor_selection',
            correlation_based_selection: 'correlation_based_selection',
            log_optimal_portfolio: 'log_optimal_portfolio',
            semi_log_optimal_portfolio: 'semi_log_optimal_portfolio',
            markowitz_portfolio: 'markowitz_portfolio'
        }
        
        # If the function is in our map, return its name
        if func in function_map:
            return function_map[func]
        # If the function has a __name__ attribute, use that
        if hasattr(func, '__name__'):
            # Clean up wrapper function names
            name = func.__name__
            return name
        
        # If all else fails, return the string representation
        return str(func)

    def _evaluate_param_combo(self, param_dict, b, price_relative_vectors):
        """
        Wrapper for evaluating a single parameter combination.
        """
        # Create a deep copy to avoid modifying the original
        param_dict = param_dict.copy()

        # Handle the case where parameters are nested in 'methods' dictionary
        if 'methods' in param_dict:
            methods = param_dict.pop('methods')
            w = param_dict.pop('w', 3)
            
            # Pass through any remaining parameters like threshold, lambda_, etc.
            # These will be handled by pattern_matching_portfolio_master
            return self.evaluate_strategy(b, price_relative_vectors, methods=methods, w=w, **param_dict)
        
        # Standard parameter evaluation for non-pattern-matching strategies
        return self.evaluate_strategy(b, price_relative_vectors, **param_dict)
        
    def _create_train_val_splits(self):
        """
        Create training and validation splits for walk-forward validation using a rolling window.
        This ensures each validation window has meaningful training data, especially for earlier windows.
        
        Returns:
        --------
        list
            List of (train_indices, val_indices) tuples
        """
        T = len(self.price_relative_df)
        val_size = int(T * self.validation_window_size)
        
        # Define minimum training size (30% of data by default)
        min_train_size = int(T * 0.4)  # Adjust this percentage as needed
        
        splits = []
        for i in range(self.validation_windows):
            val_end = T - i * val_size
            val_start = val_end - val_size
            
            # For rolling window, set a fixed-size training window that ends at val_start
            train_end = val_start
            train_start = max(0, train_end - min_train_size)
            
            val_indices = np.arange(val_start, val_end)
            train_indices = np.arange(train_start, train_end)
            
            # Only add the window if we have meaningful training data
            if len(train_indices) > 0:
                splits.append((train_indices, val_indices))
        
        # If we have fewer splits than requested windows (due to data constraints),
        # adjust the validation_windows property to match
        if len(splits) < self.validation_windows:
            # Removed debug print statement here
            self.validation_windows = len(splits)
            
        return splits[::-1]  # Reverse to go from earliest to latest
        
    def _eval_single_window(self, params, train_data, val_data):
        """Helper function to evaluate a single validation window"""
        try:
            num_assets = train_data.shape[1]
            b = np.ones(num_assets) / num_assets
            
            # For pattern matching, get window size from methods or params
            if 'methods' in params:
                w = params.get('w', 3)
            else:
                w = params.get('w', 1)
                
            # Skip if validation data is too small for the window
            if len(val_data) < w:
                return None
                
            # Train on training data
            train_result = self._evaluate_param_combo(params, b, train_data.values)
            if pd.isna(train_result.get('final_wealth', np.nan)):
                return None
                
            # Then validate on validation data
            val_result = self._evaluate_param_combo(params, b, val_data.values)
            if val_result is None or pd.isna(val_result.get('final_wealth', np.nan)):
                return None
                
            return val_result
        except Exception as e:
            print(f"Error in _eval_single_window: {str(e)}")
            return None

    def tune_strategy(self):
        """Run hyperparameter tuning for the strategy."""
        results = []
        
        # Handle param grid format
        if isinstance(self.param_grid, list):
            param_combinations = self.param_grid
        else:
            param_combinations = self._generate_param_combinations()
            
        print(f"\nEvaluating {len(param_combinations)} parameter combinations...")

        if self.use_walk_forward:
            splits = self._create_train_val_splits()
            print(f"Using {len(splits)} validation windows")
            
            if self.parallel:
                from joblib import Parallel, delayed
                n_jobs = self.n_jobs if self.n_jobs else max(1, multiprocessing.cpu_count() - 2)
                
                # Parallelize over parameter combinations
                results = []
                for params in tqdm(param_combinations):
                    window_results = []
                    for train_indices, val_indices in splits:
                        train_data = self.price_relative_df.iloc[train_indices]
                        val_data = self.price_relative_df.iloc[val_indices]
                        result = self._eval_single_window(params, train_data, val_data)
                        if result is not None:
                            window_results.append(result)
                            
                    if window_results:
                        # Average metrics across validation windows
                        avg_result = {
                            **params,
                            'avg_val_final_wealth': np.mean([r['final_wealth'] for r in window_results]),
                            'avg_val_sharpe': np.mean([r['sharpe'] for r in window_results]),
                            'avg_val_max_drawdown': np.mean([r['max_drawdown'] for r in window_results])
                        }
                        results.append(avg_result)
            else:
                for params in tqdm(param_combinations):
                    window_results = []
                    for train_indices, val_indices in splits:
                        train_data = self.price_relative_df.iloc[train_indices]
                        val_data = self.price_relative_df.iloc[val_indices]
                        result = self._eval_single_window(params, train_data, val_data)
                        if result is not None:
                            window_results.append(result)
                            
                    if window_results:
                        # Average metrics across validation windows
                        avg_result = {
                            **params,
                            'avg_val_final_wealth': np.mean([r['final_wealth'] for r in window_results]),
                            'avg_val_sharpe': np.mean([r['sharpe'] for r in window_results]),
                            'avg_val_max_drawdown': np.mean([r['max_drawdown'] for r in window_results])
                        }
                        results.append(avg_result)
        else:
            if self.parallel:
                from joblib import Parallel, delayed
                n_jobs = self.n_jobs if self.n_jobs else max(1, multiprocessing.cpu_count() - 2)
                
                # Parallelize over parameter combinations
                results = Parallel(n_jobs=n_jobs)(
                    delayed(lambda p: {
                        **p,
                        **self.evaluate_strategy(
                            np.ones(self.price_relative_df.shape[1]) / self.price_relative_df.shape[1],
                            self.price_relative_df.values,
                            **p
                        )
                    })(params)
                    for params in tqdm(param_combinations)
                )
            else:
                for params in tqdm(param_combinations):
                    b = np.ones(self.price_relative_df.shape[1]) / self.price_relative_df.shape[1]
                    result = self.evaluate_strategy(b, self.price_relative_df.values, **params)
                    results.append({**params, **result})

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best parameters
        metric_col = 'avg_val_sharpe' if self.use_walk_forward else 'sharpe'
        if metric_col in results_df.columns and not results_df[metric_col].isnull().all():
            best_idx = results_df[metric_col].idxmax()
            self.best_params = param_combinations[best_idx] if best_idx is not None else {}
            print(f"\nBest {metric_col}: {results_df[metric_col].max():.4f}")
        else:
            self.best_params = {}
            print("\nWarning: Could not determine best parameters")
        
        self.results_df = results_df
        return results_df, self.best_params
    
    def save_results(self, filename=None):
        """
        Save tuning results to CSV.
        
        
        Parameters:
        -----------
        filename : str, optional
            File name for the results CSV. If None, derives the name from the strategy name
            
        Returns:
        --------
        str
            Path to the saved file
        """
        if filename is None:
            filename = f"{self.strategy_name.lower().replace(' ', '_')}_tuning_results.csv"
            
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        output_file = os.path.join(self.output_path, filename)
        self.results_df.to_csv(output_file, index=False)
        
        print(f"Results saved to {output_file}")
        return output_file
    
    def print_best_params(self):
        """
        Print the best parameter combination and its performance.
        """
        if self.best_params is None:
            print("No tuning results available. Run tune_strategy() first.")
            return
            
        print(f"=== Best {self.strategy_name} Configuration ===")
        for key, value in self.best_params.items():
            print(f"{key}: {value}")

    def evaluate_params(self, params):
        """
        Default implementation of evaluate_params that calls evaluate_strategy.
        """
        num_assets = self.price_relative_df.shape[1]
        b = np.ones(num_assets) / num_assets
        return self.evaluate_strategy(b, self.price_relative_df.values, **params)

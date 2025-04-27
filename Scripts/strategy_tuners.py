"""
Strategy-specific tuners for hyperparameter optimization.

This module provides specialized tuner classes for different categories of
portfolio optimization strategies.
"""

import pandas as pd
import numpy as np
import itertools

from utilities import *
import time

from tuning_framework import StrategyTuner
from Strategies.benchmarks import *
from Strategies.follow_the_loser import *
from Strategies.follow_the_winner import *
from Strategies.pattern_matching import *
from Strategies.meta_learning import *


class FollowTheWinnerTuner(StrategyTuner):
    """
    Tuner for Follow-the-Winner strategies.
    """
    
    def __init__(self, strategy_name, strategy_func, price_relative_df, **kwargs):
        super().__init__(strategy_name, strategy_func, price_relative_df, **kwargs)


class FollowTheLoserTuner(StrategyTuner):
    """
    Tuner for Follow-the-Loser strategies.
    """
    
    def __init__(self, strategy_name, strategy_func, price_relative_df, **kwargs):
        super().__init__(strategy_name, strategy_func, price_relative_df, **kwargs)

    def evaluate_params(self, params):
        """
        Evaluate a set of parameters for Follow-the-Loser strategies.
        """
        import numpy as np
        import time

        try:
            # Initialize portfolio (equal weight)
            num_assets = self.price_relative_df.shape[1]
            b = np.ones(num_assets) / num_assets

            # Run the strategy
            start_time = time.time()
            b_n = self.strategy_func(b, self.price_relative_df.values, **params)
            runtime = time.time() - start_time

            # Calculate performance metrics
            final_wealth = calculate_cumulative_wealth(b_n, self.price_relative_df.values)
            n_periods = len(self.price_relative_df)
            exp_growth = calculate_exponential_growth_rate(final_wealth, n_periods)

            # Calculate Sharpe ratio
            cum_wealth = calculate_cumulative_wealth_over_time(b_n, self.price_relative_df.values)
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


class PatternMatchingTuner(StrategyTuner):
    """
    Tuner for Pattern Matching strategies.
    """
    
    def __init__(self, strategy_name, strategy_func, price_relative_df, **kwargs):
        super().__init__(strategy_name, strategy_func, price_relative_df, **kwargs)

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
            if name == 'sample_selection_wrapper':
                return 'histogram_based_selection'  # Default selection method
            elif name == 'portfolio_optimization_wrapper':
                return 'log_optimal_portfolio'  # Default optimization method
            return name
        
        # If all else fails, return the string representation
        return str(func)

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
            'markowitz_portfolio': markowitz_portfolio,
        }
        
        # Check if it's a string that exists in our mapping
        if isinstance(func_name_or_obj, str) and func_name_or_obj in function_map:
            return function_map[func_name_or_obj]
        
        # If all else fails, return a simple passthrough function
        return func_name_or_obj

    def _prepare_methods_dict(self, ss_func, po_func, ss_params, po_params):
        """Helper to prepare the methods dictionary with proper function objects"""
        methods = {
            'sample_selection': lambda X, w: ss_func(X, w, **ss_params),
            'portfolio_optimization': lambda C, X: po_func(C, X, **po_params)
        }
        return methods

    def evaluate_strategy(self, b, price_relative_vectors, **params):
        """
        Override to handle pattern matching specific parameters.
        """
        try:
            # Initialize portfolio (equal weight)
            num_assets = price_relative_vectors.shape[1]
            b = np.ones(num_assets) / num_assets

            # Handle case where methods dictionary is already provided
            if 'methods' in params:
                methods = params.pop('methods')
                
                # Convert any string function names to callable functions and store original names
                if 'sample_selection' in methods:
                    ss_func = self._function_name_to_callable(methods['sample_selection'])
                    ss_name = self._callable_to_function_name(ss_func)
                    methods['sample_selection'] = ss_func
                
                if 'portfolio_optimization' in methods:
                    po_func = self._function_name_to_callable(methods['portfolio_optimization'])
                    po_name = self._callable_to_function_name(po_func)
                    methods['portfolio_optimization'] = po_func
                
                # Execute the pattern matching strategy
                start_time = time.time()
                b_n = pattern_matching_portfolio_master(b, price_relative_vectors, methods=methods, **params)
                runtime = time.time() - start_time

                # Store the actual function names in the results
                params['sample_selection'] = ss_name
                params['portfolio_optimization'] = po_name

            # For pattern matching strategies that require special handling with individual parameters
            elif 'sample_selection' in params and 'portfolio_optimization' in params:
                ss_func = self._function_name_to_callable(params['sample_selection'])
                po_func = self._function_name_to_callable(params['portfolio_optimization'])
                
                # Store original function names
                ss_name = self._callable_to_function_name(ss_func)
                po_name = self._callable_to_function_name(po_func)
                
                methods = {
                    'sample_selection': ss_func,
                    'portfolio_optimization': po_func
                }
                
                # Execute the pattern matching strategy
                start_time = time.time()
                b_n = pattern_matching_portfolio_master(b, price_relative_vectors, methods=methods, **params)
                runtime = time.time() - start_time

                # Store the actual function names in the results
                params['sample_selection'] = ss_name
                params['portfolio_optimization'] = po_name
            else:
                # Standard execution
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

            # Create a comprehensive result dictionary with validation metrics
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': final_wealth,
                'exp_growth': exp_growth,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'runtime_seconds': runtime,
                'avg_val_final_wealth': final_wealth,  # For consistency with validation metrics
                'avg_val_sharpe': sharpe,  # For consistency with validation metrics
                'avg_val_max_drawdown': max_drawdown  # For consistency with validation metrics
            }
            
            # Add all parameters to the result
            for key, value in params.items():
                if key not in result:
                    result[key] = value
            
            return result
            
        except Exception as e:
            print(f"Error in evaluate_strategy: {str(e)}")
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': np.nan,
                'exp_growth': np.nan,
                'sharpe': np.nan,
                'max_drawdown': np.nan,
                'runtime_seconds': np.nan,
                'avg_val_final_wealth': np.nan,
                'avg_val_sharpe': np.nan,
                'avg_val_max_drawdown': np.nan,
                'error': str(e)
            }
            
            # Add all parameters to result even on error
            for key, value in params.items():
                if key not in result:
                    result[key] = value
                
            return result

    def evaluate_params(self, params):
        """
        Evaluate a set of parameters for pattern matching strategies.
        """
        try:
            from joblib import Parallel, delayed
            # Initialize portfolio
            num_assets = self.price_relative_df.shape[1]
            b = np.ones(num_assets) / num_assets
            
            # Extract methods and their parameters
            methods = params.get('methods', {})
            if isinstance(methods, dict):
                ss_func = self._function_name_to_callable(methods.get('sample_selection'))
                po_func = self._function_name_to_callable(methods.get('portfolio_optimization'))
            else:
                ss_func = self._function_name_to_callable(params.get('sample_selection'))
                po_func = self._function_name_to_callable(params.get('portfolio_optimization'))

            # Extract parameters for sample selection
            ss_params = {}
            if 'threshold' in params:
                ss_params['threshold'] = params['threshold']
            if 'num_neighbors' in params:
                ss_params['num_neighbors'] = params['num_neighbors']
            if 'rho' in params:
                ss_params['rho'] = params['rho']

            # Extract parameters for portfolio optimization
            po_params = {}
            if 'lambda_' in params:
                po_params['lambda_'] = params['lambda_']

            # Prepare methods dictionary with wrapped functions that include parameters
            methods = self._prepare_methods_dict(ss_func, po_func, ss_params, po_params)

            # Parallelize the strategy execution across time periods
            T = len(self.price_relative_df)
            chunk_size = max(1, T // (multiprocessing.cpu_count() - 1))
            chunks = [(i, min(i + chunk_size, T)) for i in range(0, T, chunk_size)]
            
            def process_chunk(start, end):
                return self.strategy_func(b, self.price_relative_df.values[start:end], methods=methods, w=params.get('w', 3))

            # Execute chunks in parallel
            b_n_chunks = Parallel(n_jobs=-1)(delayed(process_chunk)(start, end) for start, end in chunks)
            b_n = np.vstack(b_n_chunks)

            # Calculate performance metrics
            final_wealth = calculate_cumulative_wealth(b_n, self.price_relative_df.values)
            n_periods = len(self.price_relative_df)
            exp_growth = calculate_exponential_growth_rate(final_wealth, n_periods)

            # Calculate Sharpe ratio
            cum_wealth = calculate_cumulative_wealth_over_time(b_n, self.price_relative_df.values)
            daily_returns = compute_periodic_returns(cum_wealth)
            sharpe = compute_sharpe_ratio(daily_returns)

            # Calculate maximum drawdown
            max_drawdown = calculate_maximum_drawdown(cum_wealth)

            # Store results with function names instead of objects
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': final_wealth,
                'exp_growth': exp_growth,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'avg_val_final_wealth': final_wealth,
                'avg_val_sharpe': sharpe,
                'avg_val_max_drawdown': max_drawdown,
                'sample_selection': self._callable_to_function_name(ss_func),
                'portfolio_optimization': self._callable_to_function_name(po_func),
                'w': params.get('w', 3)
            }

            # Add specific parameter values
            if 'threshold' in params:
                result['threshold'] = params['threshold']
            if 'num_neighbors' in params:
                result['num_neighbors'] = params['num_neighbors']
            if 'rho' in params:
                result['rho'] = params['rho']
            if 'lambda_' in params:
                result['lambda_'] = params['lambda_']

            return result

        except Exception as e:
            print(f"Error in evaluate_params: {str(e)}")
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': np.nan,
                'exp_growth': np.nan,
                'sharpe': np.nan,
                'max_drawdown': np.nan,
                'avg_val_final_wealth': np.nan,
                'avg_val_sharpe': np.nan,
                'avg_val_max_drawdown': np.nan,
                'error': str(e)
            }

            # Add all parameters even on error
            if isinstance(methods, dict):
                result['sample_selection'] = self._callable_to_function_name(methods.get('sample_selection'))
                result['portfolio_optimization'] = self._callable_to_function_name(methods.get('portfolio_optimization'))
            for key, value in params.items():
                if key not in ['methods', 'sample_selection', 'portfolio_optimization']:
                    result[key] = value

            return result


class StrategyFactory:
    """
    Factory class to create appropriate tuners for different strategies.
    """
    
    @staticmethod
    def create_tuner(strategy_name, strategy_func, price_relative_df, **kwargs):
        """
        Create a tuner instance for the specified strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
            
        strategy_func : callable
            Function implementing the strategy
            
        price_relative_df : pandas.DataFrame
            DataFrame containing price relative vectors
            
        **kwargs : dict
            Additional parameters for the tuner
            
        Returns:
        --------
        StrategyTuner
            An instance of the appropriate tuner class
        """
        # Determine strategy category based on the function
        ftw_strategies = [follow_the_leader, exponential_gradient, follow_the_regularized_leader, 
                         aggregation_based_simple, universal_portfolios]
        
        ftl_strategies = [anticor, pamr, cwmr, olmar, rmr]
        
        pm_strategies = [pattern_matching_portfolio_master]
        
        meta_strategies = [online_gradient_update_meta, online_newton_update_meta, 
                         fast_universalization, follow_the_leading_history,
                         aggregation_algorithm_generalized]
        
        # Create appropriate tuner
        if strategy_func in ftw_strategies:
            return FollowTheWinnerTuner(strategy_name, strategy_func, price_relative_df, **kwargs)
        elif strategy_func in ftl_strategies:
            return FollowTheLoserTuner(strategy_name, strategy_func, price_relative_df, **kwargs)
        elif strategy_func in pm_strategies:
            return PatternMatchingTuner(strategy_name, strategy_func, price_relative_df, **kwargs)
        elif strategy_func in meta_strategies:
            return MetaLearningTuner(strategy_name, strategy_func, price_relative_df, **kwargs)
        else:
            # Default to base tuner
            return StrategyTuner(strategy_name, strategy_func, price_relative_df, **kwargs)

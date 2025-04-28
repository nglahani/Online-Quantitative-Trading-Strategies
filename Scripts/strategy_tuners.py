"""
Strategy-specific tuners for hyperparameter optimization.

This module provides specialized tuner classes for different categories of
portfolio optimization strategies.
"""

import pandas as pd
import numpy as np
import itertools
import multiprocessing

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


class PatternMatchingTuner(StrategyTuner):
    """
    Tuner for Pattern Matching strategies.
    """
    
    def __init__(self, strategy_name, strategy_func, price_relative_df, **kwargs):
        super().__init__(strategy_name, strategy_func, price_relative_df, **kwargs)

    def _callable_to_function_name(self, func):
        """Convert callable to string name for storage"""
        from Strategies.pattern_matching import (
            histogram_based_selection, kernel_based_selection, 
            nearest_neighbor_selection, correlation_based_selection,
            log_optimal_portfolio, semi_log_optimal_portfolio, markowitz_portfolio
        )
        
        function_map = {
            histogram_based_selection: 'histogram_based_selection',
            kernel_based_selection: 'kernel_based_selection',
            nearest_neighbor_selection: 'nearest_neighbor_selection',
            correlation_based_selection: 'correlation_based_selection',
            log_optimal_portfolio: 'log_optimal_portfolio',
            semi_log_optimal_portfolio: 'semi_log_optimal_portfolio',
            markowitz_portfolio: 'markowitz_portfolio'
        }
        
        if func in function_map:
            return function_map[func]
        return func.__name__ if hasattr(func, '__name__') else str(func)

    def _function_name_to_callable(self, func_name_or_obj):
        """Convert string name to callable if needed"""
        if callable(func_name_or_obj):
            return func_name_or_obj
            
        from Strategies.pattern_matching import (
            histogram_based_selection, kernel_based_selection, 
            nearest_neighbor_selection, correlation_based_selection,
            log_optimal_portfolio, semi_log_optimal_portfolio, markowitz_portfolio
        )
        
        function_map = {
            'histogram_based_selection': histogram_based_selection,
            'kernel_based_selection': kernel_based_selection,
            'nearest_neighbor_selection': nearest_neighbor_selection,
            'correlation_based_selection': correlation_based_selection,
            'log_optimal_portfolio': log_optimal_portfolio,
            'semi_log_optimal_portfolio': semi_log_optimal_portfolio,
            'markowitz_portfolio': markowitz_portfolio
        }
        
        if isinstance(func_name_or_obj, str) and func_name_or_obj in function_map:
            return function_map[func_name_or_obj]
        return func_name_or_obj


class MetaLearningTuner(StrategyTuner):
    """
    Tuner for Meta-Learning strategies.
    """
    
    def __init__(self, strategy_name, strategy_func, price_relative_df, **kwargs):
        super().__init__(strategy_name, strategy_func, price_relative_df, **kwargs)
        
        # Map of strategy names to functions for base experts
        self.expert_map = {
            'cwmr': cwmr,
            'pamr': pamr,
            'anticor': anticor,
            'olmar': olmar,
            'rmr': rmr,
            'follow_the_regularized_leader': follow_the_regularized_leader,
            'exponential_gradient': exponential_gradient,
            'universal_portfolios': universal_portfolios,
        }
        
    def _convert_experts_to_functions(self, experts):
        """Helper to convert expert names to function objects"""
        if isinstance(experts[0], str):
            return [self.expert_map[name] for name in experts]
        return experts
        
    def _convert_experts_to_names(self, experts):
        """Helper to convert expert functions to name strings"""
        if callable(experts[0]):
            reverse_map = {v: k for k, v in self.expert_map.items()}
            return [reverse_map[func] for func in experts]
        return experts

    def evaluate_strategy(self, b, price_relative_vectors, **params):
        """
        Override to handle meta-learning specific parameters like base experts.
        """
        try:
            # Deep copy parameters to avoid modifying the original
            params = params.copy()
            
            # Handle base_experts parameter if present
            if 'base_experts' in params:
                # Convert expert names to functions for execution
                orig_experts = params['base_experts']
                params['base_experts'] = self._convert_experts_to_functions(orig_experts)
            
            # Execute strategy
            start_time = time.time()
            b_n = self.strategy_func(b, price_relative_vectors, **params)
            runtime = time.time() - start_time
            
            # Calculate metrics
            final_wealth = calculate_cumulative_wealth(b_n, price_relative_vectors)
            n_periods = len(price_relative_vectors)
            exp_growth = calculate_exponential_growth_rate(final_wealth, n_periods)
            
            # Calculate Sharpe ratio 
            cum_wealth = calculate_cumulative_wealth_over_time(b_n, price_relative_vectors)
            daily_returns = compute_periodic_returns(cum_wealth)
            sharpe = compute_sharpe_ratio(daily_returns)
            
            # Calculate maximum drawdown
            max_drawdown = calculate_maximum_drawdown(cum_wealth)
            
            # Create result dictionary
            result = {
                'algorithm': self.strategy_name,
                'final_wealth': final_wealth,
                'exp_growth': exp_growth,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'runtime_seconds': runtime,
                'avg_val_final_wealth': final_wealth,
                'avg_val_sharpe': sharpe,
                'avg_val_max_drawdown': max_drawdown
            }
            
            # Add parameters to result
            for key, value in params.items():
                if key == 'base_experts':
                    # Convert functions back to strings for storage
                    result[key] = self._convert_experts_to_names(orig_experts)
                else:
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
            
            # Add params even on error
            for key, value in params.items():
                if key == 'base_experts':
                    result[key] = self._convert_experts_to_names(value)
                else:
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

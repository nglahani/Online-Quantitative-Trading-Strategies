# Hyperparameter Tuning Framework

This framework provides a unified approach to hyperparameter tuning for portfolio optimization strategies, with support for walk-forward validation to reduce overfitting.

## Key Features

- Object-oriented design for modularity and extensibility
- Support for all existing portfolio optimization strategies
- Walk-forward validation to prevent overfitting
- Parallel processing for faster tuning
- Flexible parameter grid specification
- Save tuning results to CSV files
- Optimizes based on Sharpe ratio

## Core Components

- `tuning_framework.py`: Base tuning functionality and walk-forward validation
- `strategy_tuners.py`: Strategy-specific tuners and factory
- `run_hyperparameter_tuning.py`: CLI for tuning a single strategy
- `tune_multiple_strategies.py`: Script for tuning multiple strategies

## Examples

### Tuning a Single Strategy

```bash
# Basic usage
python run_hyperparameter_tuning.py follow_the_leader

# With walk-forward validation
python run_hyperparameter_tuning.py follow_the_leader --walk-forward

# Custom parameters
python run_hyperparameter_tuning.py follow_the_leader --data-file ../Data/price_relative_vectors_test.csv --output-dir ../Results

# Pattern matching strategy (more complex)
python run_hyperparameter_tuning.py pattern_matching --walk-forward --val-windows 3 --val-size 0.15
```

### Tuning Multiple Strategies

```bash
# Tune specific strategies
python tune_multiple_strategies.py --strategies follow_the_leader exponential_gradient --walk-forward

# Tune all strategies in a category
python tune_multiple_strategies.py --category ftl --walk-forward

# Tune all available strategies
python tune_multiple_strategies.py --all-strategies --walk-forward --parallel-strategies

# Tune with custom parameters
python tune_multiple_strategies.py --category meta --walk-forward --val-windows 3 --val-size 0.15
```

## Understanding Walk-Forward Validation

Walk-forward validation is a technique that helps prevent overfitting in time series data by:

1. Dividing the data into multiple training and validation sets, moving through time
2. For each period:
   - Train on data up to a specific point
   - Validate on the subsequent data segment
3. Averaging performance across all validation windows

**Benefits:**
- More realistic assessment of strategy performance
- Identifies parameter stability over time
- Reduces overfitting risk
- Makes better use of limited historical data


## Adding New Strategies

To add a new strategy to the tuning framework:

1. Implement the strategy in the appropriate module in `Strategies/`
2. Add the strategy to the `strategy_map` in `run_hyperparameter_tuning.py`
3. Add default parameter grid in `get_default_param_grid()` function
4. Add the strategy to the appropriate category in `StrategyFactory` in `strategy_tuners.py`

## Customizing Parameter Grids

You can create a custom parameter grid by:

1. Creating a Python file with a `param_grid` dictionary
2. Passing the file to the tuner with the `--param-file` option

Example custom parameter file:
```python
# custom_pamr_grid.py
param_grid = {
    'epsilon': [0.7, 0.8, 0.9, 1.0],
    'C': [1.0, 5.0, 10.0, 15.0, 20.0]
}
```

Usage:
```bash
python run_hyperparameter_tuning.py pamr --param-file custom_pamr_grid.py
```

## Parallel Processing

The framework supports parallel processing at two levels:

1. Parameter combinations within a strategy (default)
2. Multiple strategies in parallel (`--parallel-strategies` option)

Control parallel execution with:
- `--no-parallel`: Disable parallel processing
- `--n-jobs <N>`: Specify number of parallel jobs
- `--parallel-strategies`: Run multiple strategies in parallel

## Output Files

Results are saved in the specified output directory (default: `../Data/Tuning Data/`):

- `<strategy_name>_tuning_results.csv`: Detailed results for each parameter combination
- `tuning_summary.csv`: Summary of best parameters for each strategy (when tuning multiple strategies)

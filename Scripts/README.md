# Hyperparameter Tuning Framework

This framework provides a unified approach to hyperparameter tuning for portfolio optimization strategies, with extensive support for walk-forward validation and parallel processing.

## Key Features

- **Advanced Strategy Support:**
  - Follow-the-Winner (FTW): Momentum-based strategies
  - Follow-the-Loser (FTL): Mean reversion strategies
  - Pattern Matching (PM): Historical similarity-based approaches
  - Meta-Learning: Ensemble and hybrid approaches

- **Flexible Validation:**
  - Walk-forward validation with configurable windows
  - Smart window sizing based on data characteristics
  - Robust cross-validation metrics

- **Performance Optimizations:**
  - Two-level parallel processing (strategies and parameters)
  - Automatic CPU core allocation
  - Efficient parameter space exploration

- **Extended Metrics:**
  - Sharpe ratio optimization
  - Maximum drawdown tracking
  - Runtime performance monitoring

## Core Components

### Base Framework (`tuning_framework.py`)
- StrategyTuner base class
- Walk-forward validation implementation
- Parallel processing management
- Performance metrics calculation

### Strategy-Specific Tuners (`strategy_tuners.py`)
- FollowTheWinnerTuner
- FollowTheLoserTuner
- PatternMatchingTuner
- MetaLearningTuner
- StrategyFactory for dynamic tuner creation

### CLI Interfaces
- `run_hyperparameter_tuning.py`: Single strategy tuning
- `tune_multiple_strategies.py`: Multi-strategy optimization

## Usage Examples

### Single Strategy Tuning

```bash
# Basic usage
python run_hyperparameter_tuning.py follow_the_leader

# With walk-forward validation
python run_hyperparameter_tuning.py follow_the_leader --walk-forward

# Custom validation windows
python run_hyperparameter_tuning.py follow_the_leader --walk-forward --val-windows 3 --val-size 0.15

# Parallel processing control
python run_hyperparameter_tuning.py follow_the_leader --n-jobs 4
```

### Multiple Strategy Tuning

```bash
# Category-based tuning
python tune_multiple_strategies.py --category ftl --walk-forward

# Specific strategies with parallel processing
python tune_multiple_strategies.py --strategies follow_the_leader exponential_gradient --parallel-strategies

# All strategies with custom validation
python tune_multiple_strategies.py --all-strategies --val-windows 3 --val-size 0.15
```

## Parameter Configuration

### Default Parameter Grids

Follow-the-Winner strategies:
```python
'follow_the_leader': {
    'gamma': [0.8, 0.9, 1.0, 1.1],
    'alpha': [1.5, 2.0, 2.5, 3.0]
}

'exponential_gradient': {
    'learning_rate': [0.05, 0.1, 0.15, 0.2],
    'smoothing': [0.0, 0.1, 0.2]
}
```

Follow-the-Loser strategies:
```python
'pamr': {
    'epsilon': [0.7, 0.8, 0.9, 1.0],
    'C': [5.0, 8.0, 10.0, 12.0]
}

'olmar': {
    'window_size': [2, 3, 4],
    'epsilon': [0.8, 0.9, 1.0, 1.1],
    'eta': [20, 25, 30, 35]
}
```

Meta-Learning strategies:
```python
'fast_universalization': {
    'learning_rate': [0.05, 0.08, 0.1, 0.12],
    'base_experts': [
        ['cwmr', 'pamr', 'follow_the_regularized_leader'],
        ['cwmr', 'pamr', 'rmr', 'olmar']
    ]
}
```

### Custom Parameter Grids

Create a Python file with your parameter grid:
```python
# custom_grid.py
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'window_size': [2, 3, 4, 5]
}
```

Use with the tuning script:
```bash
python run_hyperparameter_tuning.py strategy_name --param-file custom_grid.py
```

## Walk-Forward Validation

The framework implements walk-forward validation to prevent overfitting:

1. **Data Splitting:**
   - Divides historical data into multiple validation windows
   - Maintains temporal order of data
   - Configurable window size and count

2. **Validation Process:**
   - Trains on data up to window start
   - Validates on window period
   - Aggregates metrics across windows

3. **Benefits:**
   - More realistic performance assessment
   - Captures parameter stability
   - Reduces overfitting risk

## Adding New Strategies

1. Implement strategy in appropriate module under `Strategies/`
2. Add to strategy map in `run_hyperparameter_tuning.py`:
```python
strategy_map = {
    'your_strategy': your_strategy_function
}
```

3. Add default parameter grid:
```python
default_grids = {
    'your_strategy': {
        'param1': [value1, value2],
        'param2': [value3, value4]
    }
}
```

4. Create strategy-specific tuner if needed:
```python
class YourStrategyTuner(StrategyTuner):
    def __init__(self, strategy_name, strategy_func, price_relative_df, **kwargs):
        super().__init__(strategy_name, strategy_func, price_relative_df, **kwargs)
```

## Output Files

Results are saved in the specified output directory (default: `../Data/Tuning Data/`):

- `<strategy_name>_tuning_results.csv`: Detailed parameter combination results
  - Final wealth
  - Sharpe ratio
  - Maximum drawdown
  - Runtime performance
  - Validation metrics (if using walk-forward)

- `tuning_summary.csv`: When tuning multiple strategies
  - Best parameters per strategy
  - Comparative performance metrics
  - Runtime statistics

# Online Quantitative Trading Strategies

A comprehensive Python-based framework for evaluating and comparing online portfolio selection methods. This repository provides a robust environment for exploring a range of quantitative trading strategies—from momentum-based Follow-the-Winner approaches and mean-reversion-based Follow-the-Loser techniques to advanced pattern-matching and meta-learning ensembles—all benchmarked against traditional trading strategies.

## Overview

This project implements a unified trading infrastructure for online portfolio selection. The framework covers the entire pipeline, including:
- **Data Processing:** Extracting and preparing historical price data from CSV files to compute price-relative vectors with Price_Relative_Vector_Creation.ipynb
- **Algorithm Implementation:** Multiple algorithm families such as Follow-the-Winner (e.g., Exponential Gradient, Follow-the-Leader, Follow-the-Regularized-Leader), Follow-the-Loser (e.g., Anticorrelation, PAMR, CWMR, OLMAR, RMR), Pattern-Matching approaches, and Meta-Learning ensembles in the [Strategies](Scripts/Strategies/) folder
- **Hyperparameter Tuning:** Automated grid search with parallel processing is employed to efficiently explore various parameter combinations and fine-tune the strategies, ensuring optimal performance based on metrics such as cumulative wealth, exponential growth rate, and the Sharpe ratio. Each algorithm has it's own associated hyperparameter tuning scripts in the [Archived Tuners](Scripts/Archived_Tuners/) folder. 
- **Performance Evaluation:** Tools to assess performance metrics among algorithm types including cumulative wealth, exponential growth rate, and the Sharpe ratio in in the [Strategy_Comparisons](Scripts/Strategy_Comparisons/) folder

For a detailed discussion of the methodologies, experiments, and empirical results, please refer to the research paper available in the [Docs](Docs/) folder.

## Repository Structure

```plaintext
.
├── Data
│   ├── Input Data
│   │   └── QuantQuote_Minute.pdf
│   ├── Price Relative Vectors
│   │   ├── price_relative_vectors.csv
│   │   └── price_relative_vectors_test.csv
│   └── Tuning Data
│       └── (various tuning results CSV files)
├── Docs
│   └── Online Portfolio Selection - A Survey.pdf  # Research paper with details on methodology and results
├── Scripts
│   ├── Active_Tuner.ipynb             # Interactive notebook for parameter tuning
│   ├── Price_Relative_Vector_Creation.ipynb  # Notebook to compute price-relative vectors
│   ├── requirements.txt               # Python dependencies
│   ├── utilities.py                   # Core utility functions for data processing and evaluation
│   ├── Archived Tuners                # Archived tuning results and parameter files
│   ├── Strategies                     # Implementation of various trading strategies
│   ├── Strategy_Comparisons           # Notebooks comparing different strategies
│   └── __pycache__                    # Cache files for scripts
├── LICENSE
├── PROJECT_STRUCTURE.txt              # Detailed listing of the repository structure
└── README.md                          # This file
```

## Features

- **Advanced Tuning Framework:** Enhanced hyperparameter optimization with support for:
  - Walk-forward validation with configurable windows
  - Two-level parallel processing (strategies and parameters)
  - Customizable parameter grids
  - Strategy-specific tuner classes

- **Expanded Strategy Coverage:** 
  - Follow-the-Winner: Exponential Gradient, Follow-the-Leader, Follow-the-Regularized-Leader
  - Follow-the-Loser: Anticorrelation, PAMR, CWMR, OLMAR, RMR
  - Pattern-Matching: Histogram-based, Kernel-based, Nearest-neighbor selection methods
  - Meta-Learning: Online Gradient/Newton Updates, Fast Universalization
  - Hybrid approaches combining mean reversion and momentum strategies

- **Performance Optimization:**
  - Improved parallel processing with automatic CPU core allocation
  - Smart validation window sizing based on data characteristics
  - Enhanced error handling and recovery for long-running tuning sessions

- **Extended Metrics:**
  - Comprehensive evaluation including Sharpe ratio, maximum drawdown, exponential growth
  - Average validation metrics across time windows
  - Runtime performance tracking

- **Modular Design:** Clean separation between data processing, algorithm implementation, and performance evaluation.
- **Interactive Notebooks:** Detailed Jupyter notebooks (e.g., `Active_Tuner.ipynb`, various strategy comparisons) for hands-on experimentation.
- **Robust Performance Metrics:** Computes cumulative wealth, exponential growth rate, and Sharpe ratio for a standardized evaluation of trading methods.
- **Hyperparameter Optimization:** Incorporates grid search for fine-tuning strategy parameters to maximize key performance metrics.
- **Extensive Documentation:** Includes a research paper detailing the framework’s architecture, methodologies, and empirical findings.

## Installation and Dependencies

This project requires Python 3.13. To install all dependencies:

```bash
pip install -r Scripts/requirements.txt
```

**Core Dependencies:**
- numpy>=1.24.0
- pandas>=2.0.0
- scipy>=1.10.0
- joblib>=1.3.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- tqdm>=4.65.0

**Optional Dependencies:**
- jupyter>=1.0.0 (for running notebooks)
- pytest>=7.0.0 (for running tests)

If additional dependencies are needed, please update the `requirements.txt` accordingly.

## Usage

### 1. Data Preparation
Ensure price data is available in CSV format in the Data folder. The data pipeline includes:
- Historical price data preprocessing (`Price_Relative_Vector_Creation.ipynb`)
- Automated data validation and cleaning
- Support for various data sources and formats

### 2. Strategy Configuration
The framework offers multiple ways to configure and tune strategies:

a) Single Strategy Tuning:
```bash
python Scripts/run_hyperparameter_tuning.py strategy_name [options]
```

b) Multiple Strategy Tuning:
```bash
python Scripts/tune_multiple_strategies.py --strategies strategy1 strategy2 [options]
```

c) Category-based Tuning:
```bash
python Scripts/tune_multiple_strategies.py --category [ftw|ftl|pm|meta] [options]
```

Common options:
- `--walk-forward`: Enable walk-forward validation
- `--val-windows N`: Set number of validation windows
- `--val-size X`: Set validation window size (0-1)
- `--parallel-strategies`: Enable parallel strategy tuning
- `--n-jobs N`: Set number of parallel jobs

### 3. Performance Analysis
Results are saved in `Data/Tuning Data/` with detailed metrics for each parameter combination.

## Documentation and Research Paper

For an in-depth explanation of the project’s methodology, experiments, and findings, please refer to the research paper included in the [Docs](Docs/) folder. The paper presents detailed insights into:
- The experimental setup and data sources.
- A thorough evaluation and comparison of the various online portfolio selection strategies.
- Discussion on the trade-offs between performance and computational cost.

## Contributing

Contributions to improve the framework, extend its functionality, or refine the existing algorithms are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and submit a pull request.
4. Follow the coding conventions used in the repository.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Contact

For any questions or further details, please reach out via the contact emails provided in the research paper or open an issue in the repository.

---

We hope this framework serves as a valuable resource for research, development, and application in online quantitative trading. Happy coding and trading!


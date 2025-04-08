# Online Quantitative Trading Strategies

A comprehensive Python-based framework for evaluating and comparing online portfolio selection methods. This repository provides a robust environment for exploring a range of quantitative trading strategies—from momentum-based Follow-the-Winner approaches and mean-reversion-based Follow-the-Loser techniques to advanced pattern-matching and meta-learning ensembles—all benchmarked against traditional trading strategies.

## Overview

This project implements a unified trading infrastructure for online portfolio selection. The framework covers the entire pipeline, including:
- **Data Processing:** Extracting and preparing historical price data from CSV files to compute price-relative vectors.
- **Portfolio Initialization and Management:** Functions for uniform portfolio allocation and dynamic rebalancing.
- **Algorithm Implementation:** Multiple algorithm families such as Follow-the-Winner (e.g., Exponential Gradient, Follow-the-Leader, Follow-the-Regularized-Leader), Follow-the-Loser (e.g., Anticorrelation, PAMR, CWMR, OLMAR, RMR), Pattern-Matching approaches, and Meta-Learning ensembles.
- **Performance Evaluation:** Tools to assess performance metrics including cumulative wealth, exponential growth rate, and the Sharpe ratio, with extensive hyperparameter tuning via grid search.

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


```markdown
## Features

- **Modular Design:** Clean separation between data processing, algorithm implementation, and performance evaluation.
- **Extensive Strategy Coverage:** Implements diverse strategies from momentum to mean-reversion, as well as ensemble meta-learning methods.
- **Interactive Notebooks:** Detailed Jupyter notebooks (e.g., `Active_Tuner.ipynb`, various strategy comparisons) for hands-on experimentation.
- **Robust Performance Metrics:** Computes cumulative wealth, exponential growth rate, and Sharpe ratio for a standardized evaluation of trading methods.
- **Hyperparameter Optimization:** Incorporates grid search for fine-tuning strategy parameters to maximize key performance metrics.
- **Extensive Documentation:** Includes a research paper detailing the framework’s architecture, methodologies, and empirical findings.

## Installation and Dependencies

This project requires Python 3.x. The core dependencies are listed in the `requirements.txt` file located in the Scripts folder. To install the necessary packages, run:

```bash
pip install -r Scripts/requirements.txt

**Dependencies include (but may not be limited to):**
- numpy
- pandas
- scipy
- joblib
- matplotlib (for charting in notebooks)

If additional dependencies are needed, please update the `requirements.txt` accordingly.

## Usage

1. **Data Preparation:**  
   Ensure that the Data folder contains the expected CSV files for price data. The script `utilities.py` and the notebook `Price_Relative_Vector_Creation.ipynb` handle the extraction and preprocessing of the historical price data.

2. **Running the Notebooks:**  
   Use Jupyter Notebook or JupyterLab to open and run notebooks such as:
   - `Active_Tuner.ipynb` for hyperparameter tuning.
   - `FTL_Strategy_Comparison.ipynb`, `FTW_Strategy_Comparison.ipynb`, `ML_Strategy_Comparison.ipynb`, and `PM_Strategy_Comparison.ipynb` for interactive strategy comparisons.
   
   These notebooks provide step-by-step details on how each strategy is executed and compared.

3. **Evaluating Performance:**  
   The framework computes several performance metrics for each strategy including cumulative wealth, exponential growth rate, and the Sharpe ratio. You can customize these evaluations by modifying the parameters in the notebooks and utility functions.

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


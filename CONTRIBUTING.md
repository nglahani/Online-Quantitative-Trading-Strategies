# Contributing to Online-Quantitative-Trading-Strategies

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Implementing new strategies

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. Make your changes and ensure tests pass
3. Update documentation to reflect your changes
4. Submit a pull request

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Write docstrings for all public modules, functions, classes, and methods
- Keep functions focused and single-purpose
- Use descriptive variable names

### Documentation

- Keep READMEs updated
- Document complex algorithms and mathematical concepts
- Include references to research papers when applicable
- Add comments for non-obvious code sections

## Adding New Strategies

1. **Implementation**
   - Add strategy class in appropriate module under `Strategies/`
   - Follow existing pattern for strategy interface
   - Include proper documentation and references

2. **Integration**
   ```python
   # In run_hyperparameter_tuning.py
   strategy_map = {
       'your_strategy': your_strategy_function
   }

   # Add default parameter grid
   default_grids = {
       'your_strategy': {
           'param1': [value1, value2],
           'param2': [value3, value4]
       }
   }
   ```

3. **Testing**
   - Add unit tests
   - Include performance benchmarks
   - Document test cases

4. **Documentation**
   - Update relevant README files
   - Add tuning guidelines
   - Include usage examples

## Testing Guidelines

1. **Unit Tests**
   - Test each strategy component independently
   - Verify parameter validation
   - Check error handling
   - Test edge cases

2. **Integration Tests**
   - Test full strategy workflow
   - Verify tuning process
   - Check multi-strategy interactions

3. **Performance Tests**
   - Benchmark against known datasets
   - Test with various market conditions
   - Verify resource usage

## Environment Setup

1. **Required Tools**
   - Python 3.13+
   - Git
   - Virtual environment tool (e.g., venv, conda)

2. **Installation**
   ```bash
   # Clone repository
   git clone <repo-url>
   cd Online-Quantitative-Trading-Strategies

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows

   # Install dependencies
   pip install -r Scripts/requirements.txt
   ```

## Pull Request Process

1. Update documentation with details of changes
2. Update the CHANGES.md file
3. Include test results and benchmarks
4. Link any related issues

## Reporting Bugs

1. **Use the GitHub Issue Tracker**

2. **Bug Report Format**
   - Detailed description of the issue
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment details
   - Relevant logs or screenshots

## Feature Requests

1. **Use the GitHub Issue Tracker**

2. **Feature Request Format**
   - Clear description of the feature
   - Use cases and benefits
   - Proposed implementation approach
   - Related research or references

## Code Review Process

1. At least one project maintainer must review all code changes
2. Changes must include appropriate tests
3. Documentation must be updated
4. All CI checks must pass

## Community

- Be respectful and constructive
- Focus on what is best for the community
- Welcome newcomers and help them get started

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
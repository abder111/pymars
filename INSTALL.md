# PyMARS Installation and Setup Guide

Complete guide for installing, testing, and distributing the PyMARS package.

## Quick Install (Development Mode)

```bash
# Clone the repository
git clone https://github.com/yourusername/pymars.git
cd pymars

# Install in editable mode
pip install -e .
```

## Full Installation Options

### 1. Basic Installation

Installs core dependencies only (numpy, scipy):

```bash
pip install -e .
```

### 2. Development Installation

Includes testing and code quality tools:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest (testing)
- pytest-cov (coverage)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

### 3. With Plotting Support

```bash
pip install -e ".[plot]"
```

Adds matplotlib for visualization functions.

### 4. Complete Installation

Everything including documentation tools:

```bash
pip install -e ".[dev,plot,docs]"
```

## Verifying Installation

Test that PyMARS is correctly installed:

```python
python -c "from pymars import MARS; print('PyMARS installed successfully!')"
```

## Running Tests

### Run all tests

```bash
cd pymars
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_mars.py -v
```

### Run with coverage report

```bash
pytest tests/ --cov=pymars --cov-report=html
```

Open `htmlcov/index.html` to view coverage report.

### Run single test

```bash
pytest tests/test_mars.py::TestMARS::test_fit_predict -v
```

## Running Examples

```bash
cd examples
python demo_regression.py
```

Follow the interactive prompts to run different examples.

## Code Quality Checks

### Format code with Black

```bash
black pymars/ tests/
```

### Check style with flake8

```bash
flake8 pymars/ tests/ --max-line-length=88
```

### Type checking with mypy

```bash
mypy pymars/
```

## Building Distribution

### Build source and wheel distributions

```bash
pip install build
python -m build
```

This creates:
- `dist/pymars-0.1.0.tar.gz` (source distribution)
- `dist/pymars-0.1.0-py3-none-any.whl` (wheel distribution)

### Check distribution

```bash
pip install twine
twine check dist/*
```

## Publishing to PyPI

### Test on TestPyPI first

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ pymars
```

### Publish to PyPI

```bash
twine upload dist/*
```

After publishing, users can install with:

```bash
pip install pymars
```

## Project Structure Overview

```
pymars/
│
├── pymars/                    # Main package
│   ├── __init__.py           # Package initialization
│   ├── mars.py               # Main MARS class
│   ├── basis.py              # Basis functions
│   ├── model.py              # Forward/backward algorithms
│   ├── gcv.py                # Model selection
│   ├── utils.py              # Utility functions
│   ├── plots.py              # Visualization (optional)
│   └── interactions.py       # Interaction analysis (optional)
│
├── tests/                     # Test suite
│   ├── test_mars.py          # Main tests
│   ├── test_basis.py         # Basis function tests
│   └── test_gcv.py           # GCV tests
│
├── examples/                  # Example scripts
│   └── demo_regression.py    # Interactive demos
│
├── README.md                  # Main documentation
├── INSTALL.md                 # This file
├── LICENSE                    # MIT license
├── pyproject.toml            # Modern build configuration
├── requirements.txt          # Dependencies
└── MANIFEST.in               # Distribution files
```

## Dependencies

### Core (required)
- Python >= 3.8
- numpy >= 1.19.0
- scipy >= 1.5.0

### Optional
- matplotlib >= 3.3.0 (for plotting)
- pytest >= 7.0.0 (for testing)

## Troubleshooting

### Import Error

If you get `ModuleNotFoundError: No module named 'pymars'`:

1. Make sure you installed in editable mode: `pip install -e .`
2. Check your Python environment: `which python`
3. Verify installation: `pip list | grep pymars`

### NumPy/SciPy Issues

If you have problems installing numpy or scipy:

```bash
# On Ubuntu/Debian
sudo apt-get install python3-dev

# On macOS with Homebrew
brew install openblas

# Then reinstall
pip install --upgrade numpy scipy
```

### Test Failures

If tests fail:

1. Check Python version: `python --version` (must be >= 3.8)
2. Update dependencies: `pip install --upgrade -r requirements.txt`
3. Clear cache: `pytest --cache-clear`

## Development Workflow

### 1. Create feature branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make changes

Edit code in `pymars/`

### 3. Add tests

Add tests in `tests/`

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Format code

```bash
black pymars/ tests/
```

### 6. Commit changes

```bash
git add .
git commit -m "Add your feature"
```

### 7. Push and create PR

```bash
git push origin feature/your-feature-name
```

## Uninstalling

```bash
pip uninstall pymars
```

## Getting Help

- GitHub Issues: https://github.com/abder111/pymars/issues
- GitHub Repository: https://github.com/abder111/pymars
- Documentation: https://pymars.readthedocs.io

## Contributing

See CONTRIBUTING.md for guidelines on:
- Code style
- Testing requirements
- Documentation standards
- Pull request process

## License

PyMARS is released under the MIT License. See LICENSE file for details.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This repository contains materials for an Inventory Management course, organized as follows:

- **Exam/**: Contains exam preparation materials
  - `Exam.ipynb`: Jupyter notebook with sample exam questions and solutions
  - `formulas.py`: Python module with key inventory management formulas
  - `Sample Final Exam.pdf`: PDF document with sample exam
  - `studocu/`: Folder with additional exam preparation materials

- **Exercises/**: Contains exercise assignments and their solutions
  - `Ex1Solution/` to `Ex6Solution/`: Folders containing:
    - Assignment PDFs with problem statements
    - Solution PDFs with worked answers
    - Jupyter notebooks with Python implementations
    - Excel spreadsheets with calculations

- **Folien/**: Contains lecture slides in PDF format covering various inventory management topics

## Key Python Libraries Used

Based on the notebooks, the following Python libraries are commonly used:

1. **Core Libraries**:
   - `numpy`: For numerical operations and array handling
   - `math`: For basic mathematical functions

2. **Statistical Libraries**:
   - `scipy.stats`: Specifically for probability distributions (normal, poisson, gamma)
   - `scipy.integrate`: For numerical integration
   - `scipy.optimize`: For optimization functions

3. **Visualization**:
   - `matplotlib.pyplot`: For creating plots and visualizations

4. **Optimization (Optional)**:
   - `gurobipy`: For mathematical programming and optimization models

## Common Tasks

### Working with Notebooks

To run Jupyter notebooks in this repository:

```bash
# Navigate to the notebook directory
cd Exam/
# Or
cd Exercises/Ex1Solution/

# Launch Jupyter notebook
jupyter notebook
```

### Key Inventory Management Formulas

The `formulas.py` module in the `Exam/` directory contains implementations of key inventory management formulas:

- EOQ (Economic Order Quantity)
- Reorder Point (ROP)
- Newsvendor critical ratio and order quantities
- Safety stock calculations
- Exponential smoothing forecast
- Order-up-to level
- Common replenishment cycle
- XYZ classification

You can import these functions in notebooks:

```python
from formulas import eoq, reorder_point, newsvendor_critical_ratio, safety_stock
```

## Architecture and Workflows

The repository doesn't contain a traditional software architecture but rather educational materials following these patterns:

1. **Theoretical Problem Solving**:
   - Problems are presented in PDF format
   - Solutions are implemented in Python using Jupyter notebooks and Excel

2. **Common Inventory Management Models**:
   - EOQ (Economic Order Quantity) models
   - Newsvendor models for uncertain demand
   - Multi-period inventory models
   - Forecasting methods
   - Classification systems (like XYZ analysis)

3. **Optimization Techniques**:
   - Simple closed-form solutions
   - Numerical optimization methods
   - Mathematical programming (via Gurobi if available)

## Environment Setup Recommendations

To work with this repository, you'll need:

1. Python with these packages:
   - numpy
   - scipy
   - matplotlib
   - jupyter

2. Optional (for some optimization examples):
   - gurobipy (Gurobi Python API)

Install the required packages:

```bash
pip install numpy scipy matplotlib jupyter
```
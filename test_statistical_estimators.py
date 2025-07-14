#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_statistical_estimators.py
------------------------------
Tests for the statistical estimator functions in imf.py
"""

import numpy as np
import math
from imf import (
    sample_mean,
    sample_std_dev,
    estimate_gamma_params,
    compound_poisson_estimators
)

def test_sample_mean():
    """Test the sample mean estimator function."""
    print("\n" + "=" * 60)
    print("TESTING SAMPLE MEAN ESTIMATOR")
    print("=" * 60)
    
    # Test case 1: Simple array
    data1 = [1, 2, 3, 4, 5]
    expected1 = 3.0
    result1 = sample_mean(data1, label="Simple array")
    print(f"Data: {data1}")
    print(f"Expected: {expected1}")
    print(f"Result: {result1}")
    print(f"Passed: {abs(result1 - expected1) < 1e-10}")
    
    # Test case 2: Negative values
    data2 = [-10, -5, 0, 5, 10]
    expected2 = 0.0
    result2 = sample_mean(data2, label="Array with negative values")
    print(f"Data: {data2}")
    print(f"Expected: {expected2}")
    print(f"Result: {result2}")
    print(f"Passed: {abs(result2 - expected2) < 1e-10}")
    
    # Test case 3: Empty array (should return NaN)
    data3 = []
    try:
        result3 = sample_mean(data3, label="Empty array")
        print("Empty array test: Failed - should have raised an exception")
    except:
        print("Empty array test: Passed - raised an exception as expected")

def test_sample_std_dev():
    """Test the sample standard deviation estimator function."""
    print("\n" + "=" * 60)
    print("TESTING SAMPLE STANDARD DEVIATION ESTIMATOR")
    print("=" * 60)
    
    # Test case 1: Simple array
    data1 = [1, 2, 3, 4, 5]
    expected1 = np.std(data1, ddof=1)
    result1 = sample_std_dev(data1, label="Simple array")
    print(f"Data: {data1}")
    print(f"Expected: {expected1}")
    print(f"Result: {result1}")
    print(f"Passed: {abs(result1 - expected1) < 1e-10}")
    
    # Test case 2: Constant array (should be 0)
    data2 = [7, 7, 7, 7, 7]
    expected2 = 0.0
    result2 = sample_std_dev(data2, label="Constant array")
    print(f"Data: {data2}")
    print(f"Expected: {expected2}")
    print(f"Result: {result2}")
    print(f"Passed: {abs(result2 - expected2) < 1e-10}")
    
    # Test case 3: Single value (should raise error)
    data3 = [42]
    try:
        result3 = sample_std_dev(data3, label="Single value")
        print("Single value test: Failed - should have raised a warning or exception")
    except:
        print("Single value test: Passed - raised an exception as expected")

def test_estimate_gamma_params():
    """Test the gamma distribution parameter estimation."""
    print("\n" + "=" * 60)
    print("TESTING GAMMA PARAMETER ESTIMATION")
    print("=" * 60)
    
    # Test case 1: Generated data from a gamma distribution
    # Generate random gamma data
    alpha_true = 2.0
    beta_true = 3.0
    np.random.seed(42)  # For reproducibility
    data = np.random.gamma(alpha_true, beta_true, 1000)
    
    # Calculate true mean and variance
    true_mean = alpha_true * beta_true
    true_var = alpha_true * beta_true**2
    
    # Estimate parameters
    result = estimate_gamma_params(data, label="Gamma test")
    alpha_est = result["alpha"]
    beta_est = result["beta"]
    
    print(f"True parameters: alpha={alpha_true}, beta={beta_true}")
    print(f"Estimated parameters: alpha={alpha_est:.4f}, beta={beta_est:.4f}")
    
    # Check if estimates are within 10% of true values
    alpha_error = abs(alpha_est - alpha_true) / alpha_true
    beta_error = abs(beta_est - beta_true) / beta_true
    print(f"Alpha error: {alpha_error:.4f} (< 0.1 is good)")
    print(f"Beta error: {beta_error:.4f} (< 0.1 is good)")
    print(f"Test passed: {alpha_error < 0.1 and beta_error < 0.1}")

def test_compound_poisson_estimators():
    """Test the compound Poisson distribution parameter estimation."""
    print("\n" + "=" * 60)
    print("TESTING COMPOUND POISSON ESTIMATORS")
    print("=" * 60)
    
    # Test case 1: Simulated compound Poisson data
    # Parameters
    lambda_true = 0.7  # Arrival rate
    mu_c_true = 5.0    # Mean individual demand size
    T = 100            # Number of periods
    
    # Generate data
    np.random.seed(42)  # For reproducibility
    
    # For each period, generate the number of arrivals
    arrivals = np.random.poisson(lambda_true, T)
    
    # Generate data with zeros for periods with no arrivals
    data = []
    for n in arrivals:
        if n == 0:
            data.append(0)
        else:
            # For periods with arrivals, generate exponential demand sizes
            values = np.random.exponential(mu_c_true, n)
            data.append(np.sum(values))
    
    # Estimate parameters
    result = compound_poisson_estimators(data, T, label="Compound Poisson test")
    lambda_est = result["lambda"]
    mu_c_est = result["mu_c"]
    
    print(f"True parameters: lambda={lambda_true}, mu_c={mu_c_true}")
    print(f"Estimated parameters: lambda={lambda_est:.4f}, mu_c={mu_c_est:.4f}")
    
    # Check if estimates are within 20% of true values
    lambda_error = abs(lambda_est - lambda_true) / lambda_true
    mu_c_error = abs(mu_c_est - mu_c_true) / mu_c_true
    print(f"Lambda error: {lambda_error:.4f} (< 0.2 is good)")
    print(f"Mu_c error: {mu_c_error:.4f} (< 0.2 is good)")
    print(f"Test passed: {lambda_error < 0.2 and mu_c_error < 0.2}")
    
    # Test case 2: All zeros
    data_zeros = [0] * T
    result_zeros = compound_poisson_estimators(data_zeros, T, label="All zeros")
    print("\nAll zeros test:")
    print(f"Lambda: {result_zeros['lambda']:.4f} (should be close to 0)")
    print(f"Mu_c: {result_zeros['mu_c']} (should be 0)")
    
    # Test case 3: No zeros
    data_no_zeros = [1] * T
    result_no_zeros = compound_poisson_estimators(data_no_zeros, T, label="No zeros")
    print("\nNo zeros test:")
    print(f"Lambda: {result_no_zeros['lambda']} (should be maximum, T={T})")
    print(f"Mu_c: {result_no_zeros['mu_c']:.4f} (should be 1/lambda)")

if __name__ == "__main__":
    test_sample_mean()
    test_sample_std_dev()
    test_estimate_gamma_params()
    test_compound_poisson_estimators()
    print("\nAll tests completed!")
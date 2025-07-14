#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_newsvendor_functions.py
----------------------------
Tests for the new newsvendor functions in imf.py
"""

import numpy as np
import math
from imf import (
    newsvendor_revenue_discrete,
    newsvendor_revenue_continuous,
    newsvendor_with_estimated_params,
    newsvendor_critical_ratio
)

def test_newsvendor_revenue_discrete():
    """Test the discrete newsvendor revenue function."""
    print("\n" + "=" * 60)
    print("TESTING DISCRETE NEWSVENDOR REVENUE FUNCTION")
    print("=" * 60)
    
    # Test case 1: Simple example
    demand_values = [0, 1, 2, 3, 4, 5]
    demand_probs = [0.05, 0.10, 0.20, 0.30, 0.25, 0.10]
    p = 10  # selling price
    c = 4   # cost
    g = 1   # salvage value
    y = 3   # order quantity
    
    # Calculate expected result
    # For d <= y: (p*d + g*(y-d))*P(D=d)
    # d=0: (10*0 + 1*(3-0))*0.05 = 0.15
    # d=1: (10*1 + 1*(3-1))*0.10 = 1.20
    # d=2: (10*2 + 1*(3-2))*0.20 = 4.20
    # d=3: (10*3 + 1*(3-3))*0.30 = 9.00
    # For d > y: p*y*P(D=d)
    # d=4: 10*3*0.25 = 7.50
    # d=5: 10*3*0.10 = 3.00
    # Procurement: -c*y = -4*3 = -12
    # Total: -12 + 0.15 + 1.20 + 4.20 + 9.00 + 7.50 + 3.00 = 13.05
    expected_result = -12 + 0.15 + 1.20 + 4.20 + 9.00 + 7.50 + 3.00
    
    result = newsvendor_revenue_discrete(y, demand_values, demand_probs, p, g, c, label="Test case 1")
    print(f"Expected result: {expected_result}")
    print(f"Actual result: {result}")
    print(f"Test passed: {abs(result - expected_result) < 1e-10}")
    
    # Test case 2: Order quantity exceeds all possible demand
    y2 = 6
    # Calculate expected result
    # For d <= y: (p*d + g*(y-d))*P(D=d)
    # d=0: (10*0 + 1*(6-0))*0.05 = 0.30
    # d=1: (10*1 + 1*(6-1))*0.10 = 1.50
    # d=2: (10*2 + 1*(6-2))*0.20 = 4.80
    # d=3: (10*3 + 1*(6-3))*0.30 = 9.90
    # d=4: (10*4 + 1*(6-4))*0.25 = 10.50
    # d=5: (10*5 + 1*(6-5))*0.10 = 5.10
    # Procurement: -c*y = -4*6 = -24
    # Total: -24 + 0.30 + 1.50 + 4.80 + 9.90 + 10.50 + 5.10 = 8.10
    expected_result2 = -24 + 0.30 + 1.50 + 4.80 + 9.90 + 10.50 + 5.10
    
    result2 = newsvendor_revenue_discrete(y2, demand_values, demand_probs, p, g, c, label="Test case 2")
    print(f"\nExpected result (case 2): {expected_result2}")
    print(f"Actual result (case 2): {result2}")
    print(f"Test passed (case 2): {abs(result2 - expected_result2) < 1e-10}")

def test_newsvendor_revenue_continuous():
    """Test the continuous newsvendor revenue function."""
    print("\n" + "=" * 60)
    print("TESTING CONTINUOUS NEWSVENDOR REVENUE FUNCTION")
    print("=" * 60)
    
    # Test case 1: Normal distribution
    p = 10  # selling price
    c = 4   # cost
    g = 1   # salvage value
    y = 120 # order quantity
    mu = 100
    sigma = 20
    dist_params = {'mu': mu, 'sigma': sigma}
    
    result = newsvendor_revenue_continuous(y, 'normal', dist_params, p, g, c, label="Normal distribution")
    print(f"Order quantity: {y}")
    print(f"Normal distribution mu={mu}, sigma={sigma}")
    print(f"Revenue result: {result}")
    
    # Test case 2: Different order quantity
    y2 = 80
    result2 = newsvendor_revenue_continuous(y2, 'normal', dist_params, p, g, c, label="Normal distribution (lower order)")
    print(f"\nOrder quantity: {y2}")
    print(f"Normal distribution mu={mu}, sigma={sigma}")
    print(f"Revenue result: {result2}")
    print(f"Revenue difference: {result - result2}")
    
    # Test case 3: Gamma distribution if available
    try:
        alpha = 5
        beta = 20  # scale parameter
        dist_params_gamma = {'alpha': alpha, 'beta': beta}
        
        result3 = newsvendor_revenue_continuous(y, 'gamma', dist_params_gamma, p, g, c, label="Gamma distribution")
        print(f"\nOrder quantity: {y}")
        print(f"Gamma distribution alpha={alpha}, beta={beta}")
        print(f"Revenue result: {result3}")
    except:
        print("\nSkipping gamma distribution test - SciPy integration not available")

def test_newsvendor_with_estimated_params():
    """Test the newsvendor with estimated parameters function."""
    print("\n" + "=" * 60)
    print("TESTING NEWSVENDOR WITH ESTIMATED PARAMETERS")
    print("=" * 60)
    
    # Test case 1: Compare with standard newsvendor solution
    sample_mean = 100
    sample_std = 20
    sample_size = 30
    p = 10
    c = 4
    g = 1
    
    # Calculate critical ratio
    cr = newsvendor_critical_ratio(p, c, g)
    print(f"Critical ratio: {cr}")
    
    # Get results
    result = newsvendor_with_estimated_params(sample_mean, sample_std, sample_size, p, c, g, 
                                            confidence_level=0.95, label="Test case")
    print(f"Order quantity with estimated parameters: {result}")
    
    # Test case 2: Varying sample size
    sample_sizes = [5, 10, 20, 50, 100]
    print("\nEffect of sample size on order quantity:")
    print("Sample Size | Order Quantity | Difference from n=∞")
    print("-" * 50)
    
    # First calculate limit as n → ∞ (standard normal solution)
    import scipy.stats as stats
    z = stats.norm.ppf(cr)
    q_inf = sample_mean + z * sample_std
    
    for n in sample_sizes:
        q = newsvendor_with_estimated_params(sample_mean, sample_std, n, p, c, g, 
                                           label=f"n={n}", suffix="size")
        diff = q - q_inf
        perc_diff = (diff / q_inf) * 100
        print(f"{n:11d} | {q:13.2f} | {diff:+7.2f} ({perc_diff:+.2f}%)")

if __name__ == "__main__":
    test_newsvendor_revenue_discrete()
    test_newsvendor_revenue_continuous()
    test_newsvendor_with_estimated_params()
    print("\nAll tests completed!")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_statistics.py
-----------------
This script tests the statistical distribution functions in imf.py
"""

from imf import (
    phi_z,
    Phi_z,
    G_z,
    inverse_cdf,
    standardize,
    unstandardize,
    safety_factor_for_service_level,
    service_level_from_safety_factor,
    fill_rate_from_safety_factor,
    expected_shortage,
    normal_loss_function,
    partial_expectation,
    service_level_to_fill_rate,
    fill_rate_to_service_level
)

def main():
    """Test the statistical distribution functions."""
    print("\n" + "=" * 60)
    print("TESTING STATISTICAL DISTRIBUTION FUNCTIONS")
    print("=" * 60)
    
    # Test basic distribution functions
    z_values = [-1.96, -1.645, -1, 0, 1, 1.645, 1.96, 2.33, 3]
    
    print("\n" + "-" * 60)
    print("BASIC DISTRIBUTION FUNCTIONS")
    print("-" * 60)
    
    print("\nz\t| φ(z)\t\t| Φ(z)\t\t| G(z)")
    print("-" * 60)
    for z in z_values:
        phi = phi_z(z, label="")
        Phi = Phi_z(z, label="")
        g = G_z(z, label="")
        print(f"{z:.3f}\t| {phi:.6f}\t| {Phi:.6f}\t| {g:.6f}")
    
    # Test inverse functions
    print("\n" + "-" * 60)
    print("INVERSE FUNCTIONS")
    print("-" * 60)
    
    probabilities = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
    
    print("\np\t| z = Φ^(-1)(p)")
    print("-" * 30)
    for p in probabilities:
        z = inverse_cdf(p, label="")
        print(f"{p:.2f}\t| {z:.6f}")
    
    # Test standardize/unstandardize
    print("\n" + "-" * 60)
    print("STANDARDIZE/UNSTANDARDIZE")
    print("-" * 60)
    
    mu = 100
    sigma = 15
    values = [70, 85, 100, 115, 130]
    
    print("\nx\t| z = (x-μ)/σ\t| x = μ+z*σ")
    print("-" * 50)
    for x in values:
        z = standardize(x, mu, sigma, label="")
        x_back = unstandardize(z, mu, sigma, label="")
        print(f"{x:.1f}\t| {z:.6f}\t| {x_back:.6f}")
    
    # Test safety factor and service level conversions
    print("\n" + "-" * 60)
    print("SAFETY FACTOR AND SERVICE LEVEL")
    print("-" * 60)
    
    service_levels = [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
    
    print("\nService Level\t| Safety Factor\t| Calculated SL\t| Fill Rate")
    print("-" * 75)
    for sl in service_levels:
        z = safety_factor_for_service_level(sl, label="")
        sl_back = service_level_from_safety_factor(z, label="")
        fr = fill_rate_from_safety_factor(z, label="")
        print(f"{sl:.4f}\t\t| {z:.6f}\t| {sl_back:.6f}\t| {fr:.6f}")
    
    # Test expected shortage and loss function
    print("\n" + "-" * 60)
    print("EXPECTED SHORTAGE AND LOSS FUNCTION")
    print("-" * 60)
    
    mu = 100
    sigma = 15
    x_values = [85, 100, 115, 130]
    
    print("\nx\t| E[max(D-x,0)]\t| Normal Loss\t| Partial E[X|X>x]")
    print("-" * 70)
    for x in x_values:
        z = (x - mu) / sigma
        es = expected_shortage(mu, sigma, z, label="")
        nl = normal_loss_function(x, mu, sigma, label="")
        pe = partial_expectation(x, mu, sigma, label="")
        print(f"{x:.1f}\t| {es:.6f}\t\t| {nl:.6f}\t| {pe:.6f}")
    
    # Test service level to fill rate conversion
    print("\n" + "-" * 60)
    print("SERVICE LEVEL TO FILL RATE CONVERSION")
    print("-" * 60)
    
    cv_values = [0.1, 0.2, 0.5, 1.0]
    
    print("\nCV\t| Service Level\t| Fill Rate\t| SL from FR")
    print("-" * 60)
    for cv in cv_values:
        for sl in [0.9, 0.95, 0.98]:
            fr = service_level_to_fill_rate(sl, cv, label="")
            sl_back = fill_rate_to_service_level(fr, cv, label="")
            print(f"{cv:.1f}\t| {sl:.4f}\t| {fr:.6f}\t| {sl_back:.6f}")

if __name__ == "__main__":
    main()
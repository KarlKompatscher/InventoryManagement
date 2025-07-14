#!/usr/bin/env python3
"""
Solution for Assignment 4: Effect of Demand Distribution on Order Quantity
Using the imf_* formula library
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imf_main import (
    newsvendor_critical_ratio,
    newsvendor_uniform,
    newsvendor_poisson,
    newsvendor_gamma
)


def main():
    print("Assignment 4: Effect of Demand Distribution on Order Quantity")
    print("-----------------------------------------------------------")

    # Critical ratio for all distributions
    critical_ratio = 0.75

    # Exercise 4(a): Uniform Distribution
    print("\nExercise 4(a): Uniform Distribution")
    
    lower_bound = 4000
    upper_bound = 8000
    
    # Calculate optimal order quantity using the library function
    optimal_q_uniform = newsvendor_uniform(lower_bound, upper_bound, critical_ratio)
    print(f"Order quantity with uniformly distributed demand: {optimal_q_uniform:.0f} units")
    
    # Calculate mean and variance for uniform distribution
    uniform_mean = (lower_bound + upper_bound) / 2
    uniform_std = (upper_bound - lower_bound) / np.sqrt(12)
    print(f"Uniform mean: {uniform_mean:.0f}")
    print(f"Uniform standard deviation: {uniform_std:.2f}")
    
    # Exercise 4(b): Poisson Distribution
    print("\nExercise 4(b): Poisson Distribution")
    
    lambda_param = 6000
    
    # Calculate optimal order quantity using the library function
    optimal_q_poisson = newsvendor_poisson(lambda_param, critical_ratio)
    print(f"Order quantity with Poisson demand: {optimal_q_poisson:.0f} units")
    
    # Calculate mean and variance for Poisson distribution
    poisson_mean = lambda_param
    poisson_std = np.sqrt(lambda_param)
    print(f"Poisson mean: {poisson_mean:.0f}")
    print(f"Poisson standard deviation: {poisson_std:.2f}")
    
    # Exercise 4(c): Gamma Distribution
    print("\nExercise 4(c): Gamma Distribution")
    
    gamma_mean = 6000
    gamma_std = uniform_std  # Using same std as uniform for comparison
    
    # Calculate optimal order quantity using the library function
    optimal_q_gamma = newsvendor_gamma(gamma_mean, gamma_std, critical_ratio)
    print(f"Order quantity with Gamma demand: {optimal_q_gamma:.0f} units")
    
    # Calculate shape and scale parameters for gamma distribution
    gamma_var = gamma_std**2
    gamma_shape = gamma_mean**2 / gamma_var
    gamma_scale = gamma_var / gamma_mean
    print(f"Gamma shape parameter (α): {gamma_shape:.2f}")
    print(f"Gamma scale parameter (β): {gamma_scale:.2f}")
    
    # Create visualization for comparing different distributions
    visualize_distributions(
        critical_ratio,
        lower_bound, upper_bound, optimal_q_uniform,
        lambda_param, optimal_q_poisson,
        gamma_mean, gamma_std, gamma_shape, gamma_scale, optimal_q_gamma
    )


def visualize_distributions(critical_ratio, 
                           uni_low, uni_high, opt_uni,
                           poisson_lambda, opt_poisson,
                           gamma_mean, gamma_std, gamma_shape, gamma_scale, opt_gamma):
    """Create visualization comparing different demand distributions"""
    
    from scipy.stats import uniform, poisson, gamma
    
    # Create plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    
    # (a) Uniform distribution
    x_uni = np.arange(uni_low - 500, uni_high + 500, 100)
    axs[0].plot(x_uni, uniform.pdf(x_uni, loc=uni_low, scale=uni_high-uni_low), 'g-', label='Uniform PDF')
    axs[0].axvline(x=opt_uni, color='r', linestyle='--', label=f'Order quantity: {opt_uni:.0f}')
    axs[0].axvline(x=(uni_low + uni_high)/2, color='b', linestyle=':', label='Mean demand')
    axs[0].set_title(f'Uniform Distribution ({uni_low}-{uni_high})')
    axs[0].set_xlabel('Demand')
    axs[0].set_ylabel('Probability Density')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # (b) Poisson distribution
    x_poisson = np.arange(poisson.ppf(0.001, poisson_lambda), poisson.ppf(0.999, poisson_lambda), 10)
    poisson_pmf = poisson.pmf(x_poisson, poisson_lambda)
    axs[1].vlines(x_poisson, 0, poisson_pmf, colors='b', lw=2, alpha=0.7, label='Poisson PMF')
    axs[1].axvline(x=opt_poisson, color='r', linestyle='--', label=f'Order quantity: {opt_poisson:.0f}')
    axs[1].axvline(x=poisson_lambda, color='b', linestyle=':', label='Mean demand')
    axs[1].set_title(f'Poisson Distribution (λ = {poisson_lambda})')
    axs[1].set_xlabel('Demand')
    axs[1].set_ylabel('Probability Mass')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # (c) Gamma distribution
    x_gamma = np.linspace(gamma.ppf(0.001, gamma_shape, scale=gamma_scale),
                         gamma.ppf(0.999, gamma_shape, scale=gamma_scale), 1000)
    axs[2].plot(x_gamma, gamma.pdf(x_gamma, gamma_shape, scale=gamma_scale), 'r-', label='Gamma PDF')
    axs[2].axvline(x=opt_gamma, color='r', linestyle='--', label=f'Order quantity: {opt_gamma:.0f}')
    axs[2].axvline(x=gamma_mean, color='b', linestyle=':', label='Mean demand')
    axs[2].set_title(f'Gamma Distribution (α = {gamma_shape:.2f}, β = {gamma_scale:.2f})')
    axs[2].set_xlabel('Demand')
    axs[2].set_ylabel('Probability Density')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Add overall title
    plt.suptitle(f'Effect of Demand Distribution on Order Quantity (CR = {critical_ratio})', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig('assignment4_distributions.png')
    plt.close()
    
    # Create second plot comparing order quantities
    plt.figure(figsize=(10, 6))
    
    # Calculate means and order quantities for comparison
    means = [
        (uni_low + uni_high)/2,  # Uniform mean
        poisson_lambda,          # Poisson mean
        gamma_mean               # Gamma mean
    ]
    
    order_quantities = [opt_uni, opt_poisson, opt_gamma]
    
    # Calculate differences between order quantity and mean
    differences = [order_quantities[i] - means[i] for i in range(3)]
    
    # Create grouped bar chart
    bar_width = 0.35
    index = np.arange(3)
    
    plt.bar(index - bar_width/2, means, bar_width, label='Mean Demand', color='blue', alpha=0.6)
    plt.bar(index + bar_width/2, order_quantities, bar_width, label='Optimal Order Quantity', color='red', alpha=0.6)
    
    # Add labels
    plt.xlabel('Demand Distribution')
    plt.ylabel('Units')
    plt.title('Comparison of Mean Demand and Optimal Order Quantity Across Distributions')
    plt.xticks(index, ('Uniform', 'Poisson', 'Gamma'))
    
    # Add text labels for differences
    for i, diff in enumerate(differences):
        plt.annotate(f"+{diff:.0f}", 
                    xy=(index[i] + bar_width/2, means[i] + diff/2),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='center',
                    color='white', fontweight='bold')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('assignment4_comparison.png')
    plt.close()


if __name__ == "__main__":
    main()
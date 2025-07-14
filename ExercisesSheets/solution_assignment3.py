#!/usr/bin/env python3
"""
Solution for Assignment 3: Newsvendor Model
Using the imf_* formula library
"""

import numpy as np
import matplotlib.pyplot as plt
from imf_main import (
    newsvendor_critical_ratio,
    newsvendor_normal,
    NormalDistribution,
    expected_shortage
)


def main():
    print("Assignment 3: Newsvendor Model")
    print("------------------------------")

    # Exercise 3(a): Basic Newsvendor Model
    print("\nExercise 3(a): Basic Newsvendor Model")
    
    # Given data
    mean_demand = 2000  # units
    std_demand = 250  # units
    price = 6  # euro
    cost = 3  # euro
    salvage = 2.5  # euro
    
    # Calculate critical ratio using the library function
    critical_ratio = newsvendor_critical_ratio(price, cost, salvage)
    print(f"Critical ratio: {critical_ratio:.4f}")
    
    # Calculate optimal order quantity using the library function
    optimal_q = newsvendor_normal(mean_demand, std_demand, critical_ratio)
    print(f"Newsvendor quantity: {optimal_q:.2f} units")
    
    # Exercise 3(b): KPI calculations
    print("\nExercise 3(b): KPI calculations")
    
    # Calculate z-value
    z = (optimal_q - mean_demand) / std_demand
    print(f"Safety factor z: {z:.4f}")
    
    # Calculate expected lost sales
    els = expected_shortage(std_demand, z)
    print(f"Expected lost sales: {els:.2f} units")
    
    # Calculate expected sales
    expected_sales = mean_demand - els
    print(f"Expected sales: {expected_sales:.2f} units")
    
    # Calculate expected leftover
    expected_leftover = optimal_q - expected_sales
    print(f"Expected leftover: {expected_leftover:.2f} units")
    
    # Calculate expected profit
    expected_profit = -cost * optimal_q + price * expected_sales + salvage * expected_leftover
    print(f"Expected profit: {expected_profit:.2f} euro")
    
    # Calculate service level metrics
    service_level_availability = critical_ratio  # alpha = beta
    print(f"Service level (Availability): {service_level_availability:.4f}")
    
    fill_rate = expected_sales / mean_demand
    print(f"Service level (Fill-rate): {fill_rate:.4f}")
    
    # Create visualization of the newsvendor model solution
    visualize_newsvendor(mean_demand, std_demand, critical_ratio, optimal_q)


def visualize_newsvendor(mean_demand, std_demand, critical_ratio, optimal_q):
    """Create visualization of newsvendor model solution"""
    
    # Create range of demand values for plotting
    demand_values = np.linspace(mean_demand - 4 * std_demand, mean_demand + 4 * std_demand, 1000)
    
    # Calculate normal PDF values
    pdf_values = [NormalDistribution.pdf((x - mean_demand) / std_demand, label="", suffix="") / std_demand 
                 for x in demand_values]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot demand distribution
    plt.subplot(1, 2, 1)
    plt.plot(demand_values, pdf_values)
    plt.fill_between(demand_values[demand_values <= optimal_q], pdf_values[demand_values <= optimal_q], 
                    alpha=0.3, color='green', label='No stockout')
    plt.fill_between(demand_values[demand_values > optimal_q], pdf_values[demand_values > optimal_q], 
                    alpha=0.3, color='red', label='Stockout')
    
    # Mark key points
    plt.axvline(x=mean_demand, color='blue', linestyle='--', label='Mean demand')
    plt.axvline(x=optimal_q, color='green', linestyle='-', label='Optimal order quantity')
    
    # Add labels and legend
    plt.xlabel('Demand')
    plt.ylabel('Probability Density')
    plt.title(f'Newsvendor Solution (CR = {critical_ratio:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create a cost/profit plot
    plt.subplot(1, 2, 2)
    
    # Define range of order quantities
    q_values = np.linspace(mean_demand - 2 * std_demand, mean_demand + 2 * std_demand, 100)
    
    # Calculate expected profit for different order quantities
    # For each q, we need to calculate expected sales and leftover
    profits = []
    for q in q_values:
        z = (q - mean_demand) / std_demand
        els = NormalDistribution.expected_shortage(std_demand, z, label="", suffix="")
        exp_sales = mean_demand - els
        exp_leftover = q - exp_sales
        profit = -3 * q + 6 * exp_sales + 2.5 * exp_leftover
        profits.append(profit)
    
    plt.plot(q_values, profits)
    plt.axvline(x=optimal_q, color='green', linestyle='-', label='Optimal order quantity')
    plt.axhline(y=max(profits), color='red', linestyle='--', label='Maximum expected profit')
    
    # Add labels and legend
    plt.xlabel('Order Quantity')
    plt.ylabel('Expected Profit')
    plt.title('Expected Profit vs Order Quantity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('assignment3_newsvendor.png')
    plt.close()


if __name__ == "__main__":
    main()
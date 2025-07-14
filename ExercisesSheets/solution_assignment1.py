#!/usr/bin/env python3
"""
Solution for Assignment 1: EOQ Models and Sensitivity Analysis
Using the imf_* formula library
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imf_main import EOQ, eoq, cost_penalty


def main():
    print("Assignment 1: EOQ Models and Sensitivity Analysis")
    print("-------------------------------------------------")

    # Exercise 1(a): Basic EOQ calculation
    print("\nExercise 1(a): Basic EOQ calculation")

    # Given data
    demand = 600  # kg/week
    setup_cost = 26  # euro
    unit_cost = 1.35  # euro/kg
    annual_holding_cost = 2.2  # euro/kg/year
    holding_cost = annual_holding_cost / 52  # euro/kg/week (convert to weekly)

    # Calculate EOQ using the library function
    optimal_lot_size = eoq(demand, setup_cost, holding_cost)
    
    # Calculate total relevant cost
    total_cost = math.sqrt(2 * demand * setup_cost * holding_cost)
    
    # Calculate cycle time
    cycle_time = optimal_lot_size / demand

    print(f"Optimal lot size: {optimal_lot_size:.2f} kg")
    print(f"Total relevant cost: {total_cost:.2f} euros")
    print(f"Cycle length: {cycle_time:.2f} weeks")

    # Exercise 1(b): Sensitivity Analysis
    print("\nExercise 1(b): Sensitivity Analysis")
    
    actual_quantity = 500  # Given non-optimal order quantity
    
    # Calculate percentage deviation
    percentage_deviation = (actual_quantity - optimal_lot_size) / optimal_lot_size * 100
    
    # Calculate cost penalty using the library function
    penalty = cost_penalty(actual_quantity, optimal_lot_size)

    print(f"Percentage deviation: {percentage_deviation:.2f}%")
    print(f"Percentage cost penalty: {penalty:.2f}%")

    # Alternatively, calculate directly
    p = (actual_quantity - optimal_lot_size) / optimal_lot_size
    pcp = 50 * (p**2 / (1 + p))
    print(f"Direct calculation of cost penalty: {pcp:.2f}%")

    # Define function to calculate lot cost
    def lot_cost(q):
        return demand / q * setup_cost + 0.5 * holding_cost * q

    # Exercise 1(c): Power of two policy
    print("\nExercise 1(c): Power of two policy")
    
    t = 1
    while lot_cost(2 * demand * t) < lot_cost(demand * t):
        t = t * 2
    
    optimal_integer_cycle = t
    cost_error = 100 * (lot_cost(demand * t) / total_cost - 1)
    
    print(f"Optimal integer cycle: {optimal_integer_cycle}")
    print(f"Cost error: {cost_error:.2f}%")

    # Create visualization for EOQ and cost comparison
    visualize_eoq(demand, setup_cost, holding_cost, optimal_lot_size, actual_quantity)


def visualize_eoq(demand, setup_cost, holding_cost, optimal_q, actual_q):
    """Create visualization of EOQ components and cost curves"""
    
    # Define cost functions
    def ordering_cost(q):
        return setup_cost * demand / q
    
    def holding_cost_func(q):
        return 0.5 * holding_cost * q
    
    def total_cost_func(q):
        return ordering_cost(q) + holding_cost_func(q)
    
    # Create data for plotting
    q_values = np.linspace(100, 2000, 1000)
    ordering_costs = [ordering_cost(q) for q in q_values]
    holding_costs = [holding_cost_func(q) for q in q_values]
    total_costs = [total_cost_func(q) for q in q_values]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(q_values, ordering_costs, label='Ordering Cost')
    plt.plot(q_values, holding_costs, label='Holding Cost')
    plt.plot(q_values, total_costs, label='Total Cost')
    
    # Mark the optimal and actual quantities
    plt.axvline(x=optimal_q, color='g', linestyle='--', label=f'Optimal Q = {optimal_q:.2f}')
    plt.axvline(x=actual_q, color='r', linestyle='--', label=f'Actual Q = {actual_q}')
    
    # Add labels and legend
    plt.xlabel('Order Quantity')
    plt.ylabel('Cost')
    plt.title('EOQ Model Cost Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig('assignment1_eoq_visualization.png')
    plt.close()


if __name__ == "__main__":
    main()
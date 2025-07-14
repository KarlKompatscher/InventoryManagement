#!/usr/bin/env python3
"""
Solution for Assignment 2: Quantity Discount Models
Using the imf_* formula library
"""

import numpy as np
import matplotlib.pyplot as plt
from imf_main import eoq_all_unit_quantity_discount, eoq_incremental_quantity_discount


def main():
    print("Assignment 2: Quantity Discount Models")
    print("--------------------------------------")

    # Exercise 2(a): All-unit quantity discount model
    print("\nExercise 2(a): All-unit quantity discount model")
    
    # Given data
    demand = 40 * 52  # units/year
    setup_cost = 25  # euro
    interest_rate = 0.26  # holding cost as fraction of unit cost
    
    # Break points of order quantity and corresponding purchasing prices
    quantity_breaks = [0, 300, 500]
    unit_costs = [10, 9.7, 9.5]
    
    # Calculate optimal order quantity using the library function
    result = eoq_all_unit_quantity_discount(
        demand=demand,
        setup_cost=setup_cost,
        unit_costs=unit_costs,
        holding_rate=interest_rate,
        quantity_breaks=quantity_breaks
    )
    
    print(f"Optimal order quantity: {result['order_quantity']:.2f} units")
    print(f"Optimal unit cost: {result['unit_cost']:.2f} euro")
    print(f"Total annual cost: {result['total_cost']:.2f} euro")
    print(f"Ordering cost: {result['ordering_cost']:.2f} euro")
    print(f"Holding cost: {result['holding_cost']:.2f} euro")
    print(f"Purchase cost: {result['purchase_cost']:.2f} euro")
    
    # Exercise 2(b): Incremental quantity discount model
    print("\nExercise 2(b): Incremental quantity discount model")
    
    # Given data (modified for incremental quantity discount)
    demand = 40 * 52  # units/year
    setup_cost = 25  # euro
    interest_rate = 0.26  # holding cost as fraction of unit cost
    
    # Simplified case with just two price points for incremental discount
    quantity_breaks = [300]
    unit_costs = [10, 9.7]
    
    # Calculate optimal order quantity using the library function
    result = eoq_incremental_quantity_discount(
        demand=demand,
        setup_cost=setup_cost,
        unit_costs=unit_costs,
        holding_rate=interest_rate,
        quantity_breaks=quantity_breaks
    )
    
    print(f"Optimal order quantity: {result['order_quantity']:.2f} units")
    print(f"Optimal price range: {result['price_range']}")
    print(f"Total annual cost: {result['total_cost']:.2f} euro")
    print(f"Ordering cost: {result['ordering_cost']:.2f} euro")
    print(f"Holding cost: {result['holding_cost']:.2f} euro")
    print(f"Purchase cost: {result['purchase_cost']:.2f} euro")

    # Create visualization to compare all-unit and incremental quantity discount models
    visualize_quantity_discounts()


def visualize_quantity_discounts():
    """Create visualization comparing all-unit and incremental quantity discount models"""
    # Given data
    demand = 40 * 52  # units/year
    setup_cost = 25  # euro
    interest_rate = 0.26  # holding cost as fraction of unit cost
    
    # Define quantity range
    q_values = np.linspace(100, 700, 1000)
    
    # All-unit quantity discount
    q_breaks = [0, 300, 500]
    unit_costs = [10, 9.7, 9.5]
    
    # Calculate total cost for all-unit quantity discount
    all_unit_costs = []
    for q in q_values:
        # Determine the unit cost based on order quantity
        if q < 300:
            c = unit_costs[0]
        elif q < 500:
            c = unit_costs[1]
        else:
            c = unit_costs[2]
        
        holding_cost = interest_rate * c
        ordering_cost = setup_cost * demand / q
        inventory_cost = holding_cost * q / 2
        purchase_cost = c * demand
        
        total_cost = ordering_cost + inventory_cost + purchase_cost
        all_unit_costs.append(total_cost)
    
    # Incremental quantity discount
    q_breaks_inc = [0, 300]
    unit_costs_inc = [10, 9.7]
    
    # Calculate total cost for incremental quantity discount
    incremental_costs = []
    for q in q_values:
        # Calculate purchase cost with incremental pricing
        if q <= 300:
            c = unit_costs_inc[0]
            holding_cost = interest_rate * c
            purchase_cost = c * demand
        else:
            # First 300 units at higher price, remaining at lower price
            purchase_cost = (unit_costs_inc[0] * 300 + unit_costs_inc[1] * (q - 300)) * demand / q
            # Calculate average cost for holding cost
            avg_cost = purchase_cost / demand
            holding_cost = interest_rate * avg_cost
        
        ordering_cost = setup_cost * demand / q
        inventory_cost = holding_cost * q / 2
        
        total_cost = ordering_cost + inventory_cost + purchase_cost
        incremental_costs.append(total_cost)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(q_values, all_unit_costs, label='All-unit discount')
    plt.plot(q_values, incremental_costs, label='Incremental discount')
    
    # Mark price breaks
    for q_break in q_breaks:
        if q_break > 0:  # Skip the 0 break point
            plt.axvline(x=q_break, color='r', linestyle='--', alpha=0.5)
            plt.text(q_break, max(max(all_unit_costs), max(incremental_costs)) * 0.9, 
                     f'Q = {q_break}', rotation=90)
    
    # Add labels and legend
    plt.xlabel('Order Quantity')
    plt.ylabel('Total Annual Cost')
    plt.title('Comparison of Quantity Discount Models')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig('assignment2_quantity_discounts.png')
    plt.close()


if __name__ == "__main__":
    main()
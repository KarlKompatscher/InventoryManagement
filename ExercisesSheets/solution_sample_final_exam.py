#!/usr/bin/env python3
"""
Solution for Sample Final Exam
Using the imf_* formula library
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import norm, poisson

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imf_main import (
    # EOQ functions
    eoq, eoq_all_unit_quantity_discount,
    
    # Newsvendor functions
    newsvendor_critical_ratio, newsvendor_normal, newsvendor_poisson,
    
    # Statistical functions
    NormalDistribution, safety_factor_from_service_level,
    
    # Inventory policy functions
    safety_stock, reorder_point, order_up_to_level, service_level_safety_stock
)


def main():
    print("Sample Final Exam Solutions")
    print("==========================")
    
    # Problem 1: EOQ with All-Unit Quantity Discount
    solve_problem1()
    
    # Problem 2: Newsvendor with Normal Demand
    solve_problem2()
    
    # Problem 3: Safety Stock and Service Level
    solve_problem3()
    
    # Problem 4: Inventory Policy with Probabilistic Demand
    solve_problem4()
    
    # Problem 5: Multi-Item Replenishment
    solve_problem5()


def solve_problem1():
    """Problem 1: EOQ with All-Unit Quantity Discount"""
    print("\nProblem 1: EOQ with All-Unit Quantity Discount")
    print("----------------------------------------------")
    
    # Given data
    demand = 1500  # units/year
    setup_cost = 50  # euro/order
    holding_rate = 0.25  # 25% of unit cost
    
    # Price break points
    quantity_breaks = [0, 100, 300]
    unit_costs = [40, 38, 36]  # euro/unit
    
    # Calculate optimal order quantity and cost
    result = eoq_all_unit_quantity_discount(
        demand=demand,
        setup_cost=setup_cost,
        unit_costs=unit_costs,
        holding_rate=holding_rate,
        quantity_breaks=quantity_breaks
    )
    
    print(f"Optimal order quantity: {result['order_quantity']:.2f} units")
    print(f"Optimal unit cost: €{result['unit_cost']:.2f}")
    print(f"Total annual cost: €{result['total_cost']:.2f}")
    print(f"Ordering cost: €{result['ordering_cost']:.2f}")
    print(f"Holding cost: €{result['holding_cost']:.2f}")
    print(f"Purchase cost: €{result['purchase_cost']:.2f}")
    
    # Visualize the cost functions
    visualize_quantity_discount_costs(demand, setup_cost, holding_rate, quantity_breaks, unit_costs, result)


def solve_problem2():
    """Problem 2: Newsvendor with Normal Demand"""
    print("\nProblem 2: Newsvendor with Normal Demand")
    print("---------------------------------------")
    
    # Given data
    mean_demand = 800  # units
    std_demand = 200  # units
    price = 25  # euro/unit
    cost = 15  # euro/unit
    salvage = 8  # euro/unit
    
    # Calculate critical ratio
    critical_ratio = newsvendor_critical_ratio(price, cost, salvage)
    print(f"Critical ratio: {critical_ratio:.4f}")
    
    # Calculate optimal order quantity
    optimal_q = newsvendor_normal(mean_demand, std_demand, critical_ratio)
    print(f"Optimal order quantity: {optimal_q:.2f} units")
    
    # Calculate key performance indicators
    z = (optimal_q - mean_demand) / std_demand
    print(f"Safety factor z: {z:.4f}")
    
    expected_shortage = NormalDistribution.expected_shortage(std_demand, z, label="", suffix="")
    print(f"Expected lost sales: {expected_shortage:.2f} units")
    
    expected_sales = mean_demand - expected_shortage
    print(f"Expected sales: {expected_sales:.2f} units")
    
    expected_leftover = optimal_q - expected_sales
    print(f"Expected leftover: {expected_leftover:.2f} units")
    
    expected_profit = price * expected_sales + salvage * expected_leftover - cost * optimal_q
    print(f"Expected profit: €{expected_profit:.2f}")
    
    # Calculate service levels
    service_level_availability = NormalDistribution.cdf(z, label="", suffix="")
    print(f"Service level (Availability): {service_level_availability:.4f}")
    
    fill_rate = expected_sales / mean_demand
    print(f"Service level (Fill-rate): {fill_rate:.4f}")
    
    # Visualize the solution
    visualize_newsvendor_solution(mean_demand, std_demand, critical_ratio, optimal_q, price, cost, salvage)


def solve_problem3():
    """Problem 3: Safety Stock and Service Level"""
    print("\nProblem 3: Safety Stock and Service Level")
    print("----------------------------------------")
    
    # Given data
    mean_demand = 100  # units/week
    std_demand = 20  # units/week
    lead_time = 2  # weeks
    target_service_level = 0.98  # 98%
    
    # Calculate safety factor
    safety_factor = safety_factor_from_service_level(target_service_level)
    print(f"Safety factor z: {safety_factor:.4f}")
    
    # Calculate safety stock
    ss = safety_stock(std_demand, safety_factor, lead_time=lead_time)
    print(f"Safety stock: {ss:.2f} units")
    
    # Calculate reorder point
    rop = reorder_point(mean_demand, lead_time, safety_stock=ss)
    print(f"Reorder point: {rop:.2f} units")
    
    # Alternative method using service_level_safety_stock
    result = service_level_safety_stock(
        mu=mean_demand,
        sigma=std_demand,
        lead_time=lead_time,
        service_level=target_service_level
    )
    
    print(f"Safety stock (alternative method): {result['safety_stock']:.2f} units")
    print(f"Reorder point (alternative method): {result['reorder_point']:.2f} units")
    
    # Visualize the safety stock and service level
    visualize_safety_stock_service_level(mean_demand, std_demand, lead_time, safety_factor, ss, rop)


def solve_problem4():
    """Problem 4: Inventory Policy with Probabilistic Demand"""
    print("\nProblem 4: Inventory Policy with Probabilistic Demand")
    print("-------------------------------------------------")
    
    # Given data
    mean_demand = 50  # units/day
    std_demand = 15  # units/day
    lead_time = 3  # days
    review_period = 2  # days
    target_service_level = 0.95  # 95%
    
    # Calculate parameters for continuous review policy (s,Q)
    annual_demand = mean_demand * 365  # Convert to annual demand
    setup_cost = 100  # euro/order
    holding_cost = 0.2  # euro/unit/year, assumed for this example
    
    # Calculate EOQ
    q = eoq(annual_demand, setup_cost, holding_cost)
    print(f"Economic Order Quantity (Q): {q:.2f} units")
    
    # Calculate safety stock for continuous review
    safety_factor = safety_factor_from_service_level(target_service_level)
    ss_continuous = safety_stock(std_demand, safety_factor, lead_time=lead_time)
    print(f"Safety stock (continuous review): {ss_continuous:.2f} units")
    
    # Calculate reorder point for continuous review
    s = reorder_point(mean_demand, lead_time, safety_stock=ss_continuous)
    print(f"Reorder point (s): {s:.2f} units")
    
    print("\nContinuous review policy (s,Q):")
    print(f"Reorder when inventory position reaches: {s:.2f} units")
    print(f"Order quantity: {q:.2f} units")
    
    # Calculate parameters for periodic review policy (R,S)
    safety_factor_periodic = safety_factor_from_service_level(target_service_level)
    ss_periodic = safety_stock(std_demand, safety_factor_periodic, lead_time=lead_time, review_period=review_period)
    print(f"\nSafety stock (periodic review): {ss_periodic:.2f} units")
    
    # Calculate order-up-to level
    S = order_up_to_level(mean_demand, lead_time, safety_stock=ss_periodic, review_period=review_period)
    print(f"Order-up-to level (S): {S:.2f} units")
    
    print("\nPeriodic review policy (R,S):")
    print(f"Review period (R): {review_period} days")
    print(f"Order-up-to level (S): {S:.2f} units")
    
    # Visualize the inventory policies
    visualize_inventory_policies(mean_demand, std_demand, lead_time, review_period, s, q, S)


def solve_problem5():
    """Problem 5: Multi-Item Replenishment"""
    print("\nProblem 5: Multi-Item Replenishment")
    print("---------------------------------")
    
    # Given data for multiple items
    items = [
        {"id": "A", "demand": 1000, "setup_cost": 100, "holding_cost": 5},
        {"id": "B", "demand": 500, "setup_cost": 120, "holding_cost": 8},
        {"id": "C", "demand": 200, "setup_cost": 150, "holding_cost": 10},
    ]
    
    # Calculate individual EOQs
    for item in items:
        item["eoq"] = eoq(item["demand"], item["setup_cost"], item["holding_cost"])
        item["cycle_time"] = item["eoq"] / item["demand"]
        item["ordering_cost"] = item["setup_cost"] * item["demand"] / item["eoq"]
        item["inventory_cost"] = item["holding_cost"] * item["eoq"] / 2
        item["total_cost"] = item["ordering_cost"] + item["inventory_cost"]
    
    print("Individual EOQ solutions:")
    for item in items:
        print(f"Item {item['id']}: EOQ = {item['eoq']:.2f}, Cycle time = {item['cycle_time']:.4f}, Total cost = {item['total_cost']:.2f}")
    
    # Calculate joint replenishment parameters
    # For simplicity, let's implement a basic approach with a common cycle time
    total_setup_cost = sum(item["setup_cost"] for item in items)
    total_holding_demand = sum(item["holding_cost"] * item["demand"] for item in items)
    
    # Approximate optimal cycle time for joint replenishment
    joint_cycle_time = math.sqrt(2 * total_setup_cost / total_holding_demand)
    print(f"\nJoint replenishment cycle time: {joint_cycle_time:.4f}")
    
    # Calculate order quantities and costs for joint replenishment
    for item in items:
        item["joint_quantity"] = joint_cycle_time * item["demand"]
        item["joint_ordering_cost"] = item["setup_cost"] / joint_cycle_time
        item["joint_inventory_cost"] = item["holding_cost"] * item["joint_quantity"] / 2
        item["joint_total_cost"] = item["joint_ordering_cost"] + item["joint_inventory_cost"]
    
    print("\nJoint replenishment solutions:")
    for item in items:
        print(f"Item {item['id']}: Quantity = {item['joint_quantity']:.2f}, Total cost = {item['joint_total_cost']:.2f}")
    
    # Calculate total costs for both approaches
    individual_total_cost = sum(item["total_cost"] for item in items)
    joint_total_cost = sum(item["joint_total_cost"] for item in items)
    
    print(f"\nTotal cost with individual EOQs: {individual_total_cost:.2f}")
    print(f"Total cost with joint replenishment: {joint_total_cost:.2f}")
    print(f"Savings: {individual_total_cost - joint_total_cost:.2f} ({(individual_total_cost - joint_total_cost) / individual_total_cost * 100:.2f}%)")
    
    # Visualize the comparison
    visualize_joint_replenishment(items)


def visualize_quantity_discount_costs(demand, setup_cost, holding_rate, quantity_breaks, unit_costs, result):
    """Visualize the cost functions for the quantity discount problem"""
    q_values = np.linspace(10, 500, 1000)
    total_costs = []
    
    for q in q_values:
        # Determine unit cost based on order quantity
        unit_cost = unit_costs[0]  # Default to the highest price
        for i in range(len(quantity_breaks) - 1, 0, -1):
            if q >= quantity_breaks[i]:
                unit_cost = unit_costs[i]
                break
        
        holding_cost = holding_rate * unit_cost
        ordering_cost = setup_cost * demand / q
        inventory_cost = holding_cost * q / 2
        purchase_cost = unit_cost * demand
        
        total_cost = ordering_cost + inventory_cost + purchase_cost
        total_costs.append(total_cost)
    
    plt.figure(figsize=(10, 6))
    plt.plot(q_values, total_costs)
    
    # Mark the optimal solution
    plt.plot(result["order_quantity"], result["total_cost"], 'ro', markersize=10)
    plt.annotate(f'Optimal Q = {result["order_quantity"]:.1f}',
                xy=(result["order_quantity"], result["total_cost"]),
                xytext=(result["order_quantity"] + 20, result["total_cost"] + 200),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Mark the price break points
    for qty in quantity_breaks[1:]:
        plt.axvline(x=qty, color='r', linestyle='--', alpha=0.7)
        plt.text(qty, min(total_costs), f'Q = {qty}', rotation=90, verticalalignment='bottom')
    
    plt.xlabel('Order Quantity (Q)')
    plt.ylabel('Total Annual Cost')
    plt.title('Total Cost Function with Quantity Discounts')
    plt.grid(True, alpha=0.3)
    plt.savefig('final_exam_problem1.png')
    plt.close()


def visualize_newsvendor_solution(mean_demand, std_demand, critical_ratio, optimal_q, price, cost, salvage):
    """Visualize the newsvendor solution"""
    # Create range for demand
    demand_values = np.linspace(mean_demand - 3 * std_demand, mean_demand + 3 * std_demand, 1000)
    pdf_values = norm.pdf(demand_values, mean_demand, std_demand)
    
    # Plot demand distribution and solution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(demand_values, pdf_values)
    plt.fill_between(demand_values, pdf_values, where=(demand_values <= optimal_q), alpha=0.3, color='green', label='No stockout')
    plt.fill_between(demand_values, pdf_values, where=(demand_values > optimal_q), alpha=0.3, color='red', label='Stockout')
    
    plt.axvline(x=mean_demand, color='blue', linestyle='--', label='Mean demand')
    plt.axvline(x=optimal_q, color='green', linestyle='-', label=f'Optimal Q = {optimal_q:.1f}')
    
    plt.xlabel('Demand')
    plt.ylabel('Probability Density')
    plt.title(f'Newsvendor Solution (CR = {critical_ratio:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create profit function plot
    plt.subplot(1, 2, 2)
    q_values = np.linspace(mean_demand - 2 * std_demand, mean_demand + 2 * std_demand, 100)
    profits = []
    
    for q in q_values:
        z = (q - mean_demand) / std_demand
        expected_shortage = NormalDistribution.expected_shortage(std_demand, z, label="", suffix="")
        expected_sales = mean_demand - expected_shortage
        expected_leftover = q - expected_sales
        profit = price * expected_sales + salvage * expected_leftover - cost * q
        profits.append(profit)
    
    plt.plot(q_values, profits)
    plt.axvline(x=optimal_q, color='green', linestyle='-', label=f'Optimal Q = {optimal_q:.1f}')
    plt.axhline(y=max(profits), color='red', linestyle='--', label=f'Max profit = {max(profits):.1f}')
    
    plt.xlabel('Order Quantity')
    plt.ylabel('Expected Profit')
    plt.title('Expected Profit vs Order Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_exam_problem2.png')
    plt.close()


def visualize_safety_stock_service_level(mean_demand, std_demand, lead_time, safety_factor, ss, rop):
    """Visualize safety stock and service level concepts"""
    # Calculate lead time demand parameters
    lt_mean = mean_demand * lead_time
    lt_std = std_demand * math.sqrt(lead_time)
    
    # Create range for lead time demand
    demand_values = np.linspace(lt_mean - 4 * lt_std, lt_mean + 4 * lt_std, 1000)
    pdf_values = norm.pdf(demand_values, lt_mean, lt_std)
    
    plt.figure(figsize=(12, 6))
    
    # Plot lead time demand distribution
    plt.subplot(1, 2, 1)
    plt.plot(demand_values, pdf_values)
    plt.fill_between(demand_values, pdf_values, where=(demand_values <= rop), alpha=0.3, color='green', label='No stockout')
    plt.fill_between(demand_values, pdf_values, where=(demand_values > rop), alpha=0.3, color='red', label='Stockout')
    
    plt.axvline(x=lt_mean, color='blue', linestyle='--', label='Lead time demand')
    plt.axvline(x=rop, color='green', linestyle='-', label=f'Reorder point = {rop:.1f}')
    
    # Add safety stock visualization
    plt.annotate('', 
                xy=(rop, 0), 
                xytext=(lt_mean, 0),
                arrowprops=dict(facecolor='black', shrink=0.0, width=1.5, headwidth=8))
    plt.annotate(f'Safety Stock = {ss:.1f}', 
                xy=((lt_mean + rop)/2, 0.0005),
                ha='center')
    
    plt.xlabel('Lead Time Demand')
    plt.ylabel('Probability Density')
    plt.title(f'Lead Time Demand and Safety Stock (z = {safety_factor:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot service level vs safety factor
    plt.subplot(1, 2, 2)
    z_values = np.linspace(-3, 3, 100)
    service_levels = [NormalDistribution.cdf(z, label="", suffix="") for z in z_values]
    
    plt.plot(z_values, service_levels)
    plt.axhline(y=NormalDistribution.cdf(safety_factor, label="", suffix=""), color='red', linestyle='--', 
               label=f'Service level = {NormalDistribution.cdf(safety_factor, label="", suffix=""):.4f}')
    plt.axvline(x=safety_factor, color='green', linestyle='-', label=f'Safety factor = {safety_factor:.2f}')
    
    plt.xlabel('Safety Factor (z)')
    plt.ylabel('Service Level')
    plt.title('Service Level vs Safety Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_exam_problem3.png')
    plt.close()


def visualize_inventory_policies(mean_demand, std_demand, lead_time, review_period, s, q, S):
    """Visualize continuous and periodic review policies"""
    # Create a timeline for visualization
    days = 30
    time = np.arange(days)
    
    # Generate random demand
    np.random.seed(42)  # For reproducibility
    demand = np.random.normal(mean_demand, std_demand, days)
    demand = np.maximum(demand, 0)  # Ensure non-negative demand
    
    # Simulate continuous review (s,Q) policy
    inventory_level_cont = np.zeros(days)
    inventory_position_cont = np.zeros(days)
    orders_cont = np.zeros(days)
    
    # Initial inventory and position
    inventory_level_cont[0] = s + q
    inventory_position_cont[0] = inventory_level_cont[0]
    
    for t in range(1, days):
        # Update inventory level based on demand
        inventory_level_cont[t] = max(0, inventory_level_cont[t-1] - demand[t-1])
        
        # Check for open orders from lead time periods ago
        if t >= lead_time and orders_cont[t-lead_time] > 0:
            inventory_level_cont[t] += orders_cont[t-lead_time]
        
        # Update inventory position
        open_orders = sum(orders_cont[max(0, t-lead_time):t])
        inventory_position_cont[t] = inventory_level_cont[t] + open_orders
        
        # Check if we need to place an order
        if inventory_position_cont[t] <= s:
            orders_cont[t] = q
            inventory_position_cont[t] += q
    
    # Simulate periodic review (R,S) policy
    inventory_level_per = np.zeros(days)
    inventory_position_per = np.zeros(days)
    orders_per = np.zeros(days)
    
    # Initial inventory and position
    inventory_level_per[0] = S
    inventory_position_per[0] = inventory_level_per[0]
    
    for t in range(1, days):
        # Update inventory level based on demand
        inventory_level_per[t] = max(0, inventory_level_per[t-1] - demand[t-1])
        
        # Check for open orders from lead time periods ago
        if t >= lead_time and orders_per[t-lead_time] > 0:
            inventory_level_per[t] += orders_per[t-lead_time]
        
        # Update inventory position
        open_orders = sum(orders_per[max(0, t-lead_time):t])
        inventory_position_per[t] = inventory_level_per[t] + open_orders
        
        # Check if this is a review period and place order to reach S
        if t % review_period == 0:
            order_amount = max(0, S - inventory_position_per[t])
            orders_per[t] = order_amount
            inventory_position_per[t] += order_amount
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # Continuous review policy
    plt.subplot(2, 1, 1)
    plt.plot(time, inventory_level_cont, 'b-', label='Inventory Level')
    plt.plot(time, inventory_position_cont, 'g--', label='Inventory Position')
    plt.step(time, orders_cont, 'r-', where='post', label='Orders')
    
    plt.axhline(y=s, color='k', linestyle=':', label=f'Reorder Point (s = {s:.1f})')
    plt.axhline(y=s + q, color='k', linestyle='--', label=f's + Q = {s + q:.1f}')
    
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.title('Continuous Review (s,Q) Policy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Periodic review policy
    plt.subplot(2, 1, 2)
    plt.plot(time, inventory_level_per, 'b-', label='Inventory Level')
    plt.plot(time, inventory_position_per, 'g--', label='Inventory Position')
    plt.step(time, orders_per, 'r-', where='post', label='Orders')
    
    plt.axhline(y=S, color='k', linestyle=':', label=f'Order-Up-To Level (S = {S:.1f})')
    
    # Mark review periods
    for t in range(0, days, review_period):
        plt.axvline(x=t, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.title('Periodic Review (R,S) Policy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_exam_problem4.png')
    plt.close()


def visualize_joint_replenishment(items):
    """Visualize comparison of individual EOQ vs. joint replenishment"""
    # Create bar chart comparing costs
    plt.figure(figsize=(10, 6))
    
    item_ids = [item["id"] for item in items]
    individual_costs = [item["total_cost"] for item in items]
    joint_costs = [item["joint_total_cost"] for item in items]
    
    x = np.arange(len(item_ids))
    width = 0.35
    
    plt.bar(x - width/2, individual_costs, width, label='Individual EOQ')
    plt.bar(x + width/2, joint_costs, width, label='Joint Replenishment')
    
    plt.xlabel('Item')
    plt.ylabel('Total Cost')
    plt.title('Individual EOQ vs. Joint Replenishment')
    plt.xticks(x, item_ids)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display total costs
    total_individual = sum(individual_costs)
    total_joint = sum(joint_costs)
    savings = total_individual - total_joint
    savings_pct = savings / total_individual * 100
    
    plt.annotate(f'Total individual: {total_individual:.2f}\nTotal joint: {total_joint:.2f}\nSavings: {savings:.2f} ({savings_pct:.1f}%)',
                xy=(0.7, 0.85), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('final_exam_problem5.png')
    plt.close()


if __name__ == "__main__":
    main()
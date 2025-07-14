#!/usr/bin/env python3
"""
Solution for Assignment 5: Multi-period Lot-sizing Problem
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import gurobipy as gp
    from gurobipy import GRB, quicksum
    GUROBI_AVAILABLE = True
except ImportError:
    print("Gurobi not available. Some functionality will be limited.")
    GUROBI_AVAILABLE = False


def main():
    print("Assignment 5: Multi-period Lot-sizing Problem")
    print("--------------------------------------------")
    
    # Problem data
    demands = np.array([550, 200, 400, 110, 430, 980, 400, 300, 200, 650])
    setup_cost = 10000
    holding_cost = 0.2 * 120  # euro/unit/period
    
    num_periods = len(demands)
    
    # Exercise 5(a): Least Unit Cost Heuristic
    print("\nExercise 5(a): Least Unit Cost Heuristic")
    
    flag_setup, lot_size = least_unit_cost_heuristic(demands, setup_cost, holding_cost)
    total_cost = calculate_total_cost(flag_setup, lot_size, demands, setup_cost, holding_cost)
    
    print(f"Setup decision: {flag_setup}")
    print(f"Lot-sizing decision: {lot_size}")
    print(f"Total costs: {total_cost}")
    
    # Exercise 5(b): Silver-Meal Heuristic
    print("\nExercise 5(b): Silver-Meal Heuristic")
    
    flag_setup, lot_size = silver_meal_heuristic(demands, setup_cost, holding_cost)
    total_cost = calculate_total_cost(flag_setup, lot_size, demands, setup_cost, holding_cost)
    
    print(f"Setup decision: {flag_setup}")
    print(f"Lot-sizing decision: {lot_size}")
    print(f"Total costs: {total_cost}")
    
    # Exercise 5(c): Wagner-Whitin Algorithm
    print("\nExercise 5(c): Wagner-Whitin Algorithm")
    
    # Use a smaller subset of periods for Wagner-Whitin demonstration
    ww_demands = demands[:6]
    ww_num_periods = len(ww_demands)
    
    ww_setup_decision, ww_lot_size, ww_total_cost = wagner_whitin_algorithm(
        ww_demands, setup_cost, holding_cost)
    
    print(f"Setup decision: {ww_setup_decision}")
    print(f"Lot-sizing decision: {ww_lot_size}")
    print(f"Total costs: {ww_total_cost}")
    
    # Exercise 5(c) MILP: Mixed-Integer Linear Programming
    if GUROBI_AVAILABLE:
        print("\nExercise 5(c) MILP: Mixed-Integer Linear Programming")
        
        milp_setup_decision, milp_lot_size, milp_total_cost = milp_lot_sizing(
            ww_demands, setup_cost, holding_cost)
        
        print(f"Setup decision: {milp_setup_decision}")
        print(f"Lot-sizing decision: {milp_lot_size}")
        print(f"Total costs: {milp_total_cost}")
    
    # Visualize and compare the different methods
    visualize_lot_sizing_solutions(demands, setup_cost, holding_cost)


def least_unit_cost_heuristic(demands, setup_cost, holding_cost):
    """
    Implement the Least Unit Cost heuristic for lot-sizing
    
    Parameters:
        demands: Array of demands for each period
        setup_cost: Setup cost per production run
        holding_cost: Holding cost per unit per period
    
    Returns:
        flag_setup: Boolean array indicating setup decisions
        lot_size: Array of lot sizes for each period
    """
    num_periods = len(demands)
    flag_setup = np.full(num_periods, False, dtype=bool)
    lot_size = np.zeros(num_periods)
    
    t = 0
    while t < num_periods:
        z = t
        c_opt = calculate_luc_criterion(t, z, demands, setup_cost, holding_cost)
        
        while z < num_periods - 1 and c_opt > calculate_luc_criterion(t, z + 1, demands, setup_cost, holding_cost):
            z += 1
            c_opt = calculate_luc_criterion(t, z, demands, setup_cost, holding_cost)
        
        flag_setup[t] = True
        lot_size[t] = np.sum(demands[t:z+1])
        t = z + 1
    
    return flag_setup, lot_size


def calculate_luc_criterion(t, z, demands, setup_cost, holding_cost):
    """Calculate the Least Unit Cost criterion for periods t through z"""
    holding_periods = [i for i in range(z - t + 1)]
    total_demand = np.sum(demands[t:z+1])
    
    if total_demand == 0:
        return float('inf')  # Avoid division by zero
    
    unit_cost = (setup_cost + holding_cost * np.sum(demands[t:z+1] * holding_periods)) / total_demand
    return unit_cost


def silver_meal_heuristic(demands, setup_cost, holding_cost):
    """
    Implement the Silver-Meal heuristic for lot-sizing
    
    Parameters:
        demands: Array of demands for each period
        setup_cost: Setup cost per production run
        holding_cost: Holding cost per unit per period
    
    Returns:
        flag_setup: Boolean array indicating setup decisions
        lot_size: Array of lot sizes for each period
    """
    num_periods = len(demands)
    flag_setup = np.full(num_periods, False, dtype=bool)
    lot_size = np.zeros(num_periods)
    
    t = 0
    while t < num_periods:
        z = t
        c_opt = calculate_sm_criterion(t, z, demands, setup_cost, holding_cost)
        
        while z < num_periods - 1 and c_opt > calculate_sm_criterion(t, z + 1, demands, setup_cost, holding_cost):
            z += 1
            c_opt = calculate_sm_criterion(t, z, demands, setup_cost, holding_cost)
        
        flag_setup[t] = True
        lot_size[t] = np.sum(demands[t:z+1])
        t = z + 1
    
    return flag_setup, lot_size


def calculate_sm_criterion(t, z, demands, setup_cost, holding_cost):
    """Calculate the Silver-Meal criterion for periods t through z"""
    holding_periods = [i for i in range(z - t + 1)]
    
    period_cost = (setup_cost + holding_cost * np.sum(demands[t:z+1] * holding_periods)) / (z - t + 1)
    return period_cost


def wagner_whitin_algorithm(demands, setup_cost, holding_cost):
    """
    Implement the Wagner-Whitin dynamic programming algorithm
    
    Parameters:
        demands: Array of demands for each period
        setup_cost: Setup cost per production run
        holding_cost: Holding cost per unit per period
    
    Returns:
        setup_decision: Boolean array indicating setup decisions
        lot_size: Array of lot sizes for each period
        total_cost: Total cost of the solution
    """
    num_periods = len(demands)
    
    # 2-d array consists of the total costs for all possible lot-sizing decisions
    costs = np.full((num_periods, num_periods), np.inf)
    
    # Base case: Order in the first period
    costs[0, 0] = setup_cost
    for t in range(1, num_periods):
        costs[0, t] = costs[0, t - 1] + t * holding_cost * demands[t]
    
    # Fill in the rest of the dynamic programming table
    for i in range(1, num_periods):
        # Cost of setting up production in period i
        costs[i, i] = np.min(costs[:, i-1]) + setup_cost
        
        # Cost of producing demand for future periods
        for t in range(i+1, num_periods):
            costs[i, t] = costs[i, t-1] + (t - i) * holding_cost * demands[t]
    
    # Extract the optimal setup decisions
    index_opt = np.argmin(costs, axis=0)
    setup_decision = np.full(num_periods, False, dtype=bool)
    setup_decision[0] = True
    
    for t in range(1, num_periods):
        if index_opt[t] == index_opt[t-1]:
            setup_decision[t] = False
        else:
            setup_decision[t] = True
    
    # Calculate lot sizes based on setup decisions
    lot_size = calculate_lot_sizes(setup_decision, demands)
    
    # Calculate total cost
    total_cost = np.min(costs[:, -1])
    
    return setup_decision, lot_size, total_cost


def milp_lot_sizing(demands, setup_cost, holding_cost):
    """
    Implement the lot-sizing problem using Mixed-Integer Linear Programming
    
    Parameters:
        demands: Array of demands for each period
        setup_cost: Setup cost per production run
        holding_cost: Holding cost per unit per period
    
    Returns:
        setup_decision: Boolean array indicating setup decisions
        lot_size: Array of lot sizes for each period
        total_cost: Total cost of the solution
    """
    if not GUROBI_AVAILABLE:
        print("Gurobi not available. Cannot solve using MILP.")
        return None, None, None
    
    num_periods = len(demands)
    big_M = np.sum(demands)
    
    # Create model
    model = gp.Model("Wagner-Whitin")
    
    # Create variables
    lotsize = model.addVars(num_periods, vtype=GRB.INTEGER, name="lotsize")
    setup = model.addVars(num_periods, vtype=GRB.BINARY, name="setup indicator")
    inventories = model.addVars(num_periods, name="inventories")
    
    # Set objective
    model.setObjective(
        quicksum(
            setup[period] * setup_cost + holding_cost * inventories[period]
            for period in range(num_periods)
        ),
        GRB.MINIMIZE
    )
    
    # Inventory balance constraints
    model.addConstr(inventories[0] == lotsize[0] - demands[0])
    model.addConstrs(
        inventories[period] == lotsize[period] - demands[period] + inventories[period - 1]
        for period in range(1, num_periods)
    )
    
    # Logic constraints
    model.addConstrs(lotsize[period] <= big_M * setup[period] for period in range(num_periods))
    
    # Solve model
    model.optimize()
    
    # Extract results
    setup_decision = [bool(setup[i].X) for i in range(num_periods)]
    lot_size = [lotsize[i].X for i in range(num_periods)]
    total_cost = model.objVal
    
    return setup_decision, lot_size, total_cost


def calculate_lot_sizes(setup_decision, demands):
    """Calculate lot sizes given setup decisions and demands"""
    num_periods = len(demands)
    lot_size = np.zeros(num_periods)
    
    last_setup = 0
    for t in range(1, num_periods + 1):
        if t == num_periods or setup_decision[t]:
            lot_size[last_setup] = np.sum(demands[last_setup:t])
            last_setup = t
    
    return lot_size


def calculate_total_cost(setup_decision, lot_size, demands, setup_cost, holding_cost):
    """Calculate the total cost of a given lot-sizing solution"""
    num_periods = len(demands)
    total_setup_cost = setup_cost * np.sum(setup_decision)
    
    # Calculate inventory levels
    inventory = np.zeros(num_periods)
    inventory[0] = lot_size[0] - demands[0]
    
    for t in range(1, num_periods):
        inventory[t] = inventory[t-1] + lot_size[t] - demands[t]
    
    total_holding_cost = holding_cost * np.sum(inventory)
    total_cost = total_setup_cost + total_holding_cost
    
    return total_cost


def visualize_lot_sizing_solutions(demands, setup_cost, holding_cost):
    """Create visualization comparing different lot-sizing methods"""
    # Calculate solutions for all methods
    luc_setup, luc_lot_size = least_unit_cost_heuristic(demands, setup_cost, holding_cost)
    luc_cost = calculate_total_cost(luc_setup, luc_lot_size, demands, setup_cost, holding_cost)
    
    sm_setup, sm_lot_size = silver_meal_heuristic(demands, setup_cost, holding_cost)
    sm_cost = calculate_total_cost(sm_setup, sm_lot_size, demands, setup_cost, holding_cost)
    
    # For Wagner-Whitin and MILP, use a smaller subset due to computational complexity
    subset_demands = demands[:6]
    ww_setup, ww_lot_size, ww_cost = wagner_whitin_algorithm(subset_demands, setup_cost, holding_cost)
    
    # Create full-size arrays for WW (first 6 periods only, rest zeros)
    ww_lot_size_full = np.zeros(len(demands))
    ww_lot_size_full[:len(subset_demands)] = ww_lot_size
    
    ww_setup_full = np.full(len(demands), False, dtype=bool)
    ww_setup_full[:len(subset_demands)] = ww_setup
    
    # Create plot for lot sizes
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.bar(range(len(demands)), demands, alpha=0.3, color='gray', label='Demands')
    plt.bar(range(len(demands)), luc_lot_size, alpha=0.6, color='blue', label='LUC Lot Size')
    plt.bar(range(len(demands)), sm_lot_size, alpha=0.6, color='green', label='SM Lot Size')
    plt.bar(range(len(demands)), ww_lot_size_full, alpha=0.6, color='red', label='WW Lot Size (first 6 periods)')
    
    plt.xlabel('Period')
    plt.ylabel('Units')
    plt.title('Comparison of Lot-Sizing Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create plot for inventory levels
    plt.subplot(2, 1, 2)
    
    # Calculate inventory levels for each method
    luc_inventory = calculate_inventory_levels(luc_lot_size, demands)
    sm_inventory = calculate_inventory_levels(sm_lot_size, demands)
    ww_inventory = calculate_inventory_levels(ww_lot_size_full, demands)
    
    plt.plot(range(len(demands)), luc_inventory, 'bo-', label=f'LUC (Cost: {luc_cost})')
    plt.plot(range(len(demands)), sm_inventory, 'go-', label=f'SM (Cost: {sm_cost})')
    plt.plot(range(len(demands)), ww_inventory, 'ro-', label=f'WW (First 6 periods)')
    
    # Mark setup periods
    for t in range(len(demands)):
        if luc_setup[t]:
            plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5)
        if sm_setup[t]:
            plt.axvline(x=t, color='green', linestyle=':', alpha=0.5)
        if t < len(ww_setup) and ww_setup_full[t]:
            plt.axvline(x=t, color='red', linestyle='-.', alpha=0.5)
    
    plt.xlabel('Period')
    plt.ylabel('Inventory Level')
    plt.title('Inventory Levels by Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('assignment5_lot_sizing.png')
    plt.close()


def calculate_inventory_levels(lot_size, demands):
    """Calculate inventory levels for a given lot-sizing solution"""
    num_periods = len(demands)
    inventory = np.zeros(num_periods)
    
    inventory[0] = lot_size[0] - demands[0]
    for t in range(1, num_periods):
        inventory[t] = inventory[t-1] + lot_size[t] - demands[t]
    
    return inventory


if __name__ == "__main__":
    main()
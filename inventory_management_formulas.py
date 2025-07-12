"""
Comprehensive Inventory Management Formulas

This script contains a collection of functions for inventory management calculations,
consolidated from various exercises and examples. Use this as a reference for exam preparation.
"""

import math
import numpy as np
from scipy.stats import norm, poisson, gamma, uniform
from scipy.optimize import minimize_scalar, fsolve, root
from scipy.integrate import quad
import matplotlib.pyplot as plt
import pandas as pd


def log(label, value, unit=""):
    """
    Simple logging utility for labeled output.

    Parameters:
        label: Description of the value
        value: Value to display
        unit: Optional unit for formatting
    """
    print(f"[INFO] {label}: {round(value, 2)} {unit}")

    

#####################################################
# Basic EOQ Models
#####################################################

def eoq(D, S, h, label="EOQ"):
    """
    Economic Order Quantity (EOQ) - Basic formula

    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        h: Holding cost per unit per year
        label: Optional label for logging output (default: "EOQ")

    Returns:
        EOQ value
    """
    EOQ = math.sqrt((2 * D * S) / h)
    log(label, EOQ, unit="units")
    return EOQ


def total_relevant_cost_for_eoq(D, S, h, label="TRC"):
    """
    Total Relevant Cost (TRC) at EOQ — the minimum total cost per year.
    
    Parameters:
        D: Annual demand
        S: Fixed ordering/setup cost
        h: Holding cost per unit per year
    
    Returns:
        Total relevant cost (TRC) — ordering + holding cost at EOQ
    """
    EOQ = math.sqrt((2 * D * S) / h)
    log(label, EOQ, unit="units")
    return EOQ


def cycle_length(eoq, d):
    """
    Cycle length between orders

    Parameters:
        eoq: Economic Order Quantity
        d: Demand per period

    Returns:
        T: Cycle time (periods between orders)
    """
    return eoq / d


def lot_cost(d, A, h, q):
    """
    Total cost per period for a given order quantity

    Parameters:
        d: Demand per period
        A: Setup cost per order
        h: Holding cost per unit
        q: Order quantity

    Returns:
        Total cost per period
    """
    return d / q * A + 0.5 * h * q


def cost_penalty(q, eoq):
    """
    Percentage Cost Penalty (PCP) for ordering quantity deviating from EOQ

    Parameters:
        q: Actual order quantity
        eoq: Economic Order Quantity

    Returns:
        PCP: Percentage cost penalty
    """
    p = (q - eoq) / eoq
    return 50 * (p**2 / (1 + p))


def percentage_deviation(q, eoq):
    """
    Percentage deviation of order quantity from EOQ

    Parameters:
        q: Actual order quantity
        eoq: Economic Order Quantity

    Returns:
        Deviation in %
    """
    return 100 * (q - eoq) / eoq


def optimal_power_of_two_cycle(d, A, h):
    """
    Find optimal cycle time (as a power of two) that minimizes cost

    Parameters:
        d: Demand per period
        A: Setup cost per order
        h: Holding cost per unit

    Returns:
        t: Optimal cycle multiplier (power of 2)
        cost_error: Percentage error vs. EOQ cost
    """
    def cost(q):
        return lot_cost(d, A, h, q)
    
    t = 1
    while cost(2 * d * t) < cost(d * t):
        t *= 2
    
    TRC = total_relevant_cost_for_eoq(d, A, h)
    error = 100 * (cost(d * t) / TRC - 1)
    
    return t, round(error, 2)


def total_annual_cost(D, S, h, Q, price=0):
    """
    Total annual cost for EOQ
    
    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        h: Holding cost per unit per year
        Q: Order quantity
        price: Unit price (default 0)
    
    Returns:
        Total annual cost
    """
    purchase_cost = D * price
    ordering_cost = S * (D / Q)
    holding_cost = h * (Q / 2)
    return purchase_cost + ordering_cost + holding_cost

def eoq_price_break(D, S, i, price, q_min=0, q_max=float("inf")):
    """
    EOQ with price breaks (returns EOQ, adjusted for min/max Q)
    
    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        i: Interest rate (for holding cost)
        price: Unit price
        q_min: Minimum order quantity
        q_max: Maximum order quantity
    
    Returns:
        Economic order quantity considering price breaks
    """
    h = i * price
    Q = math.sqrt((2 * D * S) / h)
    Q = max(q_min, min(Q, q_max))
    return Q

def eoq_box_constrained(D, S, i, c, box_size):
    """
    EOQ with box size constraints (must be multiples of box_size)
    
    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        i: Interest rate (for holding cost)
        c: Unit cost
        box_size: Box size (order must be multiples of this)
    
    Returns:
        Optimal order quantity (multiple of box_size)
    """
    # Step 1: Compute EOQ without constraints
    h = i * c
    Q_eoq = math.sqrt((2 * D * S) / h)
    
    # Step 2: Find multiples of box_size around EOQ
    lower_multiple = box_size * math.floor(Q_eoq / box_size)
    upper_multiple = box_size * math.ceil(Q_eoq / box_size)
    
    # Step 3: Calculate total cost for both candidates
    def calculate_total_cost(Q):
        purchase = D * c
        ordering = S * (D / Q)
        holding = h * (Q / 2)
        return purchase + ordering + holding
    
    cost_lower = calculate_total_cost(lower_multiple)
    cost_upper = calculate_total_cost(upper_multiple)
    
    # Step 4: Choose optimal order quantity
    if cost_lower <= cost_upper:
        return lower_multiple, cost_lower
    else:
        return upper_multiple, cost_upper

def all_unit_quantity_discount(d, A, r, bp, cp):
    """
    All-unit quantity discount model
    
    Parameters:
        d: Annual demand
        A: Setup cost
        r: Interest rate
        bp: Break points of order quantity
        cp: Purchasing prices
    
    Returns:
        Optimal order quantity and cost
    """
    # Start from the cheapest price
    t = len(cp) - 1  
    holding_cost = r * cp[t]
    qt = math.sqrt(2 * A * d / holding_cost)
    
    # Check if EOQ is feasible
    if qt >= bp[t]:
        q_opt = qt
        c_opt = d / qt * A + 0.5 * holding_cost * qt + cp[t] * d
    else:
        # Calculate EOQ for less favorable price
        while t >= 1 and qt < bp[t]:
            t -= 1
            holding_cost = r * cp[t]
            qt = math.sqrt(2 * A * d / holding_cost)
            cost_break = d / bp[t + 1] * A + 0.5 * r * cp[t + 1] * bp[t + 1] + cp[t + 1] * d
            cost_eoq = d / qt * A + 0.5 * holding_cost * qt + cp[t] * d
            
            # Compare cost at break point and at EOQ
            if cost_break < cost_eoq:
                q_opt = bp[t + 1]
                c_opt = cost_break
                break
            else:
                q_opt = qt
                c_opt = cost_eoq
    
    return q_opt, c_opt

def incremental_quantity_discount(d, setup_cost, r, bp, cp):
    """
    Incremental quantity discount model
    
    Parameters:
        d: Annual demand
        setup_cost: Setup cost
        r: Interest rate
        bp: Break points of order quantity
        cp: Purchasing prices
    
    Returns:
        Optimal order quantity and cost
    """
    # Compute the sum of terms independent of Q in purchasing cost
    R = np.zeros(len(bp))
    for t in range(1, len(bp)):
        R[t] = cp[t - 1] * (bp[t] - bp[t - 1]) + R[t - 1]
    
    # Compute EOQ for all segments & check feasibility
    qt = np.zeros(len(bp))
    flag_feasible = np.full(len(bp), False, dtype=bool)
    
    for t in range(len(bp)):
        qt[t] = math.sqrt(2 * (R[t] - cp[t] * bp[t] + setup_cost) * d / (r * cp[t]))
    
    for t in range(len(bp) - 1):
        flag_feasible[t] = True if qt[t] >= bp[t] and qt[t] < bp[t + 1] else False
    
    flag_feasible[-1] = True if qt[-1] >= bp[-1] else False
    
    # Compute total cost for feasible EOQs
    qt_f = []
    cost_qt_f = []
    
    for t in range(len(bp)):
        if flag_feasible[t]:
            q = qt[t]
            c = (R[t] + cp[t] * (q - bp[t])) / q
            holding_cost = c * r
            cost_q = lot_cost(d, setup_cost, holding_cost, q)
            qt_f.append(q)
            cost_qt_f.append(cost_q)
    
    c_opt = min(cost_qt_f)
    q_opt = qt_f[np.argmin(cost_qt_f)]
    
    return q_opt, c_opt

def eoq_sensitivity_analysis(q_actual, q_optimal):
    """
    EOQ sensitivity analysis
    
    Parameters:
        q_actual: Actual order quantity
        q_optimal: Optimal order quantity
    
    Returns:
        Percentage deviation and percentage cost penalty
    """
    p = (q_actual - q_optimal) / q_optimal  # Percentage deviation
    PCP = 50 * (p**2 / (1 + p))  # Percentage cost penalty
    
    return p * 100, PCP

def power_of_two(d, A, holding_cost, q_optimal):
    """
    Find optimal integer cycle time using power of two policy
    
    Parameters:
        d: Demand
        A: Setup cost
        holding_cost: Holding cost
        q_optimal: Optimal order quantity
    
    Returns:
        Optimal integer cycle time
    """
    t = 1
    while lot_cost(d, A, holding_cost, 2 * d * t) < lot_cost(d, A, holding_cost, d * t):
        t = t * 2
    
    optimal_cost = lot_cost(d, A, holding_cost, d * t)
    optimal_eoq_cost = lot_cost(d, A, holding_cost, q_optimal)
    p_error = 100 * (optimal_cost / optimal_eoq_cost - 1)
    
    return t, p_error


#####################################################
# Newsvendor Models
#####################################################

def newsvendor_critical_ratio(p, c, g=0):
    """
    Newsvendor critical ratio
    
    Parameters:
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
    
    Returns:
        Critical ratio
    """
    return (p - c) / (p - g)

def newsvendor_normal(mu, sigma, CR):
    """
    Newsvendor order quantity for normally distributed demand
    
    Parameters:
        mu: Mean demand
        sigma: Standard deviation of demand
        CR: Critical ratio
    
    Returns:
        Optimal order quantity
    """
    z = norm.ppf(CR)
    Q = mu + z * sigma
    return Q

def newsvendor_uniform(loc, scale, beta):
    """
    Newsvendor order quantity for uniformly distributed demand
    
    Parameters:
        loc: Lower bound
        scale: Upper bound - lower bound
        beta: Critical ratio
    
    Returns:
        Optimal order quantity
    """
    return uniform.ppf(beta, loc=loc, scale=scale)

def newsvendor_poisson(lambda_val, beta):
    """
    Newsvendor order quantity for Poisson distributed demand
    
    Parameters:
        lambda_val: Lambda parameter (mean)
        beta: Critical ratio
    
    Returns:
        Optimal order quantity
    """
    return poisson.ppf(beta, lambda_val)

def newsvendor_gamma(mean, std, beta):
    """
    Newsvendor order quantity for Gamma distributed demand
    
    Parameters:
        mean: Mean demand
        std: Standard deviation of demand
        beta: Critical ratio
    
    Returns:
        Optimal order quantity
    """
    var = std**2
    alpha = mean**2 / var
    theta = var / mean
    
    return gamma.ppf(beta, alpha, scale=theta)

def newsvendor_kpi(q, mu, sigma, p, c, g=0):
    """
    Calculate key performance indicators for newsvendor model
    
    Parameters:
        q: Order quantity
        mu: Mean demand
        sigma: Standard deviation of demand
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
    
    Returns:
        Dictionary with KPIs
    """
    z = (q - mu) / sigma  # Standard normal distribution
    ELS = sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))  # Expected lost sales
    ES = mu - ELS  # Expected sales
    ELO = q - ES  # Expected leftover
    EP = -c * q + p * ES + g * ELO  # Expected profit
    alpha = norm.cdf(z)  # Service level (availability)
    fill_rate = ES / mu  # Service level (fill rate)
    
    return {
        "z": z,
        "Expected Lost Sales": ELS,
        "Expected Sales": ES,
        "Expected Leftover": ELO,
        "Expected Profit": EP,
        "Service Level (Availability)": alpha,
        "Service Level (Fill Rate)": fill_rate
    }

def newsvendor_general(label, distr, params, p, c, g=0):
    """
    Generalized newsvendor solution for different demand distributions
    
    Parameters:
        label: Name/identifier
        distr: Distribution type ('normal', 'poisson', or 'gamma')
        params: Distribution parameters dictionary
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
    
    Returns:
        Dictionary with newsvendor metrics
    """
    CR = (p - c) / (p - g)
    
    if distr == "normal":
        mu = params["mu"]
        sigma = params["sigma"]
        z = norm.ppf(CR)
        Q = mu + z * sigma
        
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        G_z = phi_z - z * (1 - Phi_z)
        expected_lost_sales = sigma * G_z
        expected_sales = mu - expected_lost_sales
        expected_leftover_inventory = Q - expected_sales
        expected_profit = -c * Q + p * expected_sales + g * expected_leftover_inventory
        non_stockout_prob = Phi_z
        fill_rate = expected_sales / mu
        
        return {
            "Label": label,
            "Distribution": "Normal",
            "Critical ratio (CR)": CR,
            "z": z,
            "phi(z)": phi_z,
            "Phi(z)": Phi_z,
            "G(z)": G_z,
            "Order Quantity": Q,
            "Expected Lost Sales": expected_lost_sales,
            "Expected Sales": expected_sales,
            "Expected Leftover Inventory": expected_leftover_inventory,
            "Expected Profit": expected_profit,
            "Non-stockout Probability (alpha)": non_stockout_prob,
            "Fill Rate (beta)": fill_rate
        }
    
    elif distr == "poisson":
        lam = params["lambda"]
        Q = 0
        while poisson.cdf(Q, lam) < CR:
            Q += 1
        
        # Expected lost sales
        max_k = Q + 1000  # Approximate upper bound
        expected_lost_sales = sum((k - Q) * poisson.pmf(k, lam) for k in range(Q + 1, max_k + 1))
        expected_sales = lam - expected_lost_sales
        expected_leftover_inventory = Q - expected_sales
        expected_profit = -c * Q + p * expected_sales + g * expected_leftover_inventory
        non_stockout_prob = poisson.cdf(Q, lam)
        fill_rate = expected_sales / lam
        
        return {
            "Label": label,
            "Distribution": "Poisson",
            "Critical ratio (CR)": CR,
            "Order Quantity": Q,
            "Expected Lost Sales": expected_lost_sales,
            "Expected Sales": expected_sales,
            "Expected Leftover Inventory": expected_leftover_inventory,
            "Expected Profit": expected_profit,
            "Non-stockout Probability (alpha)": non_stockout_prob,
            "Fill Rate (beta)": fill_rate
        }
    
    elif distr == "gamma":
        mu = params["mu"]
        sigma = params["sigma"]
        
        alpha = (mu**2) / (sigma**2)
        theta = (sigma**2) / mu
        
        Q = gamma.ppf(CR, a=alpha, scale=theta)
        
        def integrand(x):
            return x * gamma.pdf(x, a=alpha, scale=theta)
        
        integral_val, _ = quad(integrand, 0, Q)
        expected_sales = integral_val + Q * (1 - gamma.cdf(Q, a=alpha, scale=theta))
        expected_leftover_inventory = Q - expected_sales
        expected_lost_sales = mu - integral_val
        expected_profit = -c * Q + p * expected_sales + g * expected_leftover_inventory
        non_stockout_prob = gamma.cdf(Q, a=alpha, scale=theta)
        fill_rate = expected_sales / mu
        
        return {
            "Label": label,
            "Distribution": "Gamma",
            "Critical ratio (CR)": CR,
            "Shape (alpha)": alpha,
            "Scale (theta)": theta,
            "Order Quantity": Q,
            "Expected Lost Sales": expected_lost_sales,
            "Expected Sales": expected_sales,
            "Expected Leftover Inventory": expected_leftover_inventory,
            "Expected Profit": expected_profit,
            "Non-stockout Probability (alpha)": non_stockout_prob,
            "Fill Rate (beta)": fill_rate
        }
    
    else:
        raise ValueError("Unsupported distribution")

def newsvendor_find_z_for_fillrate(beta, mu, sigma):
    """
    Find z-value to achieve desired fill rate (beta)
    
    Parameters:
        beta: Desired fill rate
        mu: Mean demand
        sigma: Standard deviation
    
    Returns:
        z-value for standard normal distribution
    """
    def f(x):
        return abs((norm.pdf(x) - x * (1 - norm.cdf(x))) - (1 - beta) * mu / sigma)
    
    z = minimize_scalar(f, method="golden").x
    return z


#####################################################
# Safety Stock and Reorder Points
#####################################################

def reorder_point(D, lead_time_months):
    """
    Calculate reorder point (ROP) for constant demand
    
    Parameters:
        D: Annual demand
        lead_time_months: Lead time in months
    
    Returns:
        Reorder point
    """
    monthly_demand = D / 12
    return monthly_demand * lead_time_months

def safety_stock(z, std_dev, lead_time, review_period=0):
    """
    Safety stock calculation
    
    Parameters:
        z: Safety factor
        std_dev: Standard deviation of demand
        lead_time: Lead time
        review_period: Review period (default 0)
    
    Returns:
        Safety stock
    """
    return z * std_dev * math.sqrt(lead_time + review_period)

def service_level_safety_stock(mean_demand, std_demand, lead_time, service_level):
    """
    Calculate safety stock for a given service level
    
    Parameters:
        mean_demand: Mean demand
        std_demand: Standard deviation of demand
        lead_time: Lead time
        service_level: Desired service level (probability)
    
    Returns:
        Reorder point
    """
    z = norm.ppf(service_level)
    ss = z * std_demand * math.sqrt(lead_time)
    reorder_pt = mean_demand * lead_time + ss
    
    return ss, reorder_pt

def order_up_to_level(forecast, safety_stock):
    """
    Order-up-to level (S)
    
    Parameters:
        forecast: Demand forecast
        safety_stock: Safety stock
    
    Returns:
        Order-up-to level
    """
    return np.array(forecast) + safety_stock

def variable_lead_time_safety_stock(z, mean_demand, std_demand, mean_lead_time, std_lead_time, review_period=0):
    """
    Safety stock calculation for variable lead time
    
    Parameters:
        z: Safety factor
        mean_demand: Mean demand
        std_demand: Standard deviation of demand
        mean_lead_time: Mean lead time
        std_lead_time: Standard deviation of lead time
        review_period: Review period (default 0)
    
    Returns:
        Safety stock
    """
    return z * math.sqrt((mean_lead_time + review_period) * std_demand**2 + mean_demand**2 * std_lead_time**2)


#####################################################
# Forecasting Methods
#####################################################

def moving_average(demand, window):
    """
    Moving average forecast
    
    Parameters:
        demand: Array of historical demand
        window: Window size
    
    Returns:
        Array of moving average forecasts
    """
    return pd.Series(demand).rolling(window=window).mean().shift(1).to_numpy()

def exp_smoothing(alpha, demand, initial_forecast):
    """
    Exponential smoothing forecast
    
    Parameters:
        alpha: Smoothing parameter
        demand: Array of historical demand
        initial_forecast: Initial forecast value
    
    Returns:
        Array of forecasts
    """
    forecasts = [initial_forecast]
    for t in range(len(demand)):
        new_forecast = alpha * demand[t] + (1 - alpha) * forecasts[-1]
        forecasts.append(new_forecast)
    return forecasts[1:]  # Remove initial forecast

def exponential_smoothing_error(alpha, demand, initial_level=0):
    """
    Compute exponential smoothing error
    
    Parameters:
        alpha: Smoothing parameter
        demand: Array of historical demand
        initial_level: Initial level (default 0)
    
    Returns:
        Array of errors
    """
    exp_smoothed = [initial_level]
    for i in range(len(demand)):
        exp_smoothed.append(alpha * demand[i] + (1 - alpha) * exp_smoothed[-1])
    return np.array(exp_smoothed[1:])

def croston_method(indices, values, alpha=0.2):
    """
    Croston's method for intermittent demand
    
    Parameters:
        indices: Array of demand occurrence indices
        values: Array of demand values
        alpha: Smoothing parameter (default 0.2)
    
    Returns:
        DataFrame with Croston's method results
    """
    results = []
    
    for i in range(len(values)):
        if i == 0:
            x = np.array([indices[i]])  # Interval
            a = np.array([values[i]])   # Demand size
            forecast_day = np.array([math.floor(indices[i] + x[-1])])
            forecast_quantity = np.array([math.ceil(a[-1])])
        else:
            # Update interval estimate
            x = np.append(x, (1 - alpha) * x[-1] + alpha * (indices[i] - indices[i - 1]))
            # Update demand size estimate
            a = np.append(a, (1 - alpha) * a[-1] + alpha * values[i])
            # Forecast next occurrence
            forecast_day = np.append(forecast_day, math.floor(indices[i] + x[-1]))
            # Forecast next quantity
            forecast_quantity = np.append(forecast_quantity, math.ceil(a[-1]))
    
    # Organize results into DataFrame
    df = pd.DataFrame({
        "x": x,                               # Interval estimate
        "a": a,                               # Demand size estimate
        "forecast_day": forecast_day,         # Next forecast day
        "forecast_quantity": forecast_quantity # Next forecast quantity
    })
    
    return df


#####################################################
# Multi-Period Inventory Models
#####################################################

def calculate_luc_criterion(t, z, setup_cost, holding_cost, demands):
    """
    Calculate least unit cost criterion
    
    Parameters:
        t: Starting period
        z: Ending period
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
    
    Returns:
        Least unit cost
    """
    holding_periods = [i for i in range(z - t + 1)]
    unit_cost = (
        setup_cost + holding_cost * np.sum(demands[t:z+1] * holding_periods)
    ) / np.sum(demands[t:z+1])
    return unit_cost

def calculate_sm_criterion(t, z, setup_cost, holding_cost, demands):
    """
    Calculate Silver-Meal criterion
    
    Parameters:
        t: Starting period
        z: Ending period
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
    
    Returns:
        Silver-Meal criterion value
    """
    holding_periods = [i for i in range(z - t + 1)]
    period_cost = (
        setup_cost + holding_cost * np.sum(demands[t:z+1] * holding_periods)
    ) / (z - t + 1)
    return period_cost

def make_lot_sizing_decision(func_cost, num_periods, setup_cost, holding_cost, demands):
    """
    Make lot-sizing decision using a cost criterion function
    
    Parameters:
        func_cost: Cost criterion function (LUC or SM)
        num_periods: Number of periods
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
    
    Returns:
        Tuple of setup decision and lot size arrays
    """
    flag_setup = np.full(num_periods, False, dtype=bool)  # Setup indicator
    lot_size = np.zeros(num_periods)  # Lot size for each period
    
    t = 0
    while t < num_periods:
        z = t
        c_opt = func_cost(t, z, setup_cost, holding_cost, demands)
        
        while c_opt > func_cost(t, z + 1, setup_cost, holding_cost, demands):
            z += 1
            c_opt = func_cost(t, z, setup_cost, holding_cost, demands)
            if z == num_periods - 1:
                break
        
        flag_setup[t] = True
        lot_size[t] = np.sum(demands[t:z+1])
        t = z + 1
    
    return flag_setup, lot_size

def calculate_total_cost(flag_setup, lot_size, demands, setup_cost, holding_cost):
    """
    Calculate total cost for a lot-sizing decision
    
    Parameters:
        flag_setup: Array of setup indicators
        lot_size: Array of lot sizes
        demands: Array of demands
        setup_cost: Setup cost
        holding_cost: Holding cost
    
    Returns:
        Total cost
    """
    # Setup cost
    total_setup_cost = setup_cost * np.sum(flag_setup)
    
    # Inventory holding cost
    num_periods = len(demands)
    inventory = np.zeros(num_periods)
    inventory[0] = lot_size[0] - demands[0]
    
    for t in range(1, num_periods):
        inventory[t] = inventory[t-1] + lot_size[t] - demands[t]
    
    total_holding_cost = holding_cost * np.sum(inventory)
    total_cost = total_setup_cost + total_holding_cost
    
    return total_cost

def wagner_whitin(num_periods, setup_cost, holding_cost, demands):
    """
    Wagner-Whitin algorithm for lot-sizing
    
    Parameters:
        num_periods: Number of periods
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
    
    Returns:
        Tuple of setup decision array and total cost
    """
    # 2D array for total costs
    costs = np.full((num_periods, num_periods), np.inf)
    
    # Option 1: Order in the first period
    costs[0, 0] = setup_cost
    for t in range(1, num_periods):
        costs[0, t] = costs[0, t-1] + t * holding_cost * demands[t]
    
    # Options 2...n: Order in period j
    for j in range(1, num_periods):
        costs[j, j] = np.min(costs[:, j-1]) + setup_cost
        for t in range(j+1, num_periods):
            costs[j, t] = costs[j, t-1] + (t - j) * holding_cost * demands[t]
    
    # Get setup decision
    index_opt = np.argmin(costs, axis=0)
    setup_decision = np.full(num_periods, False, dtype=bool)
    setup_decision[0] = True
    
    for t in range(1, num_periods):
        if index_opt[t] == index_opt[t-1]:
            setup_decision[t] = False
        else:
            setup_decision[t] = True
    
    return setup_decision, np.min(costs[:, -1])


#####################################################
# Multi-Echelon Inventory Control
#####################################################

def common_cycle(A_list, h_list, d_list):
    """
    Common replenishment cycle (T*)
    
    Parameters:
        A_list: List of ordering costs
        h_list: List of holding costs
        d_list: List of demand rates
    
    Returns:
        Optimal common cycle time
    """
    total_ordering = 2 * sum(A_list)
    total_holding = sum([h * d for h, d in zip(h_list, d_list)])
    return math.sqrt(total_ordering / total_holding)

def material_requirement_planning(gross_requirements, arrivals, starting_inventory, safety_stock=0):
    """
    Calculate net requirements using Material Requirements Planning (MRP)
    
    Parameters:
        gross_requirements: Array of gross requirements
        arrivals: Array of scheduled arrivals
        starting_inventory: Initial inventory
        safety_stock: Safety stock (default 0)
    
    Returns:
        Array of net requirements
    """
    net_requirements = []
    
    # Calculate net requirement for each period
    for i in range(len(gross_requirements)):
        if i == 0:
            starting_inventory = starting_inventory + arrivals[i]
            net_requirement = max(0, safety_stock + gross_requirements[i] - starting_inventory)
        else:
            starting_inventory = starting_inventory - gross_requirements[i-1] + arrivals[i] + net_requirements[-1]
            net_requirement = max(0, gross_requirements[i] + safety_stock - starting_inventory)
        
        net_requirements.append(net_requirement)
    
    return net_requirements

def serial_system_echelon_stock(inventory_positions, net_requirements):
    """
    Calculate echelon inventory positions for a serial system
    
    Parameters:
        inventory_positions: List of local inventory positions
        net_requirements: List of net requirements
    
    Returns:
        List of echelon inventory positions
    """
    n = len(inventory_positions)
    echelon_positions = []
    
    for i in range(n):
        if i == 0:
            echelon_positions.append(inventory_positions[i])
        else:
            echelon_positions.append(inventory_positions[i] + sum(net_requirements[0:i]))
    
    return echelon_positions

def guaranteed_service_model(lead_times, holding_costs, mean, std, service_level, candidates=None):
    """
    Guaranteed Service Model for multi-echelon inventory optimization
    
    Parameters:
        lead_times: List of lead times
        holding_costs: List of holding costs
        mean: Mean demand
        std: Standard deviation of demand
        service_level: Service level
        candidates: Array of allocation candidates (optional)
    
    Returns:
        Dictionary with results
    """
    n_stage = len(holding_costs)
    safety_factor = norm.ppf(service_level)
    
    # Create allocation candidates if not provided
    if candidates is None:
        import itertools
        candidates = np.array(list(itertools.product([0, 1], repeat=n_stage-1) + [(1,)]))
    
    # Calculate coverage time for each candidate
    def func_coverage_time(decision):
        cover_time = np.zeros(n_stage)
        cumulative_time = 0
        
        for i in range(n_stage-1):
            if decision[i] == 1:
                cover_time[i] = lead_times[i] + cumulative_time
                cumulative_time = 0
            else:
                cover_time[i] = 0
                cumulative_time += lead_times[i]
        
        # Last stage always covers its lead time
        cover_time[n_stage-1] = lead_times[-1] + cumulative_time
        return cover_time
    
    coverage_times = []
    for candidate in candidates:
        coverage_times.append(func_coverage_time(candidate))
    
    coverage_times = np.array(coverage_times)
    
    # Calculate safety stock for each candidate
    safety_stocks = []
    for coverage_time in coverage_times:
        ss = []
        for t in coverage_time:
            ss.append(safety_factor * std * math.sqrt(t) if t > 0 else 0)
        safety_stocks.append(ss)
    
    safety_stocks = np.array(safety_stocks)
    
    # Calculate total cost for each candidate
    total_costs = np.sum(holding_costs * safety_stocks, axis=1)
    
    # Find optimal candidate
    opt_idx = np.argmin(total_costs)
    
    return {
        "candidates": candidates,
        "coverage_times": coverage_times,
        "safety_stocks": safety_stocks,
        "total_costs": total_costs,
        "optimal_candidate": candidates[opt_idx],
        "optimal_coverage_time": coverage_times[opt_idx],
        "optimal_safety_stock": safety_stocks[opt_idx],
        "optimal_cost": total_costs[opt_idx]
    }

def clark_scarf_model(lead_times, mu, sigma, holding_costs, penalty_cost):
    """
    Clark-Scarf model for two-stage serial system
    
    Parameters:
        lead_times: List of lead times [upstream, downstream]
        mu: Mean demand
        sigma: Standard deviation of demand
        holding_costs: List of holding costs [upstream, downstream]
        penalty_cost: Penalty cost for shortages
    
    Returns:
        Dictionary with optimal order-up-to levels
    """
    # Optimal Solutions Stage 2 (downstream)
    critical_ratio_s2 = (holding_costs[0] + penalty_cost) / (holding_costs[1] + penalty_cost)
    opt_s2 = norm.ppf(critical_ratio_s2, mu * (lead_times[1]+1), sigma * math.sqrt(lead_times[1]+1))
    
    # Optimal Solutions Stage 1 (upstream)
    def func_s1(s1):
        critical_ratio_s1 = penalty_cost / (holding_costs[1] + penalty_cost)
        # Integrate func from 0 to opt_s2
        integral = integrate.quad(
            lambda d: norm.cdf(s1-d, mu*lead_times[0], sigma*math.sqrt(lead_times[0])) * 
                    norm.pdf(d, mu*(lead_times[1]+1), sigma*math.sqrt(lead_times[1]+1)), 
            0, opt_s2)[0]
        return (integral - critical_ratio_s1)
    
    opt_s1 = fsolve(func_s1, opt_s2)[0]
    
    return {
        "optimal_s2": opt_s2,
        "optimal_s1": opt_s1,
        "critical_ratio_s2": critical_ratio_s2,
        "critical_ratio_s1": penalty_cost / (holding_costs[1] + penalty_cost)
    }

def metric_model(demand_rate, penalty_cost, holding_costs, lead_times):
    """
    METRIC model for two-echelon inventory system
    
    Parameters:
        demand_rate: Mean demand rate (λ)
        penalty_cost: Penalty cost
        holding_costs: List of holding costs [warehouse, retailer]
        lead_times: List of lead times [warehouse, retailer]
    
    Returns:
        Dictionary with results
    """
    # Bounds of retailer lead time
    lt_r_min = lead_times[1]  # Minimum possible lead time = retailer lead time
    lt_r_max = sum(lead_times)  # Maximum possible lead time = retailer + warehouse lead time
    
    def func_exp_cost_r(base_stock_r, lt):
        """Compute retailer's expected inventory cost"""
        # Range of possible demand values (up to 99th percentile)
        max_demand = poisson.ppf(0.99, demand_rate * lt)
        x = np.arange(max_demand + 1)
        
        # Probability mass function for Poisson(λ*lt)
        prob = poisson.pmf(x, demand_rate * lt)
        
        # Expected leftover inventory: E[max(S_r - X, 0)]
        leftover = np.maximum(base_stock_r - x, 0)
        exp_leftover = np.sum(leftover * prob)
        
        # Expected backorders: E[max(X - S_r, 0)] = E[X - S_r] + E[max(S_r - X, 0)]
        exp_backorder = (demand_rate * lt - base_stock_r) + exp_leftover
        
        # Total expected cost
        exp_cost_r = holding_costs[1] * exp_leftover + penalty_cost * exp_backorder
        return exp_cost_r
    
    # Find optimal S_r for min/max lead times
    s_r_candidates = range(int(demand_rate * lt_r_max * 2))  # Range of possible S_r values
    
    exp_cost_lt_min = [func_exp_cost_r(s, lt_r_min) for s in s_r_candidates]
    exp_cost_lt_max = [func_exp_cost_r(s, lt_r_max) for s in s_r_candidates]
    
    sr_min = np.argmin(exp_cost_lt_min)
    sr_max = np.argmin(exp_cost_lt_max)
    
    def func_exp_inv_w(base_stock_w):
        """Compute warehouse's expected inventory outcomes"""
        # Mean demand over warehouse lead time
        mu = demand_rate * lead_times[0]
        
        # Possible demand values (up to 99th percentile)
        max_demand = poisson.ppf(0.99, mu)
        x = np.arange(max_demand + 1)
        
        # Probability of each demand level
        prob = poisson.pmf(x, mu)
        
        # Expected leftover: E[max(S_w - X, 0)]
        leftover = np.maximum(base_stock_w - x, 0)
        exp_leftover = np.sum(leftover * prob)
        
        # Expected backorder: E[max(X - S_w, 0)]
        backorder = np.maximum(x - base_stock_w, 0)
        exp_backorder = np.sum(backorder * prob)
        
        return exp_leftover, exp_backorder
    
    def func_exp_total_cost(base_stock_w, base_stock_r):
        """Compute total expected cost"""
        # Get warehouse inventory metrics
        exp_leftover_w, exp_backorder_w = func_exp_inv_w(base_stock_w)
        
        # Approximate retailer lead time with average delay from warehouse backorders
        exp_lt_r = lead_times[1] + exp_backorder_w / demand_rate
        
        # Total cost = warehouse holding + retailer (holding + penalty)
        exp_total_cost = (
            holding_costs[0] * exp_leftover_w
            + func_exp_cost_r(base_stock_r, exp_lt_r)
        )
        return exp_total_cost
    
    # Find bounds on S_w
    s_w_candidates = range(int(demand_rate * lead_times[0] * 2))
    
    exp_total_cost_sw_min = [func_exp_total_cost(s, sr_max) for s in s_w_candidates]
    exp_total_cost_sw_max = [func_exp_total_cost(s, sr_min) for s in s_w_candidates]
    
    sw_min = np.argmin(exp_total_cost_sw_min)
    sw_max = np.argmin(exp_total_cost_sw_max)
    
    # Enumerate all combinations within bounds
    total_costs = np.zeros((sw_max - sw_min + 1, sr_max - sr_min + 1))
    
    for i in range(sw_max - sw_min + 1):
        for j in range(sr_max - sr_min + 1):
            total_costs[i, j] = func_exp_total_cost(sw_min + i, sr_min + j)
    
    # Find optimal combination
    min_idx = np.unravel_index(np.argmin(total_costs), total_costs.shape)
    opt_sw = sw_min + min_idx[0]
    opt_sr = sr_min + min_idx[1]
    opt_cost = total_costs[min_idx]
    
    return {
        "optimal_sw": opt_sw,
        "optimal_sr": opt_sr,
        "optimal_cost": opt_cost,
        "sr_bounds": (sr_min, sr_max),
        "sw_bounds": (sw_min, sw_max),
        "total_costs": total_costs
    }


#####################################################
# Classification Systems
#####################################################

def xyz_classification(sales):
    """
    XYZ classification based on coefficient of variation
    
    Parameters:
        sales: Array of sales data
    
    Returns:
        Classification ('X', 'Y', or 'Z')
    """
    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    
    if mean_sales == 0:
        return "Z"  # Avoid division by zero
    
    cv = std_sales / mean_sales
    
    if cv <= 0.5:
        return "X"  # Very stable demand
    elif cv < 1.25:
        return "Y"  # Moderate variability
    else:
        return "Z"  # High variability

def extended_xyz_classification(sales):
    """
    Extended XYZ classification with detailed information
    
    Parameters:
        sales: Array of sales data
    
    Returns:
        Dictionary with classification details
    """
    total_sales = np.sum(sales)
    if total_sales == 0:
        return {"classification": "Z", "cv": float('inf')}
    
    # Calculate coefficient of variation (CV)
    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    cv = std_sales / mean_sales if mean_sales != 0 else float('inf')
    
    # Classify based on CV thresholds
    if cv <= 0.5:
        classification = "X"  # Very stable demand
    elif cv < 1.25:
        classification = "Y"  # Moderate variability
    else:
        classification = "Z"  # High variability
    
    return {
        "classification": classification,
        "cv": cv,
        "mean": mean_sales,
        "std": std_sales
    }

def sample_variance(data):
    """
    Calculate sample variance
    
    Parameters:
        data: Array of data points
    
    Returns:
        Sample variance
    """
    return np.var(data, ddof=1)


#####################################################
# Limited Capacity Models
#####################################################

def economic_lot_scheduling_problem(demand_rates, production_rates, setup_costs, holding_costs):
    """
    Economic Lot Scheduling Problem (ELSP)
    
    Parameters:
        demand_rates: List of demand rates
        production_rates: List of production rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
    
    Returns:
        Dictionary with results
    """
    n = len(demand_rates)
    
    # Calculate EOQ for each product
    Q = []
    for i in range(n):
        numerator = 2 * demand_rates[i] * setup_costs[i]
        denominator = holding_costs[i] * (1 - demand_rates[i] / production_rates[i])
        q_value = math.sqrt(numerator / denominator)
        Q.append(q_value)
    
    # Calculate cycle time for each product
    T = [q / d for q, d in zip(Q, demand_rates)]
    
    # Calculate production time for each product
    production_times = [q / p + r for q, p, r in zip(Q, production_rates, [0] * n)]
    
    # Calculate cost for each product
    costs = []
    for i in range(n):
        ordering_cost = (demand_rates[i] / Q[i]) * setup_costs[i]
        holding_cost = (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * (Q[i] / production_rates[i])
        costs.append(ordering_cost + holding_cost)
    
    return {
        "Q": Q,
        "T": T,
        "production_times": production_times,
        "costs": costs,
        "total_cost": sum(costs),
        "total_production_time": sum(production_times),
        "min_cycle_time": min(T)
    }

def common_cycle_approach(demand_rates, production_rates, setup_costs, holding_costs, setup_times=None):
    """
    Common Cycle Approach for ELSP
    
    Parameters:
        demand_rates: List of demand rates
        production_rates: List of production rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        setup_times: List of setup times (optional)
    
    Returns:
        Dictionary with results
    """
    # Calculate unconstrained common cycle time
    numerator = 2 * sum(setup_costs)
    denominator = sum([h * d * (1 - d / p) for h, d, p in zip(holding_costs, demand_rates, production_rates)])
    T_unconstrained = math.sqrt(numerator / denominator)
    
    # Calculate capacity constraint (if setup times provided)
    if setup_times:
        numerator_c = sum(setup_times)
        denominator_c = 1 - sum([d / p for d, p in zip(demand_rates, production_rates)])
        T_constrained = numerator_c / denominator_c
        T_optimal = max(T_unconstrained, T_constrained)
    else:
        T_optimal = T_unconstrained
    
    # Calculate lot sizes and costs
    Q_optimal = [d * T_optimal for d in demand_rates]
    costs = []
    
    for i in range(len(demand_rates)):
        setup_term = setup_costs[i] / T_optimal
        holding_term = (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * (demand_rates[i] / production_rates[i]) * T_optimal
        costs.append(setup_term + holding_term)
    
    return {
        "T_optimal": T_optimal,
        "Q_optimal": Q_optimal,
        "costs": costs,
        "total_cost": sum(costs)
    }

def power_of_two_policy(demand_rates, production_rates, setup_costs, holding_costs, base_period):
    """
    Power-of-Two Policy for ELSP
    
    Parameters:
        demand_rates: List of demand rates
        production_rates: List of production rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        base_period: Base period length
    
    Returns:
        Dictionary with results
    """
    n = len(demand_rates)
    n_values = [1] * n  # Initialize all products with base frequency
    
    # Iterative improvement
    while True:
        # Calculate base cycle time
        numerator = sum([2 * setup_costs[i] / n_values[i] for i in range(n)])
        denominator = sum([holding_costs[i] * (production_rates[i] - demand_rates[i]) * 
                          (demand_rates[i] / production_rates[i]) * n_values[i] for i in range(n)])
        base_cycle = math.sqrt(numerator / denominator)
        
        # Update n values
        old_n = n_values.copy()
        
        for i in range(n):
            # Try doubling the frequency
            while True:
                cost_n = (setup_costs[i] / (n_values[i] * base_cycle) + 
                        (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * 
                        (demand_rates[i] / production_rates[i]) * n_values[i] * base_cycle)
                
                cost_n_plus_1 = (setup_costs[i] / (n_values[i] * 2 * base_cycle) + 
                                (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * 
                                (demand_rates[i] / production_rates[i]) * n_values[i] * 2 * base_cycle)
                
                if cost_n <= cost_n_plus_1:
                    break
                    
                n_values[i] = n_values[i] * 2
        
        # Check convergence
        if old_n == n_values:
            break
    
    # Calculate lot sizes and costs
    Q = [d * n * base_cycle for d, n in zip(demand_rates, n_values)]
    costs = []
    
    for i in range(n):
        setup_term = setup_costs[i] / (n_values[i] * base_cycle)
        holding_term = (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * (demand_rates[i] / production_rates[i]) * n_values[i] * base_cycle
        costs.append(setup_term + holding_term)
    
    return {
        "base_cycle": base_cycle,
        "n_values": n_values,
        "Q": Q,
        "costs": costs,
        "total_cost": sum(costs)
    }

def economic_manufacturing_quantity(d, A, r, p):
    """
    Economic Manufacturing Quantity (EMQ) with finite production rate
    
    Parameters:
        d: Demand rate
        A: Setup cost
        r: Holding cost rate
        p: Production rate
    
    Returns:
        Optimal lot size
    """
    return math.sqrt((2 * d * A) / (r * (1 - d/p)))

def capacity_constrained_lotsize(d, A, h, a, W):
    """
    Calculate lot sizes with storage capacity constraint
    
    Parameters:
        d: List of demand rates
        A: List of setup costs
        h: List of holding costs
        a: List of space requirements per unit
        W: Total warehouse capacity
    
    Returns:
        Dictionary with results
    """
    # Calculate unconstrained EOQ
    Q_unconstrained = [math.sqrt((2 * d_i * A_i) / h_i) for d_i, A_i, h_i in zip(d, A, h)]
    total_space = sum([Q_i * a_i for Q_i, a_i in zip(Q_unconstrained, a)])
    
    # If unconstrained solution is feasible, return it
    if total_space <= W:
        total_cost = sum([(d_i * A_i) / Q_i + (h_i * Q_i) / 2 for d_i, A_i, Q_i, h_i in zip(d, A, Q_unconstrained, h)])
        return {
            "Q": Q_unconstrained,
            "total_space": total_space,
            "total_cost": total_cost,
            "constrained": False
        }
    
    # If unconstrained solution is not feasible, find Lagrange multiplier
    def equation(lambda_value):
        return sum([a_i * math.sqrt(2 * d_i * A_i / (h_i + 2 * lambda_value * a_i)) 
                   for d_i, A_i, h_i, a_i in zip(d, A, h, a)]) - W
    
    result = root(equation, x0=0)
    lambda_value = result.x[0]
    
    # Calculate constrained solution
    Q_constrained = [math.sqrt((2 * d_i * A_i) / (h_i + 2 * lambda_value * a_i)) 
                     for d_i, A_i, h_i, a_i in zip(d, A, h, a)]
    
    total_space = sum([Q_i * a_i for Q_i, a_i in zip(Q_constrained, a)])
    total_cost = sum([(d_i * A_i) / Q_i + (h_i * Q_i) / 2 for d_i, A_i, Q_i, h_i in zip(d, A, Q_constrained, h)])
    
    return {
        "Q": Q_constrained,
        "total_space": total_space,
        "total_cost": total_cost,
        "constrained": True,
        "lambda": lambda_value
    }

def joint_replenishment(d, A, h, A0):
    """
    Joint Replenishment Problem
    
    Parameters:
        d: List of demand rates
        A: List of individual setup costs
        h: List of holding costs
        A0: Major setup cost
    
    Returns:
        Dictionary with results
    """
    n = len(d)
    
    # Calculate individual optimal cycle times
    T_i = [math.sqrt(2 * A_i / (h_i * d_i)) for A_i, h_i, d_i in zip(A, h, d)]
    
    # Find product with minimum cycle time
    min_index = T_i.index(min(T_i))
    
    # Initialize n values
    n_i = [math.sqrt(A_i * h[min_index] * d[min_index] / (h_i * d_i * (A0 + A[min_index]))) 
           for A_i, h_i, d_i in zip(A, h, d)]
    n_i = [round(value) for value in n_i]
    
    # Iterative improvement
    n = n_i
    while True:
        # Calculate base cycle time
        sum_A_div_n = A0 + sum([A_i / n_i for A_i, n_i in zip(A, n)])
        sum_h_d_n = sum([h_i * d_i * n_i for h_i, d_i, n_i in zip(h, d, n)])
        T = math.sqrt(2 * sum_A_div_n / sum_h_d_n)
        
        # Store old n values
        old_n = n.copy()
        
        # Update n values
        for i in range(len(n)):
            while True:
                if (n[i] * (n[i] + 1) >= 2 * A[i] / (h[i] * d[i] * T**2)):
                    break
                n[i] = n[i] + 1
        
        # Recalculate base cycle time
        sum_A_div_n = A0 + sum([A_i / n_i for A_i, n_i in zip(A, n)])
        sum_h_d_n = sum([h_i * d_i * n_i for h_i, d_i, n_i in zip(h, d, n)])
        T = math.sqrt(2 * sum_A_div_n / sum_h_d_n)
        
        # Check convergence
        if old_n == n:
            break
    
    return {
        "base_cycle": T,
        "n_values": n,
        "individual_cycles": [n_i * T for n_i in n],
        "order_quantities": [d_i * n_i * T for d_i, n_i in zip(d, n)]
    }


#####################################################
# Risk Pooling and Correlation
#####################################################

def risk_pooling_benefit(demand_means, demand_stds, correlation_matrix, service_level):
    """
    Calculate risk pooling benefit
    
    Parameters:
        demand_means: List of mean demands
        demand_stds: List of standard deviations
        correlation_matrix: Correlation matrix between demands
        service_level: Service level
    
    Returns:
        Dictionary with results
    """
    n = len(demand_means)
    z = norm.ppf(service_level)
    
    # Calculate individual safety stocks
    individual_ss = [z * std for std in demand_stds]
    individual_total = sum(individual_ss)
    
    # Calculate pooled demand parameters
    pooled_mean = sum(demand_means)
    
    # Calculate pooled standard deviation with correlation
    pooled_var = 0
    for i in range(n):
        for j in range(n):
            pooled_var += demand_stds[i] * demand_stds[j] * correlation_matrix[i][j]
    
    pooled_std = math.sqrt(pooled_var)
    pooled_ss = z * pooled_std
    
    # Calculate benefit
    absolute_benefit = individual_total - pooled_ss
    relative_benefit = absolute_benefit / individual_total * 100
    
    return {
        "individual_safety_stocks": individual_ss,
        "individual_total": individual_total,
        "pooled_mean": pooled_mean,
        "pooled_std": pooled_std,
        "pooled_safety_stock": pooled_ss,
        "absolute_benefit": absolute_benefit,
        "relative_benefit": relative_benefit
    }

def risk_pooling_correlation(mu_A, sigma_A, mu_B, sigma_B, rho, p, c, g=0):
    """
    Calculate risk pooling with correlation coefficient
    
    Parameters:
        mu_A: Mean demand for product A
        sigma_A: Standard deviation for product A
        mu_B: Mean demand for product B
        sigma_B: Standard deviation for product B
        rho: Correlation coefficient between A and B
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
    
    Returns:
        Dictionary with individual and joint order quantities
    """
    # Critical ratio and z-score
    CR = (p - c) / (p - g)
    z_CR = norm.ppf(CR)
    
    # Individual order quantities
    Q_A = mu_A + z_CR * sigma_A
    Q_B = mu_B + z_CR * sigma_B
    
    # Joint demand parameters
    mu_joint = mu_A + mu_B
    sigma_joint = math.sqrt(sigma_A**2 + sigma_B**2 + 2 * rho * sigma_A * sigma_B)
    
    # Joint order quantity
    Q_joint = mu_joint + z_CR * sigma_joint
    
    # Calculate effect of correlation
    effect = (Q_A + Q_B) - Q_joint
    
    return {
        "Q_A": Q_A,
        "Q_B": Q_B,
        "sum_individual": Q_A + Q_B,
        "mu_joint": mu_joint,
        "sigma_joint": sigma_joint,
        "Q_joint": Q_joint,
        "effect": effect,
        "CV_A": sigma_A / mu_A,
        "CV_B": sigma_B / mu_B,
        "CV_joint": sigma_joint / mu_joint
    }

def coefficient_of_correlation(data1, data2):
    """
    Calculate coefficient of correlation between two datasets
    
    Parameters:
        data1: First dataset
        data2: Second dataset
    
    Returns:
        Correlation coefficient
    """
    return np.corrcoef(data1, data2)[0, 1]

def bullwhip_effect(orders_variance, demand_variance):
    """
    Calculate bullwhip effect ratio
    
    Parameters:
        orders_variance: Variance of orders
        demand_variance: Variance of demand
    
    Returns:
        Bullwhip effect ratio
    """
    return orders_variance / demand_variance
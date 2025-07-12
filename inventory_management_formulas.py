"""
Comprehensive Inventory Management Formulas

This script contains a collection of functions for inventory management calculations,
consolidated from various exercises and examples. Use this as a reference for exam preparation.
"""

import math
import numpy as np
from scipy.stats import norm, poisson, gamma, uniform
from scipy.optimize import minimize_scalar, fsolve, root
from scipy import integrate
import pandas as pd


def log(label, value, unit="", suffix=""):
    """
    Simple logging utility for labeled output.

    Parameters:
        label: Description of the value
        value: Value to display
        unit: Optional unit for formatting
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    """
    suffix_str = f"_({suffix})" if suffix else ""
    if isinstance(value, (int, float)):
        print(f"[INFO] {label}{suffix_str}: {round(value, 2)} {unit}")
    else:
        print(f"[INFO] {label}{suffix_str}: {value} {unit}")

    

#####################################################
# Basic EOQ Models
#####################################################

def eoq(D, S, h, label="EOQ", suffix=""):
    """
    Economic Order Quantity (EOQ) - Basic formula

    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        h: Holding cost per unit per year
        label: Optional label for logging output (default: "EOQ")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        EOQ value
    """
    EOQ = math.sqrt((2 * D * S) / h)
    log(label, EOQ, unit="units", suffix=suffix)
    return EOQ


def total_relevant_cost_for_eoq(D, S, h, label="TRC", suffix=""):
    """
    Total Relevant Cost (TRC) at EOQ — the minimum total cost per year.
    
    Parameters:
        D: Annual demand
        S: Fixed ordering/setup cost
        h: Holding cost per unit per year
        label: Optional label for logging output (default: "TRC")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Total relevant cost (TRC) — ordering + holding cost at EOQ
    """
    EOQ = math.sqrt((2 * D * S) / h)
    log("Annual demand", D, suffix=suffix)
    log("Setup cost", S, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("EOQ", EOQ, unit="units", suffix=suffix)
    
    ordering_cost = (D / EOQ) * S
    holding_cost = (EOQ / 2) * h
    total_cost = ordering_cost + holding_cost
    
    log(f"{label} (ordering)", ordering_cost, suffix=suffix)
    log(f"{label} (holding)", holding_cost, suffix=suffix)
    log(label, total_cost, suffix=suffix)
    
    return total_cost


def cycle_length(eoq, d, label="Cycle length", suffix=""):
    """
    Cycle length between orders

    Parameters:
        eoq: Economic Order Quantity
        d: Demand per period
        label: Optional label for logging output (default: "Cycle length")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        T: Cycle time (periods between orders)
    """
    T = eoq / d
    log(label, T, unit="periods", suffix=suffix)
    return T


def lot_cost(d, A, h, q, label="Total cost", suffix=""):
    """
    Total cost per period for a given order quantity

    Parameters:
        d: Demand per period
        A: Setup cost per order
        h: Holding cost per unit
        q: Order quantity
        label: Optional label for logging output (default: "Total cost")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        Total cost per period
    """
    ordering_cost = d / q * A
    holding_cost = 0.5 * h * q
    total = ordering_cost + holding_cost
    
    log(f"{label} (ordering)", ordering_cost, suffix=suffix)
    log(f"{label} (holding)", holding_cost, suffix=suffix)
    log(label, total, suffix=suffix)
    
    return total


def cost_penalty(q, eoq, label="Cost penalty", suffix=""):
    """
    Percentage Cost Penalty (PCP) for ordering quantity deviating from EOQ

    Parameters:
        q: Actual order quantity
        eoq: Economic Order Quantity
        label: Optional label for logging output (default: "Cost penalty")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        PCP: Percentage cost penalty
    """
    p = (q - eoq) / eoq
    penalty = 50 * (p**2 / (1 + p))
    log(label, penalty, unit="%", suffix=suffix)
    return penalty


def percentage_deviation(q, eoq, label="Percentage deviation", suffix=""):
    """
    Percentage deviation of order quantity from EOQ

    Parameters:
        q: Actual order quantity
        eoq: Economic Order Quantity
        label: Optional label for logging output (default: "Percentage deviation")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        Deviation in %
    """
    deviation = 100 * (q - eoq) / eoq
    log(label, deviation, unit="%", suffix=suffix)
    return deviation


def optimal_power_of_two_cycle(d, A, h, label="Optimal power of two cycle", suffix=""):
    """
    Find optimal cycle time (as a power of two) that minimizes cost

    Parameters:
        d: Demand per period
        A: Setup cost per order
        h: Holding cost per unit
        label: Optional label for logging output (default: "Optimal power of two cycle")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        t: Optimal cycle multiplier (power of 2)
        cost_error: Percentage error vs. EOQ cost
    """
    def cost(q):
        return lot_cost(d, A, h, q, label="Cost", suffix=suffix)
    
    t = 1
    log(f"{label} (t initial)", t, suffix=suffix)
    
    while cost(2 * d * t) < cost(d * t):
        t *= 2
        log(f"{label} (t updated)", t, suffix=suffix)
    
    TRC = total_relevant_cost_for_eoq(d, A, h, label="TRC", suffix=suffix)
    error = 100 * (cost(d * t) / TRC - 1)
    
    log(f"{label} (optimal t)", t, suffix=suffix)
    log(f"{label} (percentage error)", error, unit="%", suffix=suffix)
    
    return t, round(error, 2)


def total_annual_cost(D, S, h, Q, price=0, label="Total annual cost", suffix=""):
    """
    Total annual cost for EOQ
    
    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        h: Holding cost per unit per year
        Q: Order quantity
        price: Unit price (default 0)
        label: Optional label for logging output (default: "Total annual cost")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Total annual cost
    """
    purchase_cost = D * price
    ordering_cost = S * (D / Q)
    holding_cost = h * (Q / 2)
    total_cost = purchase_cost + ordering_cost + holding_cost
    
    log(f"{label} (purchase)", purchase_cost, suffix=suffix)
    log(f"{label} (ordering)", ordering_cost, suffix=suffix)
    log(f"{label} (holding)", holding_cost, suffix=suffix)
    log(label, total_cost, suffix=suffix)
    
    return total_cost

def eoq_price_break(D, S, i, price, q_min=0, q_max=float("inf"), label="EOQ price break", suffix=""):
    """
    EOQ with price breaks (returns EOQ, adjusted for min/max Q)
    
    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        i: Interest rate (for holding cost)
        price: Unit price
        q_min: Minimum order quantity
        q_max: Maximum order quantity
        label: Optional label for logging output (default: "EOQ price break")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Economic order quantity considering price breaks
    """
    h = i * price
    log("Holding cost rate", h, suffix=suffix)
    
    Q = math.sqrt((2 * D * S) / h)
    log(f"{label} (unconstrained)", Q, unit="units", suffix=suffix)
    
    Q = max(q_min, min(Q, q_max))
    log(label, Q, unit="units", suffix=suffix)
    
    return Q

def eoq_box_constrained(D, S, i, c, box_size, label="EOQ box constrained", suffix=""):
    """
    EOQ with box size constraints (must be multiples of box_size)
    
    Parameters:
        D: Annual demand
        S: Fixed ordering cost
        i: Interest rate (for holding cost)
        c: Unit cost
        box_size: Box size (order must be multiples of this)
        label: Optional label for logging output (default: "EOQ box constrained")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity (multiple of box_size)
    """
    # Step 1: Compute EOQ without constraints
    h = i * c
    Q_eoq = math.sqrt((2 * D * S) / h)
    log("Holding cost rate", h, suffix=suffix)
    log(f"{label} (unconstrained)", Q_eoq, unit="units", suffix=suffix)
    
    # Step 2: Find multiples of box_size around EOQ
    lower_multiple = box_size * math.floor(Q_eoq / box_size)
    upper_multiple = box_size * math.ceil(Q_eoq / box_size)
    log(f"{label} (lower multiple)", lower_multiple, unit="units", suffix=suffix)
    log(f"{label} (upper multiple)", upper_multiple, unit="units", suffix=suffix)
    
    # Step 3: Calculate total cost for both candidates
    def calculate_total_cost(Q):
        purchase = D * c
        ordering = S * (D / Q)
        holding = h * (Q / 2)
        return purchase + ordering + holding
    
    cost_lower = calculate_total_cost(lower_multiple)
    cost_upper = calculate_total_cost(upper_multiple)
    log(f"Total cost (lower)", cost_lower, suffix=suffix)
    log(f"Total cost (upper)", cost_upper, suffix=suffix)
    
    # Step 4: Choose optimal order quantity
    if cost_lower <= cost_upper:
        log(label, lower_multiple, unit="units", suffix=suffix)
        log(f"{label} cost", cost_lower, suffix=suffix)
        return lower_multiple, cost_lower
    else:
        log(label, upper_multiple, unit="units", suffix=suffix)
        log(f"{label} cost", cost_upper, suffix=suffix)
        return upper_multiple, cost_upper

def all_unit_quantity_discount(d, A, r, bp, cp, label="All-unit quantity discount", suffix=""):
    """
    All-unit quantity discount model
    
    Parameters:
        d: Annual demand
        A: Setup cost
        r: Interest rate
        bp: Break points of order quantity
        cp: Purchasing prices
        label: Optional label for logging output (default: "All-unit quantity discount")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity and cost
    """
    # Start from the cheapest price
    t = len(cp) - 1  
    holding_cost = r * cp[t]
    qt = math.sqrt(2 * A * d / holding_cost)
    
    log(f"{label} (starting price tier)", t, suffix=suffix)
    log(f"{label} (initial holding cost)", holding_cost, suffix=suffix)
    log(f"{label} (initial EOQ)", qt, unit="units", suffix=suffix)
    
    # Check if EOQ is feasible
    if qt >= bp[t]:
        q_opt = qt
        c_opt = d / qt * A + 0.5 * holding_cost * qt + cp[t] * d
        log(f"{label} (feasible at lowest price)", "Yes", suffix=suffix)
    else:
        log(f"{label} (feasible at lowest price)", "No", suffix=suffix)
        # Calculate EOQ for less favorable price
        while t >= 1 and qt < bp[t]:
            t -= 1
            holding_cost = r * cp[t]
            qt = math.sqrt(2 * A * d / holding_cost)
            cost_break = d / bp[t + 1] * A + 0.5 * r * cp[t + 1] * bp[t + 1] + cp[t + 1] * d
            cost_eoq = d / qt * A + 0.5 * holding_cost * qt + cp[t] * d
            
            log(f"{label} (checking price tier)", t, suffix=suffix)
            log(f"{label} (holding cost at tier)", holding_cost, suffix=suffix)
            log(f"{label} (EOQ at tier)", qt, unit="units", suffix=suffix)
            log(f"{label} (cost at break point)", cost_break, suffix=suffix)
            log(f"{label} (cost at EOQ)", cost_eoq, suffix=suffix)
            
            # Compare cost at break point and at EOQ
            if cost_break < cost_eoq:
                q_opt = bp[t + 1]
                c_opt = cost_break
                log(f"{label} (break point wins)", "Yes", suffix=suffix)
                break
            else:
                q_opt = qt
                c_opt = cost_eoq
                log(f"{label} (EOQ wins)", "Yes", suffix=suffix)
    
    log(f"{label} (optimal quantity)", q_opt, unit="units", suffix=suffix)
    log(f"{label} (optimal cost)", c_opt, suffix=suffix)
    
    return q_opt, c_opt

def incremental_quantity_discount(d, setup_cost, r, bp, cp, label="Incremental quantity discount", suffix=""):
    """
    Incremental quantity discount model
    
    Parameters:
        d: Annual demand
        setup_cost: Setup cost
        r: Interest rate
        bp: Break points of order quantity
        cp: Purchasing prices
        label: Optional label for logging output (default: "Incremental quantity discount")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity and cost
    """
    # Compute the sum of terms independent of Q in purchasing cost
    R = np.zeros(len(bp))
    for t in range(1, len(bp)):
        R[t] = cp[t - 1] * (bp[t] - bp[t - 1]) + R[t - 1]
    
    log(f"{label} (R values)", R, suffix=suffix)
    
    # Compute EOQ for all segments & check feasibility
    qt = np.zeros(len(bp))
    flag_feasible = np.full(len(bp), False, dtype=bool)
    
    for t in range(len(bp)):
        qt[t] = math.sqrt(2 * (R[t] - cp[t] * bp[t] + setup_cost) * d / (r * cp[t]))
        log(f"{label} (EOQ at tier {t})", qt[t], unit="units", suffix=suffix)
    
    for t in range(len(bp) - 1):
        flag_feasible[t] = True if qt[t] >= bp[t] and qt[t] < bp[t + 1] else False
        log(f"{label} (tier {t} feasible)", flag_feasible[t], suffix=suffix)
    
    flag_feasible[-1] = True if qt[-1] >= bp[-1] else False
    log(f"{label} (tier {len(bp)-1} feasible)", flag_feasible[-1], suffix=suffix)
    
    # Compute total cost for feasible EOQs
    qt_f = []
    cost_qt_f = []
    
    for t in range(len(bp)):
        if flag_feasible[t]:
            q = qt[t]
            c = (R[t] + cp[t] * (q - bp[t])) / q
            holding_cost = c * r
            log(f"{label} (average cost at tier {t})", c, suffix=suffix)
            log(f"{label} (holding cost at tier {t})", holding_cost, suffix=suffix)
            
            cost_q = lot_cost(d, setup_cost, holding_cost, q, label=f"{label} (cost at tier {t})", suffix=suffix)
            qt_f.append(q)
            cost_qt_f.append(cost_q)
    
    if not cost_qt_f:  # If no feasible EOQ found
        log(f"{label} (no feasible EOQ found)", "True", suffix=suffix)
        return None, None
    
    c_opt = min(cost_qt_f)
    q_opt = qt_f[np.argmin(cost_qt_f)]
    
    log(f"{label} (optimal quantity)", q_opt, unit="units", suffix=suffix)
    log(f"{label} (optimal cost)", c_opt, suffix=suffix)
    
    return q_opt, c_opt

def eoq_sensitivity_analysis(q_actual, q_optimal, label="EOQ sensitivity analysis", suffix=""):
    """
    EOQ sensitivity analysis
    
    Parameters:
        q_actual: Actual order quantity
        q_optimal: Optimal order quantity
        label: Optional label for logging output (default: "EOQ sensitivity analysis")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Percentage deviation and percentage cost penalty
    """
    p = (q_actual - q_optimal) / q_optimal  # Percentage deviation
    PCP = 50 * (p**2 / (1 + p))  # Percentage cost penalty
    
    log(f"{label} (percentage deviation)", p * 100, unit="%", suffix=suffix)
    log(f"{label} (percentage cost penalty)", PCP, unit="%", suffix=suffix)
    
    return p * 100, PCP

def power_of_two(d, A, holding_cost, q_optimal, label="Power of two", suffix=""):
    """
    Find optimal integer cycle time using power of two policy
    
    Parameters:
        d: Demand
        A: Setup cost
        holding_cost: Holding cost
        q_optimal: Optimal order quantity
        label: Optional label for logging output (default: "Power of two")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal integer cycle time
    """
    t = 1
    log(f"{label} (t initial)", t, suffix=suffix)
    
    while lot_cost(d, A, holding_cost, 2 * d * t, label=f"{label} (cost 2t)", suffix=suffix) < \
          lot_cost(d, A, holding_cost, d * t, label=f"{label} (cost t)", suffix=suffix):
        t = t * 2
        log(f"{label} (t updated)", t, suffix=suffix)
    
    optimal_cost = lot_cost(d, A, holding_cost, d * t, label=f"{label} (optimal cost)", suffix=suffix)
    optimal_eoq_cost = lot_cost(d, A, holding_cost, q_optimal, label=f"{label} (EOQ cost)", suffix=suffix)
    p_error = 100 * (optimal_cost / optimal_eoq_cost - 1)
    
    log(f"{label} (optimal t)", t, suffix=suffix)
    log(f"{label} (percentage error)", p_error, unit="%", suffix=suffix)
    
    return t, p_error


#####################################################
# Newsvendor Models
#####################################################

def newsvendor_critical_ratio(p, c, g=0, label="Critical ratio", suffix=""):
    """
    Newsvendor critical ratio
    
    Parameters:
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
        label: Optional label for logging output (default: "Critical ratio")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Critical ratio
    """
    CR = (p - c) / (p - g)
    log(label, CR, suffix=suffix)
    return CR

def newsvendor_normal(mu, sigma, CR, label="Order quantity", suffix=""):
    """
    Newsvendor order quantity for normally distributed demand
    
    Parameters:
        mu: Mean demand
        sigma: Standard deviation of demand
        CR: Critical ratio
        label: Optional label for logging output (default: "Order quantity")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity
    """
    z = norm.ppf(CR)
    log("z-value", z, suffix=suffix)
    
    Q = mu + z * sigma
    log(label, Q, unit="units", suffix=suffix)
    return Q

def newsvendor_uniform(loc, scale, beta, label="Newsvendor uniform", suffix=""):
    """
    Newsvendor order quantity for uniformly distributed demand
    
    Parameters:
        loc: Lower bound
        scale: Upper bound - lower bound
        beta: Critical ratio
        label: Optional label for logging output (default: "Newsvendor uniform")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity
    """
    Q = uniform.ppf(beta, loc=loc, scale=scale)
    
    log("Lower bound", loc, suffix=suffix)
    log("Upper bound", loc + scale, suffix=suffix)
    log("Critical ratio", beta, suffix=suffix)
    log(label, Q, unit="units", suffix=suffix)
    
    return Q

def newsvendor_poisson(lambda_val, beta, label="Newsvendor Poisson", suffix=""):
    """
    Newsvendor order quantity for Poisson distributed demand
    
    Parameters:
        lambda_val: Lambda parameter (mean)
        beta: Critical ratio
        label: Optional label for logging output (default: "Newsvendor Poisson")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity
    """
    Q = poisson.ppf(beta, lambda_val)
    
    log("Lambda", lambda_val, suffix=suffix)
    log("Critical ratio", beta, suffix=suffix)
    log(label, Q, unit="units", suffix=suffix)
    
    return Q

def newsvendor_gamma(mean, std, beta, label="Newsvendor Gamma", suffix=""):
    """
    Newsvendor order quantity for Gamma distributed demand
    
    Parameters:
        mean: Mean demand
        std: Standard deviation of demand
        beta: Critical ratio
        label: Optional label for logging output (default: "Newsvendor Gamma")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal order quantity
    """
    var = std**2
    alpha = mean**2 / var
    theta = var / mean
    
    log("Mean demand", mean, suffix=suffix)
    log("Std deviation", std, suffix=suffix)
    log("Critical ratio", beta, suffix=suffix)
    log("Gamma alpha", alpha, suffix=suffix)
    log("Gamma theta", theta, suffix=suffix)
    
    Q = gamma.ppf(beta, alpha, scale=theta)
    log(label, Q, unit="units", suffix=suffix)
    
    return Q

def newsvendor_kpi(q, mu, sigma, p, c, g=0, label="Newsvendor KPI", suffix=""):
    """
    Calculate key performance indicators for newsvendor model
    
    Parameters:
        q: Order quantity
        mu: Mean demand
        sigma: Standard deviation of demand
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
        label: Optional label for logging output (default: "Newsvendor KPI")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
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
    
    log(f"{label} (order quantity)", q, unit="units", suffix=suffix)
    log(f"{label} (mean demand)", mu, suffix=suffix)
    log(f"{label} (standard deviation)", sigma, suffix=suffix)
    log(f"{label} (z-value)", z, suffix=suffix)
    log(f"{label} (expected lost sales)", ELS, unit="units", suffix=suffix)
    log(f"{label} (expected sales)", ES, unit="units", suffix=suffix)
    log(f"{label} (expected leftover)", ELO, unit="units", suffix=suffix)
    log(f"{label} (expected profit)", EP, suffix=suffix)
    log(f"{label} (availability)", alpha, suffix=suffix)
    log(f"{label} (fill rate)", fill_rate, suffix=suffix)
    
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

def newsvendor_find_z_for_fillrate(beta, mu, sigma, label="Z for fill rate", suffix=""):
    """
    Find z-value to achieve desired fill rate (beta)
    
    Parameters:
        beta: Desired fill rate
        mu: Mean demand
        sigma: Standard deviation
        label: Optional label for logging output (default: "Z for fill rate")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        z-value for standard normal distribution
    """
    def f(x):
        return abs((norm.pdf(x) - x * (1 - norm.cdf(x))) - (1 - beta) * mu / sigma)
    
    log("Desired fill rate", beta, suffix=suffix)
    log("Mean demand", mu, suffix=suffix)
    log("Standard deviation", sigma, suffix=suffix)
    
    z = minimize_scalar(f, method="golden").x
    log(label, z, suffix=suffix)
    
    return z


#####################################################
# Safety Stock and Reorder Points
#####################################################

def reorder_point(D, lead_time_months, label="Reorder point", suffix=""):
    """
    Calculate reorder point (ROP) for constant demand
    
    Parameters:
        D: Annual demand
        lead_time_months: Lead time in months
        label: Optional label for logging output (default: "Reorder point")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Reorder point
    """
    monthly_demand = D / 12
    rop = monthly_demand * lead_time_months
    
    log("Annual demand", D, suffix=suffix)
    log("Lead time (months)", lead_time_months, suffix=suffix)
    log("Monthly demand", monthly_demand, suffix=suffix)
    log(label, rop, unit="units", suffix=suffix)
    
    return rop

def safety_stock(z, std_dev, lead_time, review_period=0, label="Safety stock", suffix=""):
    """
    Safety stock calculation
    
    Parameters:
        z: Safety factor
        std_dev: Standard deviation of demand
        lead_time: Lead time
        review_period: Review period (default 0)
        label: Optional label for logging output (default: "Safety stock")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Safety stock
    """
    ss = z * std_dev * math.sqrt(lead_time + review_period)
    log(label, ss, unit="units", suffix=suffix)
    return ss

def service_level_safety_stock(mean_demand, std_demand, lead_time, service_level, label="Service level safety stock", suffix=""):
    """
    Calculate safety stock for a given service level
    
    Parameters:
        mean_demand: Mean demand
        std_demand: Standard deviation of demand
        lead_time: Lead time
        service_level: Desired service level (probability)
        label: Optional label for logging output (default: "Service level safety stock")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Reorder point
    """
    z = norm.ppf(service_level)
    ss = z * std_demand * math.sqrt(lead_time)
    reorder_pt = mean_demand * lead_time + ss
    
    log("Mean demand", mean_demand, suffix=suffix)
    log("Std demand", std_demand, suffix=suffix)
    log("Lead time", lead_time, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    log("Z-value", z, suffix=suffix)
    log(f"{label}", ss, unit="units", suffix=suffix)
    log("Reorder point", reorder_pt, unit="units", suffix=suffix)
    
    return ss, reorder_pt

def order_up_to_level(forecast, safety_stock, label="Order-up-to level", suffix=""):
    """
    Order-up-to level (S)
    
    Parameters:
        forecast: Demand forecast
        safety_stock: Safety stock
        label: Optional label for logging output (default: "Order-up-to level")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Order-up-to level
    """
    result = np.array(forecast) + safety_stock
    
    log("Forecast", forecast, suffix=suffix)
    log("Safety stock", safety_stock, unit="units", suffix=suffix)
    log(label, result, unit="units", suffix=suffix)
    
    return result

def variable_lead_time_safety_stock(z, mean_demand, std_demand, mean_lead_time, std_lead_time, review_period=0, label="Variable lead time safety stock", suffix=""):
    """
    Safety stock calculation for variable lead time
    
    Parameters:
        z: Safety factor
        mean_demand: Mean demand
        std_demand: Standard deviation of demand
        mean_lead_time: Mean lead time
        std_lead_time: Standard deviation of lead time
        review_period: Review period (default 0)
        label: Optional label for logging output (default: "Variable lead time safety stock")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Safety stock
    """
    log("Z-value", z, suffix=suffix)
    log("Mean demand", mean_demand, suffix=suffix)
    log("Std demand", std_demand, suffix=suffix)
    log("Mean lead time", mean_lead_time, suffix=suffix)
    log("Std lead time", std_lead_time, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    
    term1 = (mean_lead_time + review_period) * std_demand**2
    term2 = mean_demand**2 * std_lead_time**2
    
    log("Demand variation term", term1, suffix=suffix)
    log("Lead time variation term", term2, suffix=suffix)
    
    ss = z * math.sqrt(term1 + term2)
    log(label, ss, unit="units", suffix=suffix)
    
    return ss


#####################################################
# Forecasting Methods
#####################################################

def moving_average(demand, window, label="Moving average forecast", suffix=""):
    """
    Moving average forecast
    
    Parameters:
        demand: Array of historical demand
        window: Window size
        label: Optional label for logging output (default: "Moving average forecast")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Array of moving average forecasts
    """
    result = pd.Series(demand).rolling(window=window).mean().shift(1).to_numpy()
    
    log("Historical demand", demand, suffix=suffix)
    log("Window size", window, suffix=suffix)
    log(label, result, suffix=suffix)
    
    return result

def exp_smoothing(alpha, demand, initial_forecast, label="Exponential smoothing forecast", suffix=""):
    """
    Exponential smoothing forecast
    
    Parameters:
        alpha: Smoothing parameter
        demand: Array of historical demand
        initial_forecast: Initial forecast value
        label: Optional label for logging output (default: "Exponential smoothing forecast")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Array of forecasts
    """
    log("Alpha", alpha, suffix=suffix)
    log("Historical demand", demand, suffix=suffix)
    log("Initial forecast", initial_forecast, suffix=suffix)
    
    forecasts = [initial_forecast]
    for t in range(len(demand)):
        new_forecast = alpha * demand[t] + (1 - alpha) * forecasts[-1]
        forecasts.append(new_forecast)
        log(f"{label} (period {t+1})", new_forecast, suffix=suffix)
    
    result = forecasts[1:]  # Remove initial forecast
    log(label, result, suffix=suffix)
    
    return result

def exponential_smoothing_error(alpha, demand, initial_level=0, label="Exponential smoothing error", suffix=""):
    """
    Compute exponential smoothing error
    
    Parameters:
        alpha: Smoothing parameter
        demand: Array of historical demand
        initial_level: Initial level (default 0)
        label: Optional label for logging output (default: "Exponential smoothing error")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Array of errors
    """
    log("Alpha", alpha, suffix=suffix)
    log("Historical demand", demand, suffix=suffix)
    log("Initial level", initial_level, suffix=suffix)
    
    exp_smoothed = [initial_level]
    for i in range(len(demand)):
        new_level = alpha * demand[i] + (1 - alpha) * exp_smoothed[-1]
        exp_smoothed.append(new_level)
        log(f"{label} (period {i+1})", new_level, suffix=suffix)
    
    result = np.array(exp_smoothed[1:])
    log(label, result, suffix=suffix)
    
    return result

def croston_method(indices, values, alpha=0.2, label="Croston's method", suffix=""):
    """
    Croston's method for intermittent demand
    
    Parameters:
        indices: Array of demand occurrence indices
        values: Array of demand values
        alpha: Smoothing parameter (default 0.2)
        label: Optional label for logging output (default: "Croston's method")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        DataFrame with Croston's method results
    """
    log("Demand indices", indices, suffix=suffix)
    log("Demand values", values, suffix=suffix)
    log("Alpha", alpha, suffix=suffix)
    
    results = []
    
    for i in range(len(values)):
        if i == 0:
            x = np.array([indices[i]])  # Interval
            a = np.array([values[i]])   # Demand size
            forecast_day = np.array([math.floor(indices[i] + x[-1])])
            forecast_quantity = np.array([math.ceil(a[-1])])
            
            log(f"{label} (initial interval)", x[-1], suffix=suffix)
            log(f"{label} (initial demand size)", a[-1], suffix=suffix)
            log(f"{label} (initial forecast day)", forecast_day[-1], suffix=suffix)
            log(f"{label} (initial forecast quantity)", forecast_quantity[-1], suffix=suffix)
        else:
            # Update interval estimate
            interval = (1 - alpha) * x[-1] + alpha * (indices[i] - indices[i - 1])
            x = np.append(x, interval)
            
            # Update demand size estimate
            demand_size = (1 - alpha) * a[-1] + alpha * values[i]
            a = np.append(a, demand_size)
            
            # Forecast next occurrence
            next_day = math.floor(indices[i] + x[-1])
            forecast_day = np.append(forecast_day, next_day)
            
            # Forecast next quantity
            next_qty = math.ceil(a[-1])
            forecast_quantity = np.append(forecast_quantity, next_qty)
            
            log(f"{label} (period {i} interval)", interval, suffix=suffix)
            log(f"{label} (period {i} demand size)", demand_size, suffix=suffix)
            log(f"{label} (period {i} forecast day)", next_day, suffix=suffix)
            log(f"{label} (period {i} forecast quantity)", next_qty, suffix=suffix)
    
    # Organize results into DataFrame
    df = pd.DataFrame({
        "x": x,                               # Interval estimate
        "a": a,                               # Demand size estimate
        "forecast_day": forecast_day,         # Next forecast day
        "forecast_quantity": forecast_quantity # Next forecast quantity
    })
    
    log(f"{label} (results)", "DataFrame created", suffix=suffix)
    
    return df


#####################################################
# Multi-Period Inventory Models
#####################################################

def calculate_luc_criterion(t, z, setup_cost, holding_cost, demands, label="LUC criterion", suffix=""):
    """
    Calculate least unit cost criterion
    
    Parameters:
        t: Starting period
        z: Ending period
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
        label: Optional label for logging output (default: "LUC criterion")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Least unit cost
    """
    holding_periods = [i for i in range(z - t + 1)]
    
    log("Starting period", t, suffix=suffix)
    log("Ending period", z, suffix=suffix)
    log("Setup cost", setup_cost, suffix=suffix)
    log("Holding cost", holding_cost, suffix=suffix)
    log("Demands", demands[t:z+1], suffix=suffix)
    log("Holding periods", holding_periods, suffix=suffix)
    
    holding_cost_term = holding_cost * np.sum(demands[t:z+1] * holding_periods)
    total_demand = np.sum(demands[t:z+1])
    
    log("Holding cost term", holding_cost_term, suffix=suffix)
    log("Total demand", total_demand, suffix=suffix)
    
    unit_cost = (setup_cost + holding_cost_term) / total_demand
    log(label, unit_cost, suffix=suffix)
    
    return unit_cost

def calculate_sm_criterion(t, z, setup_cost, holding_cost, demands, label="Silver-Meal criterion", suffix=""):
    """
    Calculate Silver-Meal criterion
    
    Parameters:
        t: Starting period
        z: Ending period
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
        label: Optional label for logging output (default: "Silver-Meal criterion")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Silver-Meal criterion value
    """
    holding_periods = [i for i in range(z - t + 1)]
    
    log("Starting period", t, suffix=suffix)
    log("Ending period", z, suffix=suffix)
    log("Setup cost", setup_cost, suffix=suffix)
    log("Holding cost", holding_cost, suffix=suffix)
    log("Demands", demands[t:z+1], suffix=suffix)
    log("Holding periods", holding_periods, suffix=suffix)
    
    holding_cost_term = holding_cost * np.sum(demands[t:z+1] * holding_periods)
    total_periods = z - t + 1
    
    log("Holding cost term", holding_cost_term, suffix=suffix)
    log("Total periods", total_periods, suffix=suffix)
    
    period_cost = (setup_cost + holding_cost_term) / total_periods
    log(label, period_cost, suffix=suffix)
    
    return period_cost

def make_lot_sizing_decision(func_cost, num_periods, setup_cost, holding_cost, demands, label="Lot sizing decision", suffix=""):
    """
    Make lot-sizing decision using a cost criterion function
    
    Parameters:
        func_cost: Cost criterion function (LUC or SM)
        num_periods: Number of periods
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
        label: Optional label for logging output (default: "Lot sizing decision")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Tuple of setup decision and lot size arrays
    """
    log("Number of periods", num_periods, suffix=suffix)
    log("Setup cost", setup_cost, suffix=suffix)
    log("Holding cost", holding_cost, suffix=suffix)
    log("Demands", demands, suffix=suffix)
    
    flag_setup = np.full(num_periods, False, dtype=bool)  # Setup indicator
    lot_size = np.zeros(num_periods)  # Lot size for each period
    
    t = 0
    while t < num_periods:
        z = t
        c_opt = func_cost(t, z, setup_cost, holding_cost, demands, label=f"{label} (criterion)", suffix=suffix)
        log(f"{label} (period {t}, initial cost)", c_opt, suffix=suffix)
        
        while c_opt > func_cost(t, z + 1, setup_cost, holding_cost, demands, label=f"{label} (criterion)", suffix=suffix):
            z += 1
            c_opt = func_cost(t, z, setup_cost, holding_cost, demands, label=f"{label} (criterion)", suffix=suffix)
            log(f"{label} (period {t}, extended to {z}, cost)", c_opt, suffix=suffix)
            if z == num_periods - 1:
                log(f"{label} (reached end of horizon)", "Yes", suffix=suffix)
                break
        
        flag_setup[t] = True
        lot_size[t] = np.sum(demands[t:z+1])
        log(f"{label} (setup in period {t})", "Yes", suffix=suffix)
        log(f"{label} (lot size in period {t})", lot_size[t], suffix=suffix)
        log(f"{label} (covers periods {t} to {z})", "Yes", suffix=suffix)
        
        t = z + 1
    
    log(f"{label} (setup decisions)", flag_setup, suffix=suffix)
    log(f"{label} (lot sizes)", lot_size, suffix=suffix)
    
    return flag_setup, lot_size

def calculate_total_cost(flag_setup, lot_size, demands, setup_cost, holding_cost, label="Total lot sizing cost", suffix=""):
    """
    Calculate total cost for a lot-sizing decision
    
    Parameters:
        flag_setup: Array of setup indicators
        lot_size: Array of lot sizes
        demands: Array of demands
        setup_cost: Setup cost
        holding_cost: Holding cost
        label: Optional label for logging output (default: "Total lot sizing cost")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Total cost
    """
    log("Setup decisions", flag_setup, suffix=suffix)
    log("Lot sizes", lot_size, suffix=suffix)
    log("Demands", demands, suffix=suffix)
    log("Setup cost", setup_cost, suffix=suffix)
    log("Holding cost", holding_cost, suffix=suffix)
    
    # Setup cost
    total_setup_cost = setup_cost * np.sum(flag_setup)
    log(f"{label} (setup cost)", total_setup_cost, suffix=suffix)
    
    # Inventory holding cost
    num_periods = len(demands)
    inventory = np.zeros(num_periods)
    inventory[0] = lot_size[0] - demands[0]
    
    for t in range(1, num_periods):
        inventory[t] = inventory[t-1] + lot_size[t] - demands[t]
    
    log(f"{label} (inventory levels)", inventory, suffix=suffix)
    
    total_holding_cost = holding_cost * np.sum(inventory)
    log(f"{label} (holding cost)", total_holding_cost, suffix=suffix)
    
    total_cost = total_setup_cost + total_holding_cost
    log(label, total_cost, suffix=suffix)
    
    return total_cost

def wagner_whitin(num_periods, setup_cost, holding_cost, demands, label="Wagner-Whitin algorithm", suffix=""):
    """
    Wagner-Whitin algorithm for lot-sizing
    
    Parameters:
        num_periods: Number of periods
        setup_cost: Setup cost
        holding_cost: Holding cost
        demands: Array of demands
        label: Optional label for logging output (default: "Wagner-Whitin algorithm")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Tuple of setup decision array and total cost
    """
    log("Number of periods", num_periods, suffix=suffix)
    log("Setup cost", setup_cost, suffix=suffix)
    log("Holding cost", holding_cost, suffix=suffix)
    log("Demands", demands, suffix=suffix)
    
    # 2D array for total costs
    costs = np.full((num_periods, num_periods), np.inf)
    
    # Option 1: Order in the first period
    costs[0, 0] = setup_cost
    for t in range(1, num_periods):
        costs[0, t] = costs[0, t-1] + t * holding_cost * demands[t]
    
    log(f"{label} (first period costs)", costs[0, :], suffix=suffix)
    
    # Options 2...n: Order in period j
    for j in range(1, num_periods):
        costs[j, j] = np.min(costs[:, j-1]) + setup_cost
        for t in range(j+1, num_periods):
            costs[j, t] = costs[j, t-1] + (t - j) * holding_cost * demands[t]
        log(f"{label} (period {j} costs)", costs[j, :], suffix=suffix)
    
    # Get setup decision
    index_opt = np.argmin(costs, axis=0)
    log(f"{label} (optimal order period indices)", index_opt, suffix=suffix)
    
    setup_decision = np.full(num_periods, False, dtype=bool)
    setup_decision[0] = True
    
    for t in range(1, num_periods):
        if index_opt[t] == index_opt[t-1]:
            setup_decision[t] = False
        else:
            setup_decision[t] = True
    
    optimal_cost = np.min(costs[:, -1])
    log(f"{label} (setup decisions)", setup_decision, suffix=suffix)
    log(f"{label} (optimal cost)", optimal_cost, suffix=suffix)
    
    return setup_decision, optimal_cost


#####################################################
# Multi-Echelon Inventory Control
#####################################################

def common_cycle(A_list, h_list, d_list, label="Common replenishment cycle", suffix=""):
    """
    Common replenishment cycle (T*)
    
    Parameters:
        A_list: List of ordering costs
        h_list: List of holding costs
        d_list: List of demand rates
        label: Optional label for logging output (default: "Common replenishment cycle")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal common cycle time
    """
    log("Ordering costs", A_list, suffix=suffix)
    log("Holding costs", h_list, suffix=suffix)
    log("Demand rates", d_list, suffix=suffix)
    
    total_ordering = 2 * sum(A_list)
    log(f"{label} (ordering term)", total_ordering, suffix=suffix)
    
    total_holding = sum([h * d for h, d in zip(h_list, d_list)])
    log(f"{label} (holding term)", total_holding, suffix=suffix)
    
    result = math.sqrt(total_ordering / total_holding)
    log(label, result, unit="periods", suffix=suffix)
    
    return result

def material_requirement_planning(gross_requirements, arrivals, starting_inventory, safety_stock=0, label="MRP", suffix=""):
    """
    Calculate net requirements using Material Requirements Planning (MRP)
    
    Parameters:
        gross_requirements: Array of gross requirements
        arrivals: Array of scheduled arrivals
        starting_inventory: Initial inventory
        safety_stock: Safety stock (default 0)
        label: Optional label for logging output (default: "MRP")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Array of net requirements
    """
    log("Gross requirements", gross_requirements, suffix=suffix)
    log("Scheduled arrivals", arrivals, suffix=suffix)
    log("Starting inventory", starting_inventory, suffix=suffix)
    log("Safety stock", safety_stock, suffix=suffix)
    
    net_requirements = []
    projected_inventory = starting_inventory
    
    # Calculate net requirement for each period
    for i in range(len(gross_requirements)):
        log(f"{label} (period {i} begin inventory)", projected_inventory, suffix=suffix)
        
        if i == 0:
            projected_inventory = projected_inventory + arrivals[i]
            log(f"{label} (period {i} after arrivals)", projected_inventory, suffix=suffix)
            
            net_requirement = max(0, safety_stock + gross_requirements[i] - projected_inventory)
            log(f"{label} (period {i} net requirement)", net_requirement, suffix=suffix)
        else:
            projected_inventory = projected_inventory - gross_requirements[i-1] + arrivals[i] + net_requirements[-1]
            log(f"{label} (period {i} after arrivals)", projected_inventory, suffix=suffix)
            
            net_requirement = max(0, gross_requirements[i] + safety_stock - projected_inventory)
            log(f"{label} (period {i} net requirement)", net_requirement, suffix=suffix)
        
        net_requirements.append(net_requirement)
    
    log(f"{label} (net requirements)", net_requirements, suffix=suffix)
    
    return net_requirements

def serial_system_echelon_stock(inventory_positions, net_requirements, label="Echelon stock", suffix=""):
    """
    Calculate echelon inventory positions for a serial system
    
    Parameters:
        inventory_positions: List of local inventory positions
        net_requirements: List of net requirements
        label: Optional label for logging output (default: "Echelon stock")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        List of echelon inventory positions
    """
    log("Local inventory positions", inventory_positions, suffix=suffix)
    log("Net requirements", net_requirements, suffix=suffix)
    
    n = len(inventory_positions)
    echelon_positions = []
    
    for i in range(n):
        if i == 0:
            echelon_positions.append(inventory_positions[i])
            log(f"{label} (stage {i} echelon)", echelon_positions[i], suffix=suffix)
        else:
            upstream_inventory = sum(net_requirements[0:i])
            log(f"{label} (stage {i} upstream inventory)", upstream_inventory, suffix=suffix)
            
            echelon_positions.append(inventory_positions[i] + upstream_inventory)
            log(f"{label} (stage {i} echelon)", echelon_positions[i], suffix=suffix)
    
    log(label, echelon_positions, suffix=suffix)
    
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

def xyz_classification(sales, label="XYZ classification", suffix=""):
    """
    XYZ classification based on coefficient of variation
    
    Parameters:
        sales: Array of sales data
        label: Optional label for logging output (default: "XYZ classification")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Classification ('X', 'Y', or 'Z')
    """
    log("Sales data", sales, suffix=suffix)
    
    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    
    log("Mean sales", mean_sales, suffix=suffix)
    log("Standard deviation", std_sales, suffix=suffix)
    
    if mean_sales == 0:
        log("Mean sales is zero", "Using Z classification", suffix=suffix)
        log(label, "Z", suffix=suffix)
        return "Z"  # Avoid division by zero
    
    cv = std_sales / mean_sales
    log("Coefficient of variation", cv, suffix=suffix)
    
    if cv <= 0.5:
        result = "X"  # Very stable demand
        log("CV <= 0.5", "X classification (very stable)", suffix=suffix)
    elif cv < 1.25:
        result = "Y"  # Moderate variability
        log("0.5 < CV < 1.25", "Y classification (moderate variability)", suffix=suffix)
    else:
        result = "Z"  # High variability
        log("CV >= 1.25", "Z classification (high variability)", suffix=suffix)
    
    log(label, result, suffix=suffix)
    return result

def extended_xyz_classification(sales, label="Extended XYZ", suffix=""):
    """
    Extended XYZ classification with detailed information
    
    Parameters:
        sales: Array of sales data
        label: Optional label for logging output (default: "Extended XYZ")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with classification details
    """
    log("Sales data", sales, suffix=suffix)
    
    total_sales = np.sum(sales)
    log("Total sales", total_sales, suffix=suffix)
    
    if total_sales == 0:
        log(f"{label} (zero total sales)", "Using Z classification", suffix=suffix)
        log(f"{label} (CV)", float('inf'), suffix=suffix)
        log(f"{label} (classification)", "Z", suffix=suffix)
        return {"classification": "Z", "cv": float('inf')}
    
    # Calculate coefficient of variation (CV)
    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    cv = std_sales / mean_sales if mean_sales != 0 else float('inf')
    
    log("Mean sales", mean_sales, suffix=suffix)
    log("Standard deviation", std_sales, suffix=suffix)
    log("Coefficient of variation", cv, suffix=suffix)
    
    # Classify based on CV thresholds
    if cv <= 0.5:
        classification = "X"  # Very stable demand
        log("CV <= 0.5", "X classification (very stable)", suffix=suffix)
    elif cv < 1.25:
        classification = "Y"  # Moderate variability
        log("0.5 < CV < 1.25", "Y classification (moderate variability)", suffix=suffix)
    else:
        classification = "Z"  # High variability
        log("CV >= 1.25", "Z classification (high variability)", suffix=suffix)
    
    result = {
        "classification": classification,
        "cv": cv,
        "mean": mean_sales,
        "std": std_sales
    }
    
    log(f"{label} (result)", result, suffix=suffix)
    return result

def sample_variance(data, label="Sample variance", suffix=""):
    """
    Calculate sample variance
    
    Parameters:
        data: Array of data points
        label: Optional label for logging output (default: "Sample variance")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Sample variance
    """
    log("Data points", data, suffix=suffix)
    
    variance = np.var(data, ddof=1)
    log(label, variance, suffix=suffix)
    
    return variance


#####################################################
# Limited Capacity Models
#####################################################

def economic_lot_scheduling_problem(demand_rates, production_rates, setup_costs, holding_costs, label="ELSP", suffix=""):
    """
    Economic Lot Scheduling Problem (ELSP)
    
    Parameters:
        demand_rates: List of demand rates
        production_rates: List of production rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        label: Optional label for logging output (default: "ELSP")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with results
    """
    log("Demand rates", demand_rates, suffix=suffix)
    log("Production rates", production_rates, suffix=suffix)
    log("Setup costs", setup_costs, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    
    n = len(demand_rates)
    log("Number of products", n, suffix=suffix)
    
    # Calculate EOQ for each product
    Q = []
    for i in range(n):
        numerator = 2 * demand_rates[i] * setup_costs[i]
        denominator = holding_costs[i] * (1 - demand_rates[i] / production_rates[i])
        q_value = math.sqrt(numerator / denominator)
        Q.append(q_value)
        log(f"{label} (product {i} EOQ)", q_value, unit="units", suffix=suffix)
    
    # Calculate cycle time for each product
    T = [q / d for q, d in zip(Q, demand_rates)]
    for i in range(n):
        log(f"{label} (product {i} cycle time)", T[i], unit="periods", suffix=suffix)
    
    # Calculate production time for each product
    production_times = [q / p + r for q, p, r in zip(Q, production_rates, [0] * n)]
    for i in range(n):
        log(f"{label} (product {i} production time)", production_times[i], unit="periods", suffix=suffix)
    
    # Calculate cost for each product
    costs = []
    for i in range(n):
        ordering_cost = (demand_rates[i] / Q[i]) * setup_costs[i]
        holding_cost = (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * (Q[i] / production_rates[i])
        total_product_cost = ordering_cost + holding_cost
        costs.append(total_product_cost)
        
        log(f"{label} (product {i} ordering cost)", ordering_cost, suffix=suffix)
        log(f"{label} (product {i} holding cost)", holding_cost, suffix=suffix)
        log(f"{label} (product {i} total cost)", total_product_cost, suffix=suffix)
    
    result = {
        "Q": Q,
        "T": T,
        "production_times": production_times,
        "costs": costs,
        "total_cost": sum(costs),
        "total_production_time": sum(production_times),
        "min_cycle_time": min(T)
    }
    
    log(f"{label} (total cost)", result["total_cost"], suffix=suffix)
    log(f"{label} (total production time)", result["total_production_time"], suffix=suffix)
    log(f"{label} (min cycle time)", result["min_cycle_time"], suffix=suffix)
    
    return result

def common_cycle_approach(demand_rates, production_rates, setup_costs, holding_costs, setup_times=None, label="Common cycle approach", suffix=""):
    """
    Common Cycle Approach for ELSP
    
    Parameters:
        demand_rates: List of demand rates
        production_rates: List of production rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        setup_times: List of setup times (optional)
        label: Optional label for logging output (default: "Common cycle approach")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with results
    """
    log("Demand rates", demand_rates, suffix=suffix)
    log("Production rates", production_rates, suffix=suffix)
    log("Setup costs", setup_costs, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    if setup_times:
        log("Setup times", setup_times, suffix=suffix)
    
    # Calculate unconstrained common cycle time
    numerator = 2 * sum(setup_costs)
    denominator = sum([h * d * (1 - d / p) for h, d, p in zip(holding_costs, demand_rates, production_rates)])
    T_unconstrained = math.sqrt(numerator / denominator)
    
    log(f"{label} (unconstrained numerator)", numerator, suffix=suffix)
    log(f"{label} (unconstrained denominator)", denominator, suffix=suffix)
    log(f"{label} (unconstrained cycle time)", T_unconstrained, unit="periods", suffix=suffix)
    
    # Calculate capacity constraint (if setup times provided)
    if setup_times:
        numerator_c = sum(setup_times)
        denominator_c = 1 - sum([d / p for d, p in zip(demand_rates, production_rates)])
        T_constrained = numerator_c / denominator_c
        
        log(f"{label} (constrained numerator)", numerator_c, suffix=suffix)
        log(f"{label} (constrained denominator)", denominator_c, suffix=suffix)
        log(f"{label} (constrained cycle time)", T_constrained, unit="periods", suffix=suffix)
        
        T_optimal = max(T_unconstrained, T_constrained)
        log(f"{label} (binding constraint)", "Capacity" if T_constrained > T_unconstrained else "Cost", suffix=suffix)
    else:
        T_optimal = T_unconstrained
    
    log(f"{label} (optimal cycle time)", T_optimal, unit="periods", suffix=suffix)
    
    # Calculate lot sizes and costs
    Q_optimal = [d * T_optimal for d in demand_rates]
    costs = []
    
    for i in range(len(demand_rates)):
        setup_term = setup_costs[i] / T_optimal
        holding_term = (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * (demand_rates[i] / production_rates[i]) * T_optimal
        total_product_cost = setup_term + holding_term
        costs.append(total_product_cost)
        
        log(f"{label} (product {i} lot size)", Q_optimal[i], unit="units", suffix=suffix)
        log(f"{label} (product {i} setup cost)", setup_term, suffix=suffix)
        log(f"{label} (product {i} holding cost)", holding_term, suffix=suffix)
        log(f"{label} (product {i} total cost)", total_product_cost, suffix=suffix)
    
    total_cost = sum(costs)
    log(f"{label} (total cost)", total_cost, suffix=suffix)
    
    return {
        "T_optimal": T_optimal,
        "Q_optimal": Q_optimal,
        "costs": costs,
        "total_cost": total_cost
    }

def power_of_two_policy(demand_rates, production_rates, setup_costs, holding_costs, base_period, label="Power-of-two policy", suffix=""):
    """
    Power-of-Two Policy for ELSP
    
    Parameters:
        demand_rates: List of demand rates
        production_rates: List of production rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        base_period: Base period length
        label: Optional label for logging output (default: "Power-of-two policy")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with results
    """
    log("Demand rates", demand_rates, suffix=suffix)
    log("Production rates", production_rates, suffix=suffix)
    log("Setup costs", setup_costs, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Base period", base_period, suffix=suffix)
    
    n = len(demand_rates)
    n_values = [1] * n  # Initialize all products with base frequency
    log(f"{label} (initial n values)", n_values, suffix=suffix)
    
    iteration = 0
    # Iterative improvement
    while True:
        iteration += 1
        log(f"{label} (iteration)", iteration, suffix=suffix)
        
        # Calculate base cycle time
        numerator = sum([2 * setup_costs[i] / n_values[i] for i in range(n)])
        denominator = sum([holding_costs[i] * (production_rates[i] - demand_rates[i]) * 
                          (demand_rates[i] / production_rates[i]) * n_values[i] for i in range(n)])
        base_cycle = math.sqrt(numerator / denominator)
        
        log(f"{label} (base cycle numerator)", numerator, suffix=suffix)
        log(f"{label} (base cycle denominator)", denominator, suffix=suffix)
        log(f"{label} (base cycle time)", base_cycle, unit="periods", suffix=suffix)
        
        # Update n values
        old_n = n_values.copy()
        
        for i in range(n):
            # Try doubling the frequency
            doubling_iterations = 0
            while True:
                doubling_iterations += 1
                cost_n = (setup_costs[i] / (n_values[i] * base_cycle) + 
                        (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * 
                        (demand_rates[i] / production_rates[i]) * n_values[i] * base_cycle)
                
                cost_n_plus_1 = (setup_costs[i] / (n_values[i] * 2 * base_cycle) + 
                                (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * 
                                (demand_rates[i] / production_rates[i]) * n_values[i] * 2 * base_cycle)
                
                log(f"{label} (product {i}, n={n_values[i]}, cost)", cost_n, suffix=suffix)
                log(f"{label} (product {i}, n={n_values[i]*2}, cost)", cost_n_plus_1, suffix=suffix)
                
                if cost_n <= cost_n_plus_1:
                    log(f"{label} (product {i} stopped at n={n_values[i]})", "", suffix=suffix)
                    break
                    
                n_values[i] = n_values[i] * 2
                log(f"{label} (product {i} increased to n={n_values[i]})", "", suffix=suffix)
                
                if doubling_iterations > 10:  # Safety check to prevent infinite loops
                    log(f"{label} (safety break for product {i})", "Too many iterations", suffix=suffix)
                    break
        
        log(f"{label} (updated n values)", n_values, suffix=suffix)
        
        # Check convergence
        if old_n == n_values:
            log(f"{label} (convergence reached)", "Yes", suffix=suffix)
            break
    
    # Calculate lot sizes and costs
    Q = [d * n * base_cycle for d, n in zip(demand_rates, n_values)]
    costs = []
    
    for i in range(n):
        setup_term = setup_costs[i] / (n_values[i] * base_cycle)
        holding_term = (holding_costs[i] / 2) * (production_rates[i] - demand_rates[i]) * (demand_rates[i] / production_rates[i]) * n_values[i] * base_cycle
        total_cost = setup_term + holding_term
        costs.append(total_cost)
        
        log(f"{label} (product {i} lot size)", Q[i], unit="units", suffix=suffix)
        log(f"{label} (product {i} setup cost)", setup_term, suffix=suffix)
        log(f"{label} (product {i} holding cost)", holding_term, suffix=suffix)
        log(f"{label} (product {i} total cost)", total_cost, suffix=suffix)
    
    result = {
        "base_cycle": base_cycle,
        "n_values": n_values,
        "Q": Q,
        "costs": costs,
        "total_cost": sum(costs)
    }
    
    log(f"{label} (total cost)", result["total_cost"], suffix=suffix)
    
    return result

def economic_manufacturing_quantity(d, A, r, p, label="EMQ", suffix=""):
    """
    Economic Manufacturing Quantity (EMQ) with finite production rate
    
    Parameters:
        d: Demand rate
        A: Setup cost
        r: Holding cost rate
        p: Production rate
        label: Optional label for logging output (default: "EMQ")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Optimal lot size
    """
    log("Demand rate", d, suffix=suffix)
    log("Setup cost", A, suffix=suffix)
    log("Holding cost rate", r, suffix=suffix)
    log("Production rate", p, suffix=suffix)
    
    numerator = 2 * d * A
    denominator = r * (1 - d/p)
    
    log(f"{label} (numerator)", numerator, suffix=suffix)
    log(f"{label} (denominator)", denominator, suffix=suffix)
    
    emq = math.sqrt(numerator / denominator)
    log(label, emq, unit="units", suffix=suffix)
    
    return emq

def capacity_constrained_lotsize(d, A, h, a, W, label="Capacity constrained lotsize", suffix=""):
    """
    Calculate lot sizes with storage capacity constraint
    
    Parameters:
        d: List of demand rates
        A: List of setup costs
        h: List of holding costs
        a: List of space requirements per unit
        W: Total warehouse capacity
        label: Optional label for logging output (default: "Capacity constrained lotsize")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with results
    """
    log("Demand rates", d, suffix=suffix)
    log("Setup costs", A, suffix=suffix)
    log("Holding costs", h, suffix=suffix)
    log("Space requirements", a, suffix=suffix)
    log("Warehouse capacity", W, suffix=suffix)
    
    # Calculate unconstrained EOQ
    Q_unconstrained = [math.sqrt((2 * d_i * A_i) / h_i) for d_i, A_i, h_i in zip(d, A, h)]
    total_space = sum([Q_i * a_i for Q_i, a_i in zip(Q_unconstrained, a)])
    
    log(f"{label} (unconstrained EOQ)", Q_unconstrained, unit="units", suffix=suffix)
    log(f"{label} (total space required)", total_space, suffix=suffix)
    
    # If unconstrained solution is feasible, return it
    if total_space <= W:
        log(f"{label} (is constrained)", "No", suffix=suffix)
        
        costs = []
        for i in range(len(d)):
            setup_cost = (d[i] * A[i]) / Q_unconstrained[i]
            holding_cost = (h[i] * Q_unconstrained[i]) / 2
            total_item_cost = setup_cost + holding_cost
            costs.append(total_item_cost)
            
            log(f"{label} (item {i} setup cost)", setup_cost, suffix=suffix)
            log(f"{label} (item {i} holding cost)", holding_cost, suffix=suffix)
            log(f"{label} (item {i} total cost)", total_item_cost, suffix=suffix)
        
        total_cost = sum(costs)
        log(f"{label} (total cost)", total_cost, suffix=suffix)
        
        return {
            "Q": Q_unconstrained,
            "total_space": total_space,
            "total_cost": total_cost,
            "constrained": False
        }
    
    # If unconstrained solution is not feasible, find Lagrange multiplier
    log(f"{label} (is constrained)", "Yes", suffix=suffix)
    
    def equation(lambda_value):
        return sum([a_i * math.sqrt(2 * d_i * A_i / (h_i + 2 * lambda_value * a_i)) 
                   for d_i, A_i, h_i, a_i in zip(d, A, h, a)]) - W
    
    result = root(equation, x0=0)
    lambda_value = result.x[0]
    log(f"{label} (Lagrange multiplier)", lambda_value, suffix=suffix)
    
    # Calculate constrained solution
    Q_constrained = []
    for i in range(len(d)):
        q = math.sqrt((2 * d[i] * A[i]) / (h[i] + 2 * lambda_value * a[i]))
        Q_constrained.append(q)
        log(f"{label} (item {i} constrained Q)", q, unit="units", suffix=suffix)
    
    total_space = sum([Q_i * a_i for Q_i, a_i in zip(Q_constrained, a)])
    log(f"{label} (total constrained space)", total_space, suffix=suffix)
    
    costs = []
    for i in range(len(d)):
        setup_cost = (d[i] * A[i]) / Q_constrained[i]
        holding_cost = (h[i] * Q_constrained[i]) / 2
        total_item_cost = setup_cost + holding_cost
        costs.append(total_item_cost)
        
        log(f"{label} (item {i} setup cost)", setup_cost, suffix=suffix)
        log(f"{label} (item {i} holding cost)", holding_cost, suffix=suffix)
        log(f"{label} (item {i} total cost)", total_item_cost, suffix=suffix)
    
    total_cost = sum(costs)
    log(f"{label} (total cost)", total_cost, suffix=suffix)
    
    return {
        "Q": Q_constrained,
        "total_space": total_space,
        "total_cost": total_cost,
        "constrained": True,
        "lambda": lambda_value
    }

def joint_replenishment(d, A, h, A0, label="Joint replenishment", suffix=""):
    """
    Joint Replenishment Problem
    
    Parameters:
        d: List of demand rates
        A: List of individual setup costs
        h: List of holding costs
        A0: Major setup cost
        label: Optional label for logging output (default: "Joint replenishment")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with results
    """
    log("Demand rates", d, suffix=suffix)
    log("Individual setup costs", A, suffix=suffix)
    log("Holding costs", h, suffix=suffix)
    log("Major setup cost", A0, suffix=suffix)
    
    n = len(d)
    log("Number of items", n, suffix=suffix)
    
    # Calculate individual optimal cycle times
    T_i = [math.sqrt(2 * A_i / (h_i * d_i)) for A_i, h_i, d_i in zip(A, h, d)]
    log(f"{label} (individual cycle times)", T_i, suffix=suffix)
    
    # Find product with minimum cycle time
    min_index = T_i.index(min(T_i))
    log(f"{label} (item with min cycle time)", min_index, suffix=suffix)
    
    # Initialize n values
    n_i = [math.sqrt(A_i * h[min_index] * d[min_index] / (h_i * d_i * (A0 + A[min_index]))) 
           for A_i, h_i, d_i in zip(A, h, d)]
    n_i = [round(value) for value in n_i]
    log(f"{label} (initial n values)", n_i, suffix=suffix)
    
    # Iterative improvement
    n = n_i
    iteration = 0
    
    while True:
        iteration += 1
        log(f"{label} (iteration)", iteration, suffix=suffix)
        
        # Calculate base cycle time
        sum_A_div_n = A0 + sum([A_i / n_i for A_i, n_i in zip(A, n)])
        sum_h_d_n = sum([h_i * d_i * n_i for h_i, d_i, n_i in zip(h, d, n)])
        T = math.sqrt(2 * sum_A_div_n / sum_h_d_n)
        
        log(f"{label} (setup term)", sum_A_div_n, suffix=suffix)
        log(f"{label} (holding term)", sum_h_d_n, suffix=suffix)
        log(f"{label} (base cycle time)", T, unit="periods", suffix=suffix)
        
        # Store old n values
        old_n = n.copy()
        
        # Update n values
        for i in range(len(n)):
            old_ni = n[i]
            while True:
                if (n[i] * (n[i] + 1) >= 2 * A[i] / (h[i] * d[i] * T**2)):
                    break
                n[i] = n[i] + 1
            
            if old_ni != n[i]:
                log(f"{label} (item {i} n value updated)", f"from {old_ni} to {n[i]}", suffix=suffix)
        
        # Recalculate base cycle time
        sum_A_div_n = A0 + sum([A_i / n_i for A_i, n_i in zip(A, n)])
        sum_h_d_n = sum([h_i * d_i * n_i for h_i, d_i, n_i in zip(h, d, n)])
        T = math.sqrt(2 * sum_A_div_n / sum_h_d_n)
        
        log(f"{label} (updated base cycle time)", T, unit="periods", suffix=suffix)
        
        # Check convergence
        if old_n == n:
            log(f"{label} (convergence reached)", "Yes", suffix=suffix)
            break
    
    individual_cycles = [n_i * T for n_i in n]
    order_quantities = [d_i * n_i * T for d_i, n_i in zip(d, n)]
    
    for i in range(n):
        log(f"{label} (item {i} n value)", n[i], suffix=suffix)
        log(f"{label} (item {i} cycle time)", individual_cycles[i], unit="periods", suffix=suffix)
        log(f"{label} (item {i} order quantity)", order_quantities[i], unit="units", suffix=suffix)
    
    return {
        "base_cycle": T,
        "n_values": n,
        "individual_cycles": individual_cycles,
        "order_quantities": order_quantities
    }


#####################################################
# Risk Pooling and Correlation
#####################################################

def risk_pooling_benefit(demand_means, demand_stds, correlation_matrix, service_level, label="Risk pooling benefit", suffix=""):
    """
    Calculate risk pooling benefit
    
    Parameters:
        demand_means: List of mean demands
        demand_stds: List of standard deviations
        correlation_matrix: Correlation matrix between demands
        service_level: Service level
        label: Optional label for logging output (default: "Risk pooling benefit")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with results
    """
    log("Demand means", demand_means, suffix=suffix)
    log("Demand standard deviations", demand_stds, suffix=suffix)
    log("Correlation matrix", correlation_matrix, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    
    n = len(demand_means)
    z = norm.ppf(service_level)
    log("Z-value", z, suffix=suffix)
    
    # Calculate individual safety stocks
    individual_ss = [z * std for std in demand_stds]
    individual_total = sum(individual_ss)
    
    log("Individual safety stocks", individual_ss, unit="units", suffix=suffix)
    log("Total individual safety stock", individual_total, unit="units", suffix=suffix)
    
    # Calculate pooled demand parameters
    pooled_mean = sum(demand_means)
    log("Pooled mean demand", pooled_mean, suffix=suffix)
    
    # Calculate pooled standard deviation with correlation
    pooled_var = 0
    for i in range(n):
        for j in range(n):
            term = demand_stds[i] * demand_stds[j] * correlation_matrix[i][j]
            pooled_var += term
            log(f"Variance term ({i},{j})", term, suffix=suffix)
    
    log("Pooled variance", pooled_var, suffix=suffix)
    
    pooled_std = math.sqrt(pooled_var)
    log("Pooled standard deviation", pooled_std, suffix=suffix)
    
    pooled_ss = z * pooled_std
    log("Pooled safety stock", pooled_ss, unit="units", suffix=suffix)
    
    # Calculate benefit
    absolute_benefit = individual_total - pooled_ss
    relative_benefit = absolute_benefit / individual_total * 100
    
    log(f"{label} (absolute)", absolute_benefit, unit="units", suffix=suffix)
    log(f"{label} (relative)", relative_benefit, unit="%", suffix=suffix)
    
    return {
        "individual_safety_stocks": individual_ss,
        "individual_total": individual_total,
        "pooled_mean": pooled_mean,
        "pooled_std": pooled_std,
        "pooled_safety_stock": pooled_ss,
        "absolute_benefit": absolute_benefit,
        "relative_benefit": relative_benefit
    }

def risk_pooling_correlation(mu_A, sigma_A, mu_B, sigma_B, rho, p, c, g=0, label="Risk pooling correlation", suffix=""):
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
        label: Optional label for logging output (default: "Risk pooling correlation")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Dictionary with individual and joint order quantities
    """
    log("Mean demand A", mu_A, suffix=suffix)
    log("Std dev A", sigma_A, suffix=suffix)
    log("Mean demand B", mu_B, suffix=suffix)
    log("Std dev B", sigma_B, suffix=suffix)
    log("Correlation coefficient", rho, suffix=suffix)
    log("Selling price", p, suffix=suffix)
    log("Unit cost", c, suffix=suffix)
    log("Salvage value", g, suffix=suffix)
    
    # Critical ratio and z-score
    CR = (p - c) / (p - g)
    z_CR = norm.ppf(CR)
    
    log("Critical ratio", CR, suffix=suffix)
    log("Z-value", z_CR, suffix=suffix)
    
    # Individual order quantities
    Q_A = mu_A + z_CR * sigma_A
    Q_B = mu_B + z_CR * sigma_B
    
    log("Order quantity A", Q_A, unit="units", suffix=suffix)
    log("Order quantity B", Q_B, unit="units", suffix=suffix)
    log("Sum of individual order quantities", Q_A + Q_B, unit="units", suffix=suffix)
    
    # Joint demand parameters
    mu_joint = mu_A + mu_B
    sigma_joint = math.sqrt(sigma_A**2 + sigma_B**2 + 2 * rho * sigma_A * sigma_B)
    
    log("Joint mean", mu_joint, suffix=suffix)
    log("Joint standard deviation", sigma_joint, suffix=suffix)
    
    # Joint order quantity
    Q_joint = mu_joint + z_CR * sigma_joint
    log("Joint order quantity", Q_joint, unit="units", suffix=suffix)
    
    # Calculate effect of correlation
    effect = (Q_A + Q_B) - Q_joint
    log(f"{label} (pooling effect)", effect, unit="units", suffix=suffix)
    
    CV_A = sigma_A / mu_A
    CV_B = sigma_B / mu_B
    CV_joint = sigma_joint / mu_joint
    
    log("CV A", CV_A, suffix=suffix)
    log("CV B", CV_B, suffix=suffix)
    log("Joint CV", CV_joint, suffix=suffix)
    
    return {
        "Q_A": Q_A,
        "Q_B": Q_B,
        "sum_individual": Q_A + Q_B,
        "mu_joint": mu_joint,
        "sigma_joint": sigma_joint,
        "Q_joint": Q_joint,
        "effect": effect,
        "CV_A": CV_A,
        "CV_B": CV_B,
        "CV_joint": CV_joint
    }

def coefficient_of_correlation(data1, data2, label="Correlation coefficient", suffix=""):
    """
    Calculate coefficient of correlation between two datasets
    
    Parameters:
        data1: First dataset
        data2: Second dataset
        label: Optional label for logging output (default: "Correlation coefficient")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Correlation coefficient
    """
    log("Dataset 1", data1, suffix=suffix)
    log("Dataset 2", data2, suffix=suffix)
    
    correlation = np.corrcoef(data1, data2)[0, 1]
    log(label, correlation, suffix=suffix)
    
    return correlation

def bullwhip_effect(orders_variance, demand_variance, label="Bullwhip effect", suffix=""):
    """
    Calculate bullwhip effect ratio
    
    Parameters:
        orders_variance: Variance of orders
        demand_variance: Variance of demand
        label: Optional label for logging output (default: "Bullwhip effect")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
    
    Returns:
        Bullwhip effect ratio
    """
    ratio = orders_variance / demand_variance
    
    # Log input parameters
    log("Orders variance", orders_variance, suffix=suffix)
    log("Demand variance", demand_variance, suffix=suffix)
    log(label, ratio, suffix=suffix)
    
    return ratio


def compare_scenarios(value1, value2, label="Comparison", unit="", suffix1="A", suffix2="B"):
    """
    Compare two values from different scenarios and calculate the difference
    
    Parameters:
        value1: First value
        value2: Second value
        label: Label for the comparison (default: "Comparison")
        unit: Optional unit for formatting
        suffix1: Suffix for the first value (default: "A")
        suffix2: Suffix for the second value (default: "B")
    
    Returns:
        Absolute difference between values
    """
    abs_diff = abs(value1 - value2)
    perc_diff = abs_diff / min(abs(value1), abs(value2)) * 100
    
    log(f"{label}", value1, unit=unit, suffix=suffix1)
    log(f"{label}", value2, unit=unit, suffix=suffix2)
    log(f"{label} absolute difference", abs_diff, unit=unit)
    log(f"{label} percentage difference", perc_diff, unit="%")
    
    return abs_diff, perc_diff
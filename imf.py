"""
Comprehensive Inventory Management Formulas

This script contains a collection of functions for inventory management calculations,
consolidated from various exercises and examples. Use this as a reference for exam preparation.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gamma, uniform
from scipy.optimize import minimize_scalar, fsolve, root, brentq
from scipy.integrate import quad


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


def total_relevant_cost(Q, D, S, h, label="TRC", suffix="", log_components=True):
    """
    Total Relevant Cost (TRC): ordering + holding cost for given Q.

    Parameters:
        Q: Order quantity
        D: Demand per period
        S: Setup cost per order
        h: Holding cost per unit
        label: Optional label for logging
        suffix: Optional scenario suffix (e.g. A/B)
        log_components: If True, log individual cost components

    Returns:
        Total relevant cost per period
    """
    ordering_cost = (D / Q) * S
    holding_cost = (Q / 2) * h
    total_cost = ordering_cost + holding_cost

    if log_components:
        log(f"{label} (ordering)", ordering_cost, suffix=suffix)
        log(f"{label} (holding)", holding_cost, suffix=suffix)
    log(f"{label} (cost per period)", total_cost, suffix=suffix)

    return total_cost


def cycle_length(q, d, unit="period", label="Cycle length", suffix=""):
    """
    Cycle length between orders

    Parameters:
        q: Order Quantity
        d: Demand per period
        unit: Optional unit for time period (default: "period")
        label: Optional label for logging output (default: "Cycle length")
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        T: Cycle time (periods between orders)
    """
    T = q / d
    log(label, T, unit=unit, suffix=suffix)
    return T


def cost_penalty(q, eoq, label="Percentage Cost Penalty (PCP)", suffix=""):
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
    p = 0.01 * percentage_deviation(q, eoq)
    penalty = 50 * (p**2 / (1 + p))
    log(label, penalty, unit="%", suffix=suffix)
    return penalty


def percentage_deviation(
    q, eoq, label="Percentage deviation of order quantity", suffix=""
):
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


def optimal_power_of_two_cycle(
    D, A, h, label="Power-of-two cycle", suffix="", verbose=True
):
    """
    Find the optimal cycle multiplier t (as a power of two) minimizing cost.

    Parameters:
        D: Demand per period
        A: Setup cost per order
        h: Holding cost per unit
        label: Optional label for logging output
        suffix: Optional suffix for distinguishing scenarios (e.g., A/B)
        verbose: If True, print debug comparisons

    Returns:
        t: Optimal power-of-two multiplier
        p_error: Percentage cost error vs. EOQ policy
    """

    def cycle_cost_T(T):
        return A / T + 0.5 * h * D * T

    def cycle_cost_Q(Q):
        return total_relevant_cost(Q, D, A, h, label=label, suffix=suffix)

    # --- Phase 1: Minimize cost over cycle length T ---
    t = 1
    log(f"{label} (initital t)", t, suffix=suffix)
    while True:
        current_cost = cycle_cost_T(t)
        next_cost = cycle_cost_T(2 * t)

        if current_cost > next_cost:
            print(f"Continue as t={t}: {current_cost:.2f} > t={2 * t}: {next_cost:.2f}")
        else:
            print(f"BREAK as t={t}: {current_cost:.2f} <= t={2 * t}: {next_cost:.2f}")
            break

        t *= 2
        log(f"{label} (updated t)", t, suffix=suffix)

    log(f"{label} (optimal t)", t, suffix=suffix)

    print("\n")

    # --- Phase 2: Minimize cost over lot sizes ---
    t = 1
    log(f"{label} (initital t)", t, suffix=suffix)
    while True:
        current_cost = cycle_cost_Q(D * t)
        next_cost = cycle_cost_Q(2 * D * t)

        if current_cost > next_cost:
            print(f"Continue as t={t}: {current_cost:.2f} > t={2 * t}: {next_cost:.2f}")
        else:
            print(f"BREAK as t={t}: {current_cost:.2f} <= t={2 * t}: {next_cost:.2f}")
            break

        t *= 2
        log(f"{label} (updated t)", t, suffix=suffix)

    log(f"{label} (optimal t)", t, suffix=suffix)

    # --- Final report ---
    EOQ = eoq(D, A, h)
    Q_pow2 = D * t
    p_error = cost_penalty(Q_pow2, EOQ)

    log(f"{label} (percentage error vs EOQ)", p_error, unit="%", suffix=suffix)

    return t, p_error


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
    # if q = EOQ, ordering_cost + holding_cost == sqrt(2*d*A*h)
    ordering_cost = S * (D / Q)
    holding_cost = h * (Q / 2)
    total_cost = purchase_cost + ordering_cost + holding_cost

    log(f"{label} (purchase)", purchase_cost, suffix=suffix)
    log(f"{label} (ordering)", ordering_cost, suffix=suffix)
    log(f"{label} (holding)", holding_cost, suffix=suffix)
    log(label, total_cost, suffix=suffix)

    return total_cost


def eoq_all_unit_quantity_discount(
    d, A, r, bp, cp, label="All-unit quantity discount", suffix=""
):
    """
    EOQ model with all-unit quantity discounts.

    This function determines the optimal order quantity (EOQ) in the presence of
    all-unit quantity discounts, where the unit purchase price decreases at certain
    quantity breakpoints. It evaluates EOQ and total cost at each discount level,
    adjusting for feasibility and selecting the quantity with the lowest total cost.

    Parameters:
        d (float): Annual demand (units/year)
        A (float): Ordering or setup cost per order
        r (float): Holding cost rate (e.g., 0.2 for 20% per year)
        bp (list of float): Breakpoints for order quantities.
                            Must be sorted in ascending order.
                            Each breakpoint `bp[i]` defines the minimum order
                            quantity for price `cp[i]`.
                            The first tier applies to quantities: bp[0] ≤ Q < bp[1],
                            and so on. The last tier covers Q ≥ bp[-1].
        cp (list of float): Corresponding unit prices at each breakpoint (same length as `bp`)
        label (str, optional): Label used for logging output (default: "All-unit quantity discount")
        suffix (str, optional): Suffix to differentiate scenarios in logs (default: "")

    Returns:
        q_opt (float): Optimal order quantity minimizing total cost
        c_opt (float): Corresponding minimum total cost including purchase, ordering, and holding

    Notes:
        - The total cost includes:
            - Ordering cost = (d / Q) * A
            - Holding cost = 0.5 * r * unit_price * Q
            - Purchase cost = unit_price * d
        - EOQs that fall below their tier’s breakpoint are rounded up to the minimum
          quantity allowed in that tier (if cheaper).
        - The algorithm evaluates EOQ and breakpoint costs for all tiers, and returns
          the quantity with the lowest feasible total cost.
    """
    best_cost = float("inf")
    best_q = None

    for i in range(len(cp)):
        price = cp[i]
        q_min = bp[i]
        h = r * price

        q_eoq = math.sqrt(2 * A * d / h)
        q_used = max(q_eoq, q_min)  # Step 2: Adjust EOQ if not feasible

        # Step 3: Total cost = setup + holding + purchasing
        total_cost = (d / q_used) * A + 0.5 * h * q_used + d * price

        # Logging
        log(f"{label} (tier {i}) unit price", price, suffix=suffix)
        log(f"{label} (tier {i}) EOQ", q_eoq, unit="units", suffix=suffix)
        log(f"{label} (tier {i}) adjusted q", q_used, unit="units", suffix=suffix)
        log(f"{label} (tier {i}) total cost", total_cost, suffix=suffix)

        # Step 4: Keep track of the best
        if total_cost < best_cost:
            best_cost = total_cost
            best_q = q_used

    # Final output
    log(f"{label} (optimal quantity)", best_q, unit="units", suffix=suffix)
    log(f"{label} (optimal cost)", best_cost, suffix=suffix)

    return best_q, best_cost




def eoq_incremental_quantity_discount(
    d, A, r, bp, cp, label="Incremental quantity discount", suffix=""
):
    """
    EOQ model with incremental quantity discounts.

    This model computes the optimal order quantity and associated cost when unit prices
    follow an incremental discount structure. That is, only the units ordered beyond each
    breakpoint receive the lower unit price.

    Parameters:
        d (float): Annual demand
        A (float): Setup cost per order
        r (float): Annual holding cost rate (e.g., 0.2 for 20%)
        bp (list of float): Breakpoints for quantity (must be sorted ascending, starting with 0)
        cp (list of float): Unit prices per quantity tier (same length as bp)
        label (str): Optional label for logging
        suffix (str): Optional suffix for scenario tracking/logging

    Returns:
        q_opt (float): Optimal order quantity
        c_opt (float): Minimum total cost
    """

    n = len(bp)
    assert len(cp) == n, "Breakpoints and price tiers must have the same length"
    assert bp[0] == 0, "Breakpoints should start at 0 for incremental discounts"

    # Step 1: Compute cumulative purchasing cost R[t] up to each breakpoint
    R = np.zeros(n)
    for t in range(1, n):
        R[t] = cp[t - 1] * (bp[t] - bp[t - 1]) + R[t - 1]

    qt = np.zeros(len(bp))
    flag_feasible = np.full(len(bp), False, dtype=bool)  # the indicator of feasibility
    for t in range(len(bp)):
        qt[t] = math.sqrt(2 * (R[t] - cp[t] * bp[t] + A) * d / (r * cp[t]))
    for t in range(len(bp) - 1):
        flag_feasible[t] = True if qt[t] >= bp[t] and qt[t] < bp[t + 1] else False
    flag_feasible[-1] = True if qt[-1] >= bp[-1] else False

    # Step 4: Compute total cost for each feasible EOQ
    qt_f = []
    cost_qt_f = []

    def total_cost(q, h, c):
        return d / q * A + 0.5 * h * q + c * d

    for t in range(n):
        if flag_feasible[t]:
            q = qt[t]
            # Effective average price under incremental scheme
            avg_price = (R[t] + cp[t] * (q - bp[t])) / q
            h = avg_price * r
            cost = total_cost(q, h, avg_price)

            log(f"{label} (feasible tier {t}) quantity", q, unit="units", suffix=suffix)
            log(f"{label} (feasible tier {t}) cost", cost, suffix=suffix)

            qt_f.append(q)
            cost_qt_f.append(cost)

    if not qt_f:
        raise ValueError("No feasible EOQ found for any price tier.")

    idx = np.argmin(cost_qt_f)
    q_opt = qt_f[idx]
    c_opt = cost_qt_f[idx]

    log(f"{label} (optimal quantity)", q_opt, unit="units", suffix=suffix)
    log(f"{label} (optimal cost)", c_opt, suffix=suffix)

    return q_opt, c_opt


def eoq_sensitivity_analysis(
    q_actual, q_optimal, label="EOQ sensitivity analysis", suffix=""
):
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


def eoq_sensitivity_analysis_complete(
    q_actual, q_optimal, D, A, h, c=None, label="EOQ sensitivity analysis", suffix=""
):
    """
    EOQ sensitivity analysis with optional total cost comparison

    Parameters:
        q_actual: Actual order quantity
        q_optimal: Economic order quantity
        D: Annual demand
        A: Setup cost per order
        h: Holding cost per unit per year
        c: Unit purchase cost (optional)
        label: Optional label for logging
        suffix: Optional suffix for logging output

    Returns:
        deviation (%), PCP (approx), PCP (exact), TCC (% if c given)
    """

    approx_pcp = cost_penalty(q_actual, q_optimal, suffix=suffix)

    # --- Exact TRC comparison ---
    trc_opt = total_relevant_cost(q_optimal, D, A, h, label="TRC (EOQ)", suffix=suffix)
    trc_actual = total_relevant_cost(
        q_actual, D, A, h, label="TRC (actual)", suffix=suffix
    )
    exact_pcp = 100 * (trc_actual - trc_opt) / trc_opt
    log(f"{label} (exact PCP)", exact_pcp, unit="%", suffix=suffix)

    # --- Optional total cost comparison ---
    if c is not None:
        tc_opt = total_annual_cost(
            D, A, h, q_optimal, c, label="Total cost (EOQ)", suffix=suffix
        )
        tc_actual = total_annual_cost(
            D, A, h, q_actual, c, label="Total cost (actual)", suffix=suffix
        )
        tcc = 100 * (tc_actual - tc_opt) / tc_opt
        log(f"{label} (total cost comparison)", tcc, unit="%", suffix=suffix)
    else:
        tcc = None

    return approx_pcp, exact_pcp, tcc


#####################################################
# Newsvendor Models
#####################################################


def newsvendor_critical_ratio(p, c, g=0, label="Critical ratio", suffix=""):
    """
    Newsvendor critical ratio (used for finding optimal order quantity)

    Parameters:
        p: Selling price
        c: Cost per unit
        g: Salvage value (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Critical ratio
    """
    CR = (p - c) / (p - g)
    log(label, CR, suffix=suffix)
    return CR


def newsvendor_critical_fractile(co, cu, label="Critical fractile", suffix=""):
    """
    Calculate critical fractile based on overage and underage costs
    
    Parameters:
        co: Overage cost (cost of excess inventory)
        cu: Underage cost (cost of shortage)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Critical fractile (optimal service level)
    """
    log("Overage cost", co, suffix=suffix)
    log("Underage cost", cu, suffix=suffix)
    
    critical_fractile = cu / (co + cu)
    log(label, critical_fractile, suffix=suffix)
    
    return critical_fractile


def newsvendor_revenue_discrete(y, demand_values, demand_probs, p, g, c, label="Revenue function (discrete)", suffix=""):
    """
    Revenue function for newsvendor model with discrete demand distribution
    
    Parameters:
        y: Order quantity
        demand_values: Possible demand values
        demand_probs: Probability of each demand value
        p: Selling price
        g: Salvage value
        c: Procurement cost
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Expected profit/revenue
    """
    log("Order quantity", y, suffix=suffix)
    log("Selling price", p, suffix=suffix)
    log("Salvage value", g, suffix=suffix)
    log("Procurement cost", c, suffix=suffix)
    
    # Procurement cost
    revenue = -c * y
    
    # Revenue from demand <= order quantity (sell demand, salvage remainder)
    for d, prob in zip(demand_values, demand_probs):
        if d <= y:
            rev = (p * d + g * (y - d)) * prob
            revenue += rev
            log(f"{label} (d={d}, revenue contribution)", rev, suffix=suffix)
    
    # Revenue from demand > order quantity (sell everything, no salvage)
    for d, prob in zip(demand_values, demand_probs):
        if d > y:
            rev = p * y * prob
            revenue += rev
            log(f"{label} (d={d}, revenue contribution)", rev, suffix=suffix)
    
    log(label, revenue, suffix=suffix)
    return revenue


def newsvendor_revenue_continuous(y, distribution, dist_params, p, g, c, label="Revenue function (continuous)", suffix=""):
    """
    Revenue function for newsvendor model with continuous demand distribution
    
    Parameters:
        y: Order quantity
        distribution: Distribution type ('normal', 'gamma', etc.)
        dist_params: Dictionary of distribution parameters
        p: Selling price
        g: Salvage value
        c: Procurement cost
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Expected profit/revenue
    """
    log("Order quantity", y, suffix=suffix)
    log("Distribution", distribution, suffix=suffix)
    log("Distribution parameters", dist_params, suffix=suffix)
    log("Selling price", p, suffix=suffix)
    log("Salvage value", g, suffix=suffix)
    log("Procurement cost", c, suffix=suffix)
    
    # Procurement cost
    revenue = -c * y
    
    if distribution == 'normal':
        mu = dist_params['mu']
        sigma = dist_params['sigma']
        
        # Standardize y
        z = (y - mu) / sigma
        
        # Expected lost sales using loss function
        els = sigma * G_z(z, label=f"{label} (loss function)", suffix=suffix)
        
        # Expected sales
        es = mu - els
        
        # Expected leftover inventory
        elo = y - es
        
        # Calculate revenue components
        revenue_sold = p * es
        revenue_salvage = g * elo
        
        revenue += revenue_sold + revenue_salvage
        
        log(f"{label} (expected lost sales)", els, suffix=suffix)
        log(f"{label} (expected sales)", es, suffix=suffix)
        log(f"{label} (expected leftover)", elo, suffix=suffix)
        log(f"{label} (revenue from sales)", revenue_sold, suffix=suffix)
        log(f"{label} (revenue from salvage)", revenue_salvage, suffix=suffix)
    
    elif distribution == 'gamma':
        # For gamma distribution, we need numerical integration
        try:
            from scipy.integrate import quad
            
            alpha = dist_params['alpha']
            beta = dist_params['beta']  # scale parameter
            
            # Expected sales function to integrate
            def expected_sales_integrand(d):
                return d * gamma.pdf(d, a=alpha, scale=beta)
            
            # For d <= y
            es_below_y, _ = quad(expected_sales_integrand, 0, y)
            
            # For d > y
            excess_prob = 1 - gamma.cdf(y, a=alpha, scale=beta)
            es_above_y = y * excess_prob
            
            # Total expected sales
            es = es_below_y + es_above_y
            
            # Mean of the distribution
            mu = alpha * beta
            
            # Expected lost sales
            els = mu - es
            
            # Expected leftover inventory
            elo = y - es
            
            # Calculate revenue components
            revenue_sold = p * es
            revenue_salvage = g * elo
            
            revenue += revenue_sold + revenue_salvage
            
            log(f"{label} (expected lost sales)", els, suffix=suffix)
            log(f"{label} (expected sales)", es, suffix=suffix)
            log(f"{label} (expected leftover)", elo, suffix=suffix)
            log(f"{label} (revenue from sales)", revenue_sold, suffix=suffix)
            log(f"{label} (revenue from salvage)", revenue_salvage, suffix=suffix)
            
        except ImportError:
            log(f"{label} (error)", "SciPy integration not available", suffix=suffix)
            revenue = None
    
    else:
        log(f"{label} (error)", f"Distribution {distribution} not supported", suffix=suffix)
        revenue = None
    
    log(label, revenue, suffix=suffix)
    return revenue


def newsvendor_with_costs(unit_cost, price, holding_cost, stockout_cost, 
                        demand_mean=None, demand_std=None, demand_distr=None, demand_probs=None,
                        label="Newsvendor with costs", suffix=""):
    """
    Newsvendor model based on explicit cost parameters
    
    Parameters:
        unit_cost: Cost per unit
        price: Selling price per unit
        holding_cost: Holding cost per unit of excess inventory
        stockout_cost: Stockout cost per unit of shortage
        demand_mean: Mean demand (for continuous distributions)
        demand_std: Standard deviation of demand (for continuous distributions)
        demand_distr: List of possible demand values (for discrete distribution)
        demand_probs: List of probabilities for demand values (for discrete distribution)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimal order quantity and expected costs
    """
    log("Unit cost", unit_cost, suffix=suffix)
    log("Price", price, suffix=suffix)
    log("Holding cost", holding_cost, suffix=suffix)
    log("Stockout cost", stockout_cost, suffix=suffix)
    
    # Calculate overage and underage costs
    co = unit_cost + holding_cost - 0  # Overage cost (0 is salvage value)
    cu = price - unit_cost + stockout_cost  # Underage cost
    log("Overage cost (co)", co, suffix=suffix)
    log("Underage cost (cu)", cu, suffix=suffix)
    
    # Calculate critical fractile
    critical_fractile = newsvendor_critical_fractile(co, cu, label=f"{label} (critical fractile)", suffix=suffix)
    
    if demand_distr is not None and demand_probs is not None:
        # Discrete demand case
        log("Demand values", demand_distr, suffix=suffix)
        log("Demand probabilities", demand_probs, suffix=suffix)
        
        # Find optimal order quantity
        cum_prob = 0
        optimal_q = None
        for i, q in enumerate(demand_distr):
            cum_prob += demand_probs[i]
            if cum_prob >= critical_fractile:
                optimal_q = q
                break
        
        # If we didn't find a value, use the last one
        if optimal_q is None:
            optimal_q = demand_distr[-1]
        
        # Calculate expected overage and underage
        expected_overage = 0
        expected_underage = 0
        for demand, prob in zip(demand_distr, demand_probs):
            if demand < optimal_q:
                expected_overage += (optimal_q - demand) * prob
            elif demand > optimal_q:
                expected_underage += (demand - optimal_q) * prob
        
        # Calculate expected costs
        expected_overage_cost = co * expected_overage
        expected_underage_cost = cu * expected_underage
        expected_total_cost = expected_overage_cost + expected_underage_cost
        
        log(f"{label} (optimal order quantity)", optimal_q, suffix=suffix)
        log(f"{label} (expected overage)", expected_overage, suffix=suffix)
        log(f"{label} (expected underage)", expected_underage, suffix=suffix)
        log(f"{label} (expected overage cost)", expected_overage_cost, suffix=suffix)
        log(f"{label} (expected underage cost)", expected_underage_cost, suffix=suffix)
        log(f"{label} (expected total cost)", expected_total_cost, suffix=suffix)
        
        return {
            "optimal_order_quantity": optimal_q,
            "critical_fractile": critical_fractile,
            "expected_overage": expected_overage,
            "expected_underage": expected_underage,
            "expected_overage_cost": expected_overage_cost,
            "expected_underage_cost": expected_underage_cost,
            "expected_total_cost": expected_total_cost
        }
    
    elif demand_mean is not None and demand_std is not None:
        # Continuous demand case (assuming normal distribution)
        log("Mean demand", demand_mean, suffix=suffix)
        log("Standard deviation", demand_std, suffix=suffix)
        
        # Calculate optimal order quantity
        z = inverse_cdf(critical_fractile, label=f"{label} (z-value)", suffix=suffix)
        optimal_q = demand_mean + z * demand_std
        
        # Calculate expected overage and underage
        expected_underage = demand_std * G_z(z, label=f"{label} (loss function)", suffix=suffix)
        expected_overage = expected_underage + (optimal_q - demand_mean)
        
        # Calculate expected costs
        expected_overage_cost = co * expected_overage
        expected_underage_cost = cu * expected_underage
        expected_total_cost = expected_overage_cost + expected_underage_cost
        
        log(f"{label} (optimal order quantity)", optimal_q, suffix=suffix)
        log(f"{label} (expected overage)", expected_overage, suffix=suffix)
        log(f"{label} (expected underage)", expected_underage, suffix=suffix)
        log(f"{label} (expected overage cost)", expected_overage_cost, suffix=suffix)
        log(f"{label} (expected underage cost)", expected_underage_cost, suffix=suffix)
        log(f"{label} (expected total cost)", expected_total_cost, suffix=suffix)
        
        return {
            "optimal_order_quantity": optimal_q,
            "critical_fractile": critical_fractile,
            "expected_overage": expected_overage,
            "expected_underage": expected_underage,
            "expected_overage_cost": expected_overage_cost,
            "expected_underage_cost": expected_underage_cost,
            "expected_total_cost": expected_total_cost
        }
    
    else:
        raise ValueError("Either demand_mean and demand_std OR demand_distr and demand_probs must be provided")




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


def newsvendor_with_estimated_params(sample_mean, sample_std, sample_size, p, c, g=0, 
                                    confidence_level=0.95, label="Newsvendor with estimated parameters", suffix=""):
    """
    Newsvendor solution with estimated parameters using t-distribution
    
    This function calculates the optimal order quantity accounting for parameter uncertainty
    when demand parameters are estimated from sample data.
    
    Parameters:
        sample_mean: Sample mean of demand
        sample_std: Sample standard deviation of demand
        sample_size: Number of observations in sample
        p: Selling price
        c: Unit cost
        g: Salvage value (default 0)
        confidence_level: Confidence level (default 0.95)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Optimal order quantity accounting for parameter uncertainty
    """
    from scipy.stats import t
    
    log("Sample mean", sample_mean, suffix=suffix)
    log("Sample std dev", sample_std, suffix=suffix)
    log("Sample size", sample_size, suffix=suffix)
    log("Selling price", p, suffix=suffix)
    log("Unit cost", c, suffix=suffix)
    log("Salvage value", g, suffix=suffix)
    log("Confidence level", confidence_level, suffix=suffix)
    
    # Calculate critical ratio
    CR = newsvendor_critical_ratio(p, c, g, label=f"{label} (critical ratio)", suffix=suffix)
    
    # Get t-value for given confidence level and degrees of freedom
    degrees_of_freedom = sample_size - 1
    t_value = t.ppf(CR, df=degrees_of_freedom)
    log(f"{label} (t-value)", t_value, suffix=suffix)
    
    # Calculate correction factor for parameter uncertainty
    correction_factor = math.sqrt(1 + 1/sample_size)
    log(f"{label} (correction factor)", correction_factor, suffix=suffix)
    
    # Calculate optimal order quantity with correction
    optimal_q = sample_mean + t_value * sample_std * correction_factor
    log(label, optimal_q, unit="units", suffix=suffix)
    
    # Compare with standard normal solution
    standard_q = newsvendor_normal(sample_mean, sample_std, CR, label=f"{label} (without correction)", suffix=suffix)
    
    # Calculate difference
    difference = optimal_q - standard_q
    percent_difference = (difference / standard_q) * 100 if standard_q != 0 else 0
    
    log(f"{label} (absolute difference)", difference, unit="units", suffix=suffix)
    log(f"{label} (percentage difference)", percent_difference, unit="%", suffix=suffix)
    
    return optimal_q


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
            "Fill Rate (beta)": fill_rate,
        }

    elif distr == "poisson":
        lam = params["lambda"]
        Q = 0
        while poisson.cdf(Q, lam) < CR:
            Q += 1

        # Expected lost sales
        max_k = Q + 1000  # Approximate upper bound
        expected_lost_sales = sum(
            (k - Q) * poisson.pmf(k, lam) for k in range(Q + 1, max_k + 1)
        )
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
            "Fill Rate (beta)": fill_rate,
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
            "Fill Rate (beta)": fill_rate,
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
# Statistical Distribution Utility Functions
#####################################################


def phi_z(z, label="Normal PDF", suffix=""):
    """
    Standard normal probability density function φ(z)
    
    Parameters:
        z: Standard normal quantile value
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Value of φ(z)
    """
    result = norm.pdf(z)
    log(label, result, suffix=suffix)
    return result


def Phi_z(z, label="Normal CDF", suffix=""):
    """
    Standard normal cumulative distribution function Φ(z)
    
    Parameters:
        z: Standard normal quantile value
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Value of Φ(z)
    """
    result = norm.cdf(z)
    log(label, result, suffix=suffix)
    return result


def G_z(z, label="Loss function G(z)", suffix=""):
    """
    Standard normal loss function G(z) = φ(z) - z·(1-Φ(z))
    
    Parameters:
        z: Standard normal quantile value
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Value of the loss function G(z)
    """
    result = phi_z(z, label="") - z * (1 - Phi_z(z, label=""))
    log(label, result, suffix=suffix)
    return result


def inverse_G(G_target, min=-1000, max=1000, label="Inverse G(z)", suffix=""):
    """
    Finds z such that G(z) = G_target
    
    Parameters:
        G_target: Target value of the loss function
        min: Lower bound for search range
        max: Upper bound for search range
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        z value such that G(z) = G_target
    """
    result = brentq(lambda z: G_z(z, label="", suffix="") - G_target, min, max)
    log(label, result, suffix=suffix)
    return result


def inverse_cdf(p, label="Inverse normal CDF", suffix=""):
    """
    Inverse of the standard normal CDF (probit function).
    
    Parameters:
        p: Probability value in (0, 1), like Phi or CR
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
    
    Returns:
        z such that Φ(z) = p
    """
    result = norm.ppf(p)
    log(label, result, suffix=suffix)
    return result


def inverse_pdf(y, min=0, max=1000, label="Inverse normal PDF", suffix=""):
    """
    Inverse of the standard normal PDF.
    Finds z ≥ 0 such that φ(z) = y.
    
    Parameters:
        y: PDF value in (0, 1/√(2π)]
        min: Lower bound for search range
        max: Upper bound for search range
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
    
    Returns:
        z ≥ 0 such that φ(z) = y
    """
    result = brentq(lambda z: norm.pdf(z) - y, min, max)
    log(label, result, suffix=suffix)
    return result


def standardize(x, mu, sigma, label="Standardized value", suffix=""):
    """
    Standardize a value to get z-score: z = (x - μ)/σ
    
    Parameters:
        x: Value to standardize
        mu: Mean
        sigma: Standard deviation
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Standardized value (z-score)
    """
    result = (x - mu) / sigma
    log(label, result, suffix=suffix)
    return result


def unstandardize(z, mu, sigma, label="Original value", suffix=""):
    """
    Convert z-score back to original scale: x = μ + z·σ
    
    Parameters:
        z: Z-score
        mu: Mean
        sigma: Standard deviation
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Value in original scale
    """
    result = mu + z * sigma
    log(label, result, suffix=suffix)
    return result


def safety_factor_for_service_level(service_level, label="Safety factor", suffix=""):
    """
    Calculate safety factor (z) for a given service level
    
    Parameters:
        service_level: Service level (probability of no stockout)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Safety factor z
    """
    return inverse_cdf(service_level, label=label, suffix=suffix)


def service_level_from_safety_factor(z, label="Service level", suffix=""):
    """
    Calculate service level for a given safety factor
    
    Parameters:
        z: Safety factor
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Service level (probability of no stockout)
    """
    result = Phi_z(z, label="", suffix="")
    log(label, result, suffix=suffix)
    return result


def fill_rate_from_safety_factor(z, label="Fill rate", suffix=""):
    """
    Calculate fill rate (fraction of demand that can be satisfied immediately from stock)
    given a safety factor z, assuming normally distributed demand during lead time
    
    Parameters:
        z: Safety factor
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Fill rate (fraction of demand satisfied)
    """
    result = 1 - G_z(z, label="", suffix="") / z if z > 0 else 0
    log(label, result, suffix=suffix)
    return result


def expected_shortage(mu, sigma, z, label="Expected shortage", suffix=""):
    """
    Calculate expected shortage for a given safety factor z and normal distribution
    
    Parameters:
        mu: Mean demand
        sigma: Standard deviation of demand
        z: Safety factor
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Expected shortage quantity
    """
    g_z = G_z(z, label="Loss function", suffix=suffix)
    result = sigma * g_z
    log(label, result, suffix=suffix)
    return result


def mad_to_stdev(mad, label="Standard deviation", suffix=""):
    """
    Convert Mean Absolute Deviation (MAD) to standard deviation
    for normal distribution: sigma ≈ 1.25 × MAD
    
    Parameters:
        mad: Mean Absolute Deviation
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Approximate standard deviation
    """
    result = 1.25 * mad
    log(label, result, suffix=suffix)
    return result


def stdev_to_mad(sigma, label="MAD", suffix=""):
    """
    Convert standard deviation to Mean Absolute Deviation (MAD)
    for normal distribution: MAD ≈ 0.8 × sigma
    
    Parameters:
        sigma: Standard deviation
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Approximate MAD
    """
    result = 0.8 * sigma
    log(label, result, suffix=suffix)
    return result


def normal_loss_function(x, mu, sigma, label="Normal loss function", suffix=""):
    """
    Normal loss function for non-standard normal distribution
    E[max(X-x, 0)] where X ~ N(mu, sigma²)
    
    Parameters:
        x: Target value
        mu: Mean of distribution
        sigma: Standard deviation
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Expected value of max(X-x, 0)
    """
    z = (x - mu) / sigma
    g_z = G_z(-z, label="", suffix="")  # Note the negative z
    result = sigma * g_z
    log(label, result, suffix=suffix)
    return result


def partial_expectation(x, mu, sigma, label="Partial expectation", suffix=""):
    """
    Partial expectation for normal random variable X
    E[X | X > x] × P(X > x)
    
    Parameters:
        x: Threshold value
        mu: Mean
        sigma: Standard deviation
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Partial expectation
    """
    z = (x - mu) / sigma
    result = mu * (1 - Phi_z(z, label="", suffix="")) + sigma * phi_z(z, label="", suffix="")
    log(label, result, suffix=suffix)
    return result
#####################################################
# Safety Stock and Reorder Points
#####################################################


def safety_stock(z, std_dev, lead_time, review_period=0, label="Safety stock", suffix=""):
    """
    Standard safety stock calculation

    Parameters:
        z: Safety factor
        std_dev: Standard deviation of demand
        lead_time: Lead time
        review_period: Review period (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Safety stock
    """
    ss = z * std_dev * math.sqrt(lead_time + review_period)
    log(label, ss, unit="units", suffix=suffix)
    return ss


def reorder_point(D, lead_time_months, label="Reorder point", suffix=""):
    """
    Calculate reorder point (ROP) for constant demand

    Parameters:
        D: Annual demand
        lead_time_months: Lead time in months
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

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


def service_level_safety_stock(mean_demand, std_demand, lead_time, service_level, 
                              label="Service level safety stock", suffix=""):
    """
    Calculate safety stock for a given service level

    Parameters:
        mean_demand: Mean demand
        std_demand: Standard deviation of demand
        lead_time: Lead time
        service_level: Desired service level (probability)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Tuple (safety_stock, reorder_point)
    """
    z = inverse_cdf(service_level, label="Z-value", suffix=suffix)
    ss = z * std_demand * math.sqrt(lead_time)
    reorder_pt = mean_demand * lead_time + ss

    log("Mean demand", mean_demand, suffix=suffix)
    log("Std demand", std_demand, suffix=suffix)
    log("Lead time", lead_time, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    log(f"{label}", ss, unit="units", suffix=suffix)
    log("Reorder point", reorder_pt, unit="units", suffix=suffix)

    return ss, reorder_pt


def variable_lead_time_safety_stock(z, mean_demand, std_demand, mean_lead_time, std_lead_time,
                                   review_period=0, label="Variable lead time safety stock", suffix=""):
    """
    Safety stock calculation for variable lead time

    Parameters:
        z: Safety factor
        mean_demand: Mean demand
        std_demand: Standard deviation of demand
        mean_lead_time: Mean lead time
        std_lead_time: Standard deviation of lead time
        review_period: Review period (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Dictionary with safety stock and reorder point
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
    
    # Calculate reorder point
    reorder_point = mean_demand * (mean_lead_time + review_period) + ss
    log("Reorder point", reorder_point, unit="units", suffix=suffix)

    return {
        "safety_stock": ss,
        "reorder_point": reorder_point,
        "std_dev_effective": math.sqrt(term1 + term2)
    }


def reorder_point_variable_lead_time(mean_demand, std_demand, mean_lead_time, 
                                    var_lead_time, z, review_period=0, 
                                    label="Reorder point (variable LT)", suffix=""):
    """
    Calculate reorder point with variable lead time (Formula from Ex3 Assignment Question 4c)
    
    Parameters:
        mean_demand: Mean demand per period
        std_demand: Standard deviation of demand
        mean_lead_time: Mean lead time
        var_lead_time: Variance of lead time
        z: Safety factor
        review_period: Review period (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Reorder point
    """
    log("Mean demand", mean_demand, suffix=suffix)
    log("Standard deviation", std_demand, suffix=suffix)
    log("Mean lead time", mean_lead_time, suffix=suffix)
    log("Lead time variance", var_lead_time, suffix=suffix)
    log("Z-value", z, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    
    # Calculate safety stock considering both demand and lead time variability
    safety_stock = z * math.sqrt(
        (mean_lead_time + review_period) * std_demand**2 + 
        mean_demand**2 * var_lead_time
    )
    log("Safety stock", safety_stock, unit="units", suffix=suffix)
    
    # Calculate reorder point
    reorder_point = mean_demand * (mean_lead_time + review_period) + safety_stock
    log(label, reorder_point, unit="units", suffix=suffix)
    
    return reorder_point


#####################################################
# Service Level and Fill Rate Calculations
#####################################################


def service_level_to_fill_rate(service_level, cv, label="Fill rate", suffix=""):
    """
    Convert service level (probability of no stockout) to fill rate
    (fraction of demand satisfied immediately)
    
    Parameters:
        service_level: Service level (alpha)
        cv: Coefficient of variation of demand
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Approximate fill rate (beta)
    """
    z = safety_factor_for_service_level(service_level, label="", suffix="")
    g_z = G_z(z, label="", suffix="")
    fill_rate = 1 - (g_z * cv) / service_level
    log(label, fill_rate, suffix=suffix)
    return fill_rate


def fill_rate_to_service_level(fill_rate, cv, label="Service level", suffix=""):
    """
    Convert fill rate to service level (approximate conversion)
    
    Parameters:
        fill_rate: Fill rate (beta)
        cv: Coefficient of variation of demand
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Approximate service level (alpha)
    """
    # This is an approximation using iterative approach
    service_level = fill_rate  # Initial guess
    
    for _ in range(5):
        z = safety_factor_for_service_level(service_level, label="", suffix="")
        g_z = G_z(z, label="", suffix="")
        service_level = (1 - fill_rate) / (cv * g_z) if g_z > 0 else fill_rate
    
    log(label, service_level, suffix=suffix)
    return service_level


def calculate_z_for_shortage_cost(D, B, h, Q, sigma_L, label="Z for shortage cost", suffix=""):
    """
    Calculate z value when considering shortage cost (Formula from Ex3 Assignment Question 1)
    
    Parameters:
        D: Annual demand
        B: Shortage penalty cost per occasion
        h: Holding cost per unit per year
        Q: Order quantity
        sigma_L: Standard deviation of demand during lead time
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        z value for safety factor
    """
    log("Annual demand", D, suffix=suffix)
    log("Shortage penalty cost", B, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Order quantity", Q, suffix=suffix)
    log("Lead time standard deviation", sigma_L, suffix=suffix)
    
    # Formula from Ex3 Assignment Question 1
    z = math.sqrt(
        2 * math.log(
            D * B / (math.sqrt(2 * math.pi) * Q * h * sigma_L)
        )
    )
    
    log(label, z, suffix=suffix)
    return z


def calculate_z_for_service_cost_ratio(h, Q, p, D, label="Z for service level", suffix=""):
    """
    Calculate z value based on cost ratio (Formula from Ex3 Assignment Question 2)
    
    Parameters:
        h: Holding cost per unit per time period
        Q: Order quantity
        p: Shortage penalty cost per unit
        D: Demand rate
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        z value for safety factor
    """
    log("Holding cost", h, suffix=suffix)
    log("Order quantity", Q, suffix=suffix)
    log("Shortage penalty cost", p, suffix=suffix)
    log("Demand rate", D, suffix=suffix)
    
    # Formula from Ex3 Assignment Question 2
    service_level = 1 - ((h * Q) / (p * D))
    log(f"{label} (implied service level)", service_level, suffix=suffix)
    
    z = inverse_cdf(service_level, label=label, suffix=suffix)
    return z


def z_for_fill_rate(mean_demand, std_demand, fill_rate, lead_time, review_period=0, label="Z for fill rate", suffix=""):
    """
    Calculate z value for a desired fill rate (Formula from Ex3 Assignment Question 4)
    
    Parameters:
        mean_demand: Mean demand per period
        std_demand: Standard deviation of demand
        fill_rate: Target fill rate (beta)
        lead_time: Lead time
        review_period: Review period (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        z value for safety factor
    """
    log("Mean demand", mean_demand, suffix=suffix)
    log("Standard deviation", std_demand, suffix=suffix)
    log("Fill rate", fill_rate, suffix=suffix)
    log("Lead time", lead_time, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    
    # Formula from Ex3 Assignment Question 4
    g_z_target = mean_demand * (1 - fill_rate) / (std_demand * math.sqrt(lead_time + review_period))
    log(f"{label} (target G(z) value)", g_z_target, suffix=suffix)
    
    # Find z value that gives this G(z)
    z = inverse_G(g_z_target, label=label, suffix=suffix)
    return z


def calculate_fill_rate(z, Q, sigma_L, label="Fill rate", suffix=""):
    """
    Calculate fill rate (beta) for a given z-value and order quantity
    
    Parameters:
        z: Service level z-value
        Q: Order quantity
        sigma_L: Standard deviation of demand during lead time
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Fill rate (beta)
    """
    log("z-value", z, suffix=suffix)
    log("Order quantity", Q, unit="units", suffix=suffix)
    log("Std dev lead time demand", sigma_L, suffix=suffix)
    
    g_z = G_z(z, label=f"{label} (loss function)", suffix=suffix)
    fill_rate = 1 - (sigma_L * g_z / Q)
    log(label, fill_rate, suffix=suffix)
    
    return fill_rate


def calculate_adjusted_fill_rate(mean_demand, std_demand, order_up_to_level, lead_time, review_period=0, 
                               label="Adjusted fill rate (γ)", suffix=""):
    """
    Calculate adjusted fill rate (γ-Service-level) for (R,S) policy
    
    Parameters:
        mean_demand: Mean demand per period
        std_demand: Standard deviation of demand per period
        order_up_to_level: Order-up-to level (S)
        lead_time: Lead time
        review_period: Review period (R)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Adjusted fill rate (γ)
    """
    log("Mean demand", mean_demand, suffix=suffix)
    log("Std deviation", std_demand, suffix=suffix)
    log("Order-up-to level (S)", order_up_to_level, suffix=suffix)
    log("Lead time (L)", lead_time, suffix=suffix)
    log("Review period (R)", review_period, suffix=suffix)
    
    # Calculate parameters
    mean_demand_lead_review = mean_demand * (lead_time + review_period)
    std_demand_lead_review = std_demand * math.sqrt(lead_time + review_period)
    
    log("Mean demand during L+R", mean_demand_lead_review, suffix=suffix)
    log("Std dev during L+R", std_demand_lead_review, suffix=suffix)
    
    # Calculate safety factor
    z = (order_up_to_level - mean_demand_lead_review) / std_demand_lead_review
    log("Safety factor (z)", z, suffix=suffix)
    
    # Calculate expected shortage using loss function G(z)
    g_z = G_z(z, label=f"{label} (loss function)", suffix=suffix)
    expected_shortage = std_demand_lead_review * g_z
    log("Expected shortage", expected_shortage, suffix=suffix)
    
    # Calculate adjusted fill rate
    adjusted_fill_rate = 1 - (expected_shortage / mean_demand)
    log(label, adjusted_fill_rate, suffix=suffix)
    
    return adjusted_fill_rate


def base_stock_level_stochastic_lead_time(mean_demand, std_demand, mean_lead_time, std_lead_time, 
                                        review_period, service_level, label="Base stock level", suffix=""):
    """
    Calculate base stock level (S) for (R,S) policy with stochastic lead time
    
    Parameters:
        mean_demand: Mean demand per period
        std_demand: Standard deviation of demand per period
        mean_lead_time: Mean lead time
        std_lead_time: Standard deviation of lead time
        review_period: Review period (R)
        service_level: Target service level (non-stockout probability)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Base stock level (S)
    """
    log("Mean demand", mean_demand, suffix=suffix)
    log("Std demand", std_demand, suffix=suffix)
    log("Mean lead time", mean_lead_time, suffix=suffix)
    log("Std lead time", std_lead_time, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    
    # Calculate safety factor for service level
    z = inverse_cdf(service_level, label=f"{label} (safety factor)", suffix=suffix)
    
    # Calculate mean demand during lead time + review period
    mean_demand_total = mean_demand * (mean_lead_time + review_period)
    log(f"{label} (mean demand during L+R)", mean_demand_total, suffix=suffix)
    
    # Calculate variance considering both demand and lead time uncertainty
    # variance = (L+R) * σ²ᴅ + μ²ᴅ * σ²ʟ
    variance_demand_term = (mean_lead_time + review_period) * (std_demand ** 2)
    variance_lead_term = (mean_demand ** 2) * (std_lead_time ** 2)
    total_variance = variance_demand_term + variance_lead_term
    
    log(f"{label} (variance from demand)", variance_demand_term, suffix=suffix)
    log(f"{label} (variance from lead time)", variance_lead_term, suffix=suffix)
    log(f"{label} (total variance)", total_variance, suffix=suffix)
    
    std_total = math.sqrt(total_variance)
    log(f"{label} (total std dev)", std_total, suffix=suffix)
    
    # Calculate base stock level
    base_stock = mean_demand_total + z * std_total
    log(label, base_stock, unit="units", suffix=suffix)
    
    return base_stock


def order_up_to_level(forecast, safety_stock, label="Order-up-to level", suffix=""):
    """
    Order-up-to level (S)

    Parameters:
        forecast: Demand forecast
        safety_stock: Safety stock
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Order-up-to level
    """
    result = np.array(forecast) + safety_stock

    log("Forecast", forecast, suffix=suffix)
    log("Safety stock", safety_stock, unit="units", suffix=suffix)
    log(label, result, unit="units", suffix=suffix)

    return result


def order_up_to_level_fill_rate(mean_demand, std_demand, z, lead_time, review_period=0, label="Order-up-to level (fill rate)", suffix=""):
    """
    Calculate order-up-to level (S) for a given z-value with fill rate
    
    Parameters:
        mean_demand: Mean demand per period
        std_demand: Standard deviation of demand
        z: Safety factor
        lead_time: Lead time
        review_period: Review period (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Order-up-to level (S)
    """
    log("Mean demand", mean_demand, suffix=suffix)
    log("Standard deviation", std_demand, suffix=suffix)
    log("Z-value", z, suffix=suffix)
    log("Lead time", lead_time, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    
    S = mean_demand * (lead_time + review_period) + z * std_demand * math.sqrt(lead_time + review_period)
    log(label, S, unit="units", suffix=suffix)
    return S


def order_up_to_policy(demand, forecast, safety_stock, initial_inventory=0, 
                      label="Order-up-to policy", suffix=""):
    """
    Calculate order quantities using order-up-to policy
    
    Parameters:
        demand: Array of actual demand values
        forecast: Array of demand forecasts
        safety_stock: Safety stock value or array
        initial_inventory: Initial inventory level (default 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with order quantities and inventory levels
    """
    log("Historical demand", demand, suffix=suffix)
    log("Forecast values", forecast, suffix=suffix)
    log("Safety stock", safety_stock, suffix=suffix)
    log("Initial inventory", initial_inventory, suffix=suffix)
    
    n = len(demand)
    
    # Convert safety_stock to array if it's a scalar
    if np.isscalar(safety_stock):
        safety_stock = np.full(n+1, safety_stock)
    
    # Calculate order-up-to level for each period
    S = forecast + safety_stock
    log(f"{label} (order-up-to levels)", S, suffix=suffix)
    
    # Initialize inventory and orders
    inventory = np.zeros(n+1)
    inventory[0] = initial_inventory
    orders = np.zeros(n)
    
    # Calculate orders and inventory
    for t in range(n):
        # Calculate order quantity
        orders[t] = max(0, S[t+1] - inventory[t])
        
        # Update inventory
        inventory[t+1] = inventory[t] + orders[t] - demand[t]
        
        log(f"{label} (period {t} order)", orders[t], suffix=suffix)
        log(f"{label} (period {t+1} inventory)", inventory[t+1], suffix=suffix)
    
    return {
        "order_up_to_levels": S,
        "orders": orders,
        "inventory": inventory,
        "average_order": np.mean(orders),
        "order_variability": np.std(orders),
        "service_level": np.mean(inventory[1:] >= 0)
    }


#####################################################
# Inventory Models with Shortage Costs
#####################################################

def eoq_with_shortage_cost(D, A, h, p, label="EOQ with shortage", suffix=""):
    """
    EOQ model with stockout/shortage costs
    
    Parameters:
        D: Annual demand
        A: Fixed ordering cost
        h: Holding cost per unit per year
        p: Stockout/penalty cost per unit per year
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with order quantity, max inventory, max shortage, and cost
    """
    log("Annual demand", D, suffix=suffix)
    log("Fixed ordering cost", A, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Stockout/penalty cost", p, suffix=suffix)
    
    # Calculate optimal order quantity
    Q_star = math.sqrt(2 * D * A * (h + p) / (h * p))
    log(f"{label} (order quantity)", Q_star, unit="units", suffix=suffix)
    
    # Calculate maximum inventory level
    S_star = (h * Q_star) / (h + p)
    log(f"{label} (max inventory)", S_star, unit="units", suffix=suffix)
    
    # Calculate maximum shortage/backorder level
    B_star = Q_star - S_star
    log(f"{label} (max shortage)", B_star, unit="units", suffix=suffix)
    
    # Calculate total relevant cost
    TRC = math.sqrt(2 * D * A * h * p / (h + p))
    log(f"{label} (total relevant cost)", TRC, suffix=suffix)
    
    return {
        "order_quantity": Q_star,
        "max_inventory": S_star,
        "max_shortage": B_star,
        "total_cost": TRC
    }


def joint_optimization_s_Q(D, A, h, p, mu_L, sigma_L, label="Joint optimization (s,Q)", suffix=""):
    """
    Joint optimization of reorder point (s) and order quantity (Q) with stockout costs
    
    Parameters:
        D: Annual demand
        A: Fixed ordering cost
        h: Holding cost per unit per year
        p: Stockout/penalty cost per unit
        mu_L: Mean demand during lead time
        sigma_L: Standard deviation of demand during lead time
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with order quantity, reorder point, and cost
    """
    log("Annual demand", D, suffix=suffix)
    log("Fixed ordering cost", A, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Stockout/penalty cost", p, suffix=suffix)
    log("Mean lead time demand", mu_L, suffix=suffix)
    log("Std dev lead time demand", sigma_L, suffix=suffix)
    
    # Start with EOQ as initial value for Q
    Q = eoq(D, A, h, label=f"{label} (initial Q)", suffix=suffix)
    
    iterations = 5
    for i in range(iterations):
        log(f"{label} (iteration {i+1})", "", suffix=suffix)
        
        # Step 1: Calculate z-value based on current Q
        z = math.sqrt(2 * math.log(D * p / (math.sqrt(2 * math.pi) * Q * h * sigma_L)))
        log(f"{label} (z-value)", z, suffix=suffix)
        
        # Step 2: Calculate safety stock and reorder point
        safety_stock = z * sigma_L
        s = mu_L + safety_stock
        log(f"{label} (safety stock)", safety_stock, unit="units", suffix=suffix)
        log(f"{label} (reorder point)", s, unit="units", suffix=suffix)
        
        # Step 3: Calculate expected shortage
        g_z = G_z(z, label=f"{label} (loss function)", suffix=suffix)
        expected_shortage = sigma_L * g_z
        log(f"{label} (expected shortage)", expected_shortage, unit="units", suffix=suffix)
        
        # Step 4: Update Q
        Q_new = math.sqrt(2 * D * (A + p * expected_shortage) / h)
        log(f"{label} (updated Q)", Q_new, unit="units", suffix=suffix)
        
        # Step 5: Calculate costs
        holding_cost = (Q_new / 2 + safety_stock) * h
        shortage_cost = (D / Q_new) * p * expected_shortage
        ordering_cost = (D / Q_new) * A
        total_cost = holding_cost + shortage_cost + ordering_cost
        
        log(f"{label} (holding cost)", holding_cost, suffix=suffix)
        log(f"{label} (shortage cost)", shortage_cost, suffix=suffix)
        log(f"{label} (ordering cost)", ordering_cost, suffix=suffix)
        log(f"{label} (total cost)", total_cost, suffix=suffix)
        
        # Update Q for next iteration
        Q = Q_new
    
    return {
        "order_quantity": Q,
        "reorder_point": s,
        "safety_stock": safety_stock,
        "z_value": z,
        "expected_shortage": expected_shortage,
        "total_cost": total_cost
    }


def discrete_lead_time_demand(demands, probabilities, s, h, p, label="Discrete lead time demand", suffix=""):
    """
    Calculate costs for discrete lead time demand distribution
    
    Parameters:
        demands: List of possible demand values during lead time
        probabilities: List of corresponding probabilities
        s: Reorder point
        h: Holding cost per unit
        p: Shortage/stockout cost per unit
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with expected costs
    """
    log("Possible demand values", demands, suffix=suffix)
    log("Demand probabilities", probabilities, suffix=suffix)
    log("Reorder point", s, unit="units", suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Stockout cost", p, suffix=suffix)
    
    if len(demands) != len(probabilities):
        raise ValueError("Demands and probabilities lists must have the same length")
    
    expected_overage = 0
    expected_underage = 0
    expected_overage_cost = 0
    expected_underage_cost = 0
    
    for demand, prob in zip(demands, probabilities):
        if demand > s:
            # Underage case
            underage = demand - s
            expected_underage += underage * prob
            expected_underage_cost += underage * prob * p
        elif demand < s:
            # Overage case
            overage = s - demand
            expected_overage += overage * prob
            expected_overage_cost += overage * prob * h
    
    log(f"{label} (expected overage)", expected_overage, unit="units", suffix=suffix)
    log(f"{label} (expected underage)", expected_underage, unit="units", suffix=suffix)
    log(f"{label} (expected overage cost)", expected_overage_cost, suffix=suffix)
    log(f"{label} (expected underage cost)", expected_underage_cost, suffix=suffix)
    
    return {
        "expected_overage": expected_overage,
        "expected_underage": expected_underage,
        "expected_overage_cost": expected_overage_cost,
        "expected_underage_cost": expected_underage_cost,
        "total_expected_cost": expected_overage_cost + expected_underage_cost
    }


def joint_optimization_s_Q_discrete(D, A, h, p, demands, probabilities, label="Joint (s,Q) discrete", suffix=""):
    """
    Joint optimization of (s,Q) with discrete lead time demand
    
    Parameters:
        D: Annual demand
        A: Fixed ordering cost
        h: Holding cost per unit
        p: Stockout cost per unit
        demands: List of possible demand values during lead time
        probabilities: List of corresponding probabilities
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimized s, Q, and costs
    """
    log("Annual demand", D, suffix=suffix)
    log("Fixed ordering cost", A, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Stockout cost", p, suffix=suffix)
    
    best_cost = float('inf')
    best_s = None
    best_Q = None
    best_results = None
    
    # Try different reorder points
    possible_s = sorted(list(set(demands)))
    possible_s.extend([s + 50 for s in possible_s[-2:]])  # Add some higher values
    
    for s in possible_s:
        # Calculate underage costs for this s
        results = discrete_lead_time_demand(demands, probabilities, s, h, p, label=f"{label} (s={s})", suffix=suffix)
        
        # Calculate optimal Q given expected underage
        Q = math.sqrt(2 * D * (A + results["expected_underage_cost"]) / h)
        
        # Calculate total cost with this (s,Q) pair
        total_cost = (D / Q) * A + h * (Q / 2 + results["expected_overage"]) + (D / Q) * results["expected_underage_cost"]
        
        log(f"{label} (s={s}, Q={Q:.2f})", total_cost, suffix=suffix)
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_s = s
            best_Q = Q
            best_results = results
    
    log(f"{label} (optimal s)", best_s, unit="units", suffix=suffix)
    log(f"{label} (optimal Q)", best_Q, unit="units", suffix=suffix)
    log(f"{label} (optimal cost)", best_cost, suffix=suffix)
    
    return {
        "reorder_point": best_s,
        "order_quantity": best_Q,
        "expected_overage": best_results["expected_overage"],
        "expected_underage": best_results["expected_underage"],
        "total_cost": best_cost
    }


def optimal_Q_with_expected_shortage(D, A, h, p, demands, probabilities, s, label="Q with expected shortage", suffix=""):
    """
    Calculate optimal order quantity with expected shortage costs
    for a given reorder point and discrete demand distribution.
    
    This implements the exact formula from Assignment 3, Question 3:
    Q = sqrt(2 * annual_demand * (order_cost + expected_underage_cost) / holding_cost)
    
    Parameters:
        D: Annual demand
        A: Fixed ordering cost
        h: Holding cost per unit
        p: Stockout/penalty cost per unit
        demands: List of possible demand values during lead time
        probabilities: List of corresponding probabilities
        s: Reorder point
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with order quantity and costs
    """
    log("Annual demand", D, suffix=suffix)
    log("Fixed ordering cost", A, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Stockout/penalty cost", p, suffix=suffix)
    log("Reorder point", s, suffix=suffix)
    log("Demand values", demands, suffix=suffix)
    log("Demand probabilities", probabilities, suffix=suffix)
    
    # Calculate expected overage and underage
    expected_overage = 0
    expected_underage = 0
    expected_overage_cost = 0
    expected_underage_cost = 0
    
    for demand, probability in zip(demands, probabilities):
        if demand > s:
            # Underage case (demand exceeds reorder point)
            underage = demand - s
            expected_underage += underage * probability
            expected_underage_cost += underage * probability * p
        elif demand < s:
            # Overage case (reorder point exceeds demand)
            overage = s - demand
            expected_overage += overage * probability
            expected_overage_cost += overage * probability * h
    
    log(f"{label} (expected overage)", expected_overage, unit="units", suffix=suffix)
    log(f"{label} (expected underage)", expected_underage, unit="units", suffix=suffix)
    log(f"{label} (expected overage cost)", expected_overage_cost, suffix=suffix)
    log(f"{label} (expected underage cost)", expected_underage_cost, suffix=suffix)
    
    # Calculate optimal order quantity using the formula from Assignment 3
    Q = math.sqrt(2 * D * (A + expected_underage_cost) / h)
    log(f"{label} (optimal order quantity)", Q, unit="units", suffix=suffix)
    
    # Calculate total expected cost
    expected_total_cost = (
        (D / Q) * A +                        # Ordering cost
        h * (Q / 2 + expected_overage) +     # Holding cost
        (D / Q) * p * expected_underage      # Stockout cost
    )
    log(f"{label} (total expected cost)", expected_total_cost, suffix=suffix)
    
    return {
        "order_quantity": Q,
        "expected_overage": expected_overage,
        "expected_underage": expected_underage,
        "expected_overage_cost": expected_overage_cost,
        "expected_underage_cost": expected_underage_cost,
        "total_cost": expected_total_cost
    }


#####################################################
# Alternative Distribution Models
#####################################################


def gamma_service_level(mean, std, lead_time, review_period, service_level, 
                       label="Gamma service level", suffix=""):
    """
    Calculate order-up-to level using gamma distribution (Formula from Ex3 Assignment Question 5)
    
    Parameters:
        mean: Mean demand per period
        std: Standard deviation of demand
        lead_time: Lead time
        review_period: Review period
        service_level: Desired service level
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Order-up-to level using gamma distribution
    """
    log("Mean demand", mean, suffix=suffix)
    log("Standard deviation", std, suffix=suffix)
    log("Lead time", lead_time, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    
    # Adjust parameters for lead time and review period
    mu_new = mean * (lead_time + review_period)
    sigma_new = std * math.sqrt(lead_time + review_period)
    log("Adjusted mean", mu_new, suffix=suffix)
    log("Adjusted std dev", sigma_new, suffix=suffix)
    
    # Calculate gamma distribution parameters
    alpha = mu_new**2 / sigma_new**2  # Shape parameter
    beta = sigma_new**2 / mu_new      # Scale parameter
    log("Gamma shape parameter (alpha)", alpha, suffix=suffix)
    log("Gamma scale parameter (beta)", beta, suffix=suffix)
    
    # Calculate order-up-to level
    order_up_to = gamma.ppf(service_level, alpha, scale=beta)
    log(label, order_up_to, unit="units", suffix=suffix)
    
    return order_up_to


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


def exp_smoothing(
    alpha, demand, initial_forecast, label="Exponential smoothing forecast", suffix=""
):
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


def exponential_smoothing_error(
    alpha, demand, initial_level=0, label="Exponential smoothing error", suffix=""
):
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


def holts_trend_forecasting(demand, initial_level, initial_trend, 
                          alpha=0.2, beta=0.1, periods_ahead=1,
                          label="Holt's trend model", suffix=""):
    """
    Holt's Trend Model for forecasting with trend component
    
    Parameters:
        demand: Array of historical demand
        initial_level: Initial level value
        initial_trend: Initial trend value
        alpha: Level smoothing parameter (default 0.2)
        beta: Trend smoothing parameter (default 0.1)
        periods_ahead: Number of periods to forecast ahead (default 1)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with level, trend, fitted values and forecasts
    """
    log("Historical demand", demand, suffix=suffix)
    log("Initial level", initial_level, suffix=suffix)
    log("Initial trend", initial_trend, suffix=suffix)
    log("Alpha (level)", alpha, suffix=suffix)
    log("Beta (trend)", beta, suffix=suffix)
    
    n = len(demand)
    
    # Initialize arrays
    level = np.zeros(n+1)
    trend = np.zeros(n+1)
    fitted = np.zeros(n+1)
    
    # Set initial values
    level[0] = initial_level
    trend[0] = initial_trend
    fitted[0] = level[0]
    
    # Calculate level and trend
    for t in range(1, n+1):
        if t <= n:
            # Update level
            level[t] = alpha * demand[t-1] + (1-alpha) * (level[t-1] + trend[t-1])
            # Update trend
            trend[t] = beta * (level[t] - level[t-1]) + (1-beta) * trend[t-1]
            # Fitted value (one-step ahead forecast)
            fitted[t] = level[t-1] + trend[t-1]
            
            log(f"{label} (period {t} level)", level[t], suffix=suffix)
            log(f"{label} (period {t} trend)", trend[t], suffix=suffix)
            log(f"{label} (period {t} fitted)", fitted[t], suffix=suffix)
    
    # Generate forecasts for future periods
    forecasts = np.zeros(periods_ahead)
    for h in range(periods_ahead):
        forecasts[h] = level[n] + (h+1) * trend[n]
        log(f"{label} (forecast for period {n+h+1})", forecasts[h], suffix=suffix)
        
    return {
        "level": level,
        "trend": trend,
        "fitted": fitted[1:],  # Remove initial value
        "forecasts": forecasts
    }


def forecast_error_metrics(actual, forecast, alpha=0.1, label="Forecast error metrics", suffix=""):
    """
    Calculate various error metrics for a forecast
    
    Parameters:
        actual: Array of actual demand values
        forecast: Array of forecast values
        alpha: Smoothing parameter for MAD and ERR (default 0.1)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with error metrics
    """
    # Remove NaN values from comparison
    valid_indices = ~np.isnan(forecast)
    valid_actual = np.array(actual)[valid_indices]
    valid_forecast = np.array(forecast)[valid_indices]
    
    # Calculate errors
    errors = valid_actual - valid_forecast
    abs_errors = np.abs(errors)
    
    # Simple metrics
    mae = np.mean(abs_errors)
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    
    # Smoothed MAD calculation
    mad_values = [abs_errors[0]]
    for i in range(1, len(abs_errors)):
        mad_values.append(alpha * abs_errors[i] + (1-alpha) * mad_values[-1])
    
    # Smoothed ERR calculation
    err_values = [errors[0]]
    for i in range(1, len(errors)):
        err_values.append(alpha * errors[i] + (1-alpha) * err_values[-1])
    
    # SIG calculation (bias indicator)
    sig_values = [mad_values[i] / err_values[i] if err_values[i] != 0 else 0 
                 for i in range(len(err_values))]
    
    # Log results
    log(f"{label} (MAE)", mae, suffix=suffix)
    log(f"{label} (MSE)", mse, suffix=suffix)
    log(f"{label} (RMSE)", rmse, suffix=suffix)
    log(f"{label} (Final MAD)", mad_values[-1], suffix=suffix)
    log(f"{label} (Final ERR)", err_values[-1], suffix=suffix)
    log(f"{label} (Final SIG)", sig_values[-1], suffix=suffix)
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mad_values": np.array(mad_values),
        "err_values": np.array(err_values),
        "sig_values": np.array(sig_values)
    }


def compare_forecast_models(demand, models=None, label="Forecast comparison", suffix=""):
    """
    Compare different forecasting models on the same historical data
    
    Parameters:
        demand: Array of historical demand
        models: Dictionary of model configurations to run
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with results from each model and comparison
    """
    log("Historical demand", demand, suffix=suffix)
    
    if models is None:
        # Default models to compare
        models = {
            "Moving Average (2)": {"type": "MA", "window": 2},
            "Moving Average (3)": {"type": "MA", "window": 3},
            "Exp Smoothing (0.1)": {"type": "ES", "alpha": 0.1, "initial": demand[0]},
            "Exp Smoothing (0.2)": {"type": "ES", "alpha": 0.2, "initial": demand[0]},
            "Holt's Trend (0.2,0.1)": {"type": "Holt", "alpha": 0.2, "beta": 0.1, 
                                     "initial_level": demand[0], "initial_trend": 0}
        }
    
    results = {}
    best_mae = float('inf')
    best_model = None
    
    for model_name, config in models.items():
        log(f"{label} (running model)", model_name, suffix=suffix)
        
        if config["type"] == "MA":
            forecast = moving_average(demand, config["window"], f"{model_name}", suffix)
            
        elif config["type"] == "ES":
            forecast = exp_smoothing(config["alpha"], demand, config["initial"], f"{model_name}", suffix)
            
        elif config["type"] == "Holt":
            result = holts_trend_forecasting(
                demand, config["initial_level"], config["initial_trend"],
                config["alpha"], config["beta"], label=f"{model_name}", suffix=suffix
            )
            forecast = result["fitted"]
            
        # Calculate error metrics
        metrics = forecast_error_metrics(demand, forecast, label=f"{model_name} metrics", suffix=suffix)
        
        # Store results
        results[model_name] = {
            "forecast": forecast,
            "metrics": metrics
        }
        
        # Check if this is the best model
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_model = model_name
            
    log(f"{label} (best model)", best_model, suffix=suffix)
    log(f"{label} (best MAE)", best_mae, suffix=suffix)
    
    return {
        "results": results,
        "best_model": best_model,
        "best_mae": best_mae
    }


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

    for i in range(len(values)):
        if i == 0:
            x = np.array([indices[i]])  # Interval
            a = np.array([values[i]])  # Demand size
            forecast_day = np.array([math.floor(indices[i] + x[-1])])
            forecast_quantity = np.array([math.ceil(a[-1])])

            log(f"{label} (initial interval)", x[-1], suffix=suffix)
            log(f"{label} (initial demand size)", a[-1], suffix=suffix)
            log(f"{label} (initial forecast day)", forecast_day[-1], suffix=suffix)
            log(
                f"{label} (initial forecast quantity)",
                forecast_quantity[-1],
                suffix=suffix,
            )
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
    df = pd.DataFrame(
        {
            "x": x,  # Interval estimate
            "a": a,  # Demand size estimate
            "forecast_day": forecast_day,  # Next forecast day
            "forecast_quantity": forecast_quantity,  # Next forecast quantity
        }
    )

    log(f"{label} (results)", "DataFrame created", suffix=suffix)

    return df


#####################################################
# Multi-Period Inventory Models
#####################################################


def least_unit_cost_criterion(demand, period, end_period, setup_cost, holding_cost, label="LUC", suffix=""):
    """
    Calculate the Least Unit Cost criterion for lot-sizing decisions.
    
    Parameters:
        demand: Array of demand values
        period: Current period
        end_period: End period to consider
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Unit cost
    """
    holding_periods = [i for i in range(end_period - period + 1)]
    total_demand = sum(demand[period:end_period+1])
    
    if total_demand == 0:
        log(f"{label} (period {period}-{end_period} criterion)", float("inf"), suffix=suffix)
        return float("inf")
    
    total_holding_cost = holding_cost * sum([demand[period+i] * i for i in holding_periods])
    unit_cost = (setup_cost + total_holding_cost) / total_demand
    
    log(f"{label} (period {period}-{end_period} total demand)", total_demand, suffix=suffix)
    log(f"{label} (period {period}-{end_period} total holding cost)", total_holding_cost, suffix=suffix)
    log(f"{label} (period {period}-{end_period} criterion)", unit_cost, suffix=suffix)
    
    return unit_cost


def silver_meal_criterion(demand, period, end_period, setup_cost, holding_cost, label="SM", suffix=""):
    """
    Calculate the Silver-Meal criterion for lot-sizing decisions.
    
    Parameters:
        demand: Array of demand values
        period: Current period
        end_period: End period to consider
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Period cost
    """
    holding_periods = [i for i in range(end_period - period + 1)]
    total_holding_cost = holding_cost * sum([demand[period+i] * i for i in holding_periods])
    periods_covered = end_period - period + 1
    
    period_cost = (setup_cost + total_holding_cost) / periods_covered
    
    log(f"{label} (period {period}-{end_period} periods covered)", periods_covered, suffix=suffix)
    log(f"{label} (period {period}-{end_period} total holding cost)", total_holding_cost, suffix=suffix)
    log(f"{label} (period {period}-{end_period} criterion)", period_cost, suffix=suffix)
    
    return period_cost


def make_lot_sizing_decision(demands, setup_cost, holding_cost, criterion_func, label="Lot-sizing", suffix=""):
    """
    Make lot-sizing decisions using the specified criterion function.
    
    Parameters:
        demands: Array of demand values
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        criterion_func: Function to calculate the criterion (LUC or Silver-Meal)
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Tuple of (setup_decision, lot_sizes)
    """
    num_periods = len(demands)
    setup_decision = np.full(num_periods, False, dtype=bool)
    lot_sizes = np.zeros(num_periods)
    
    t = 0
    while t < num_periods:
        z = t
        c_opt = criterion_func(demands, t, z, setup_cost, holding_cost, label, suffix)
        
        # Find the optimal end period for this lot size
        while z < num_periods - 1:
            next_c = criterion_func(demands, t, z+1, setup_cost, holding_cost, label, suffix)
            if next_c >= c_opt:  # Stop when the criterion starts to increase
                break
            z += 1
            c_opt = next_c
        
        log(f"{label} (starting at period {t}, ending at period {z})", "Selected", suffix=suffix)
        
        # Set the decision
        setup_decision[t] = True
        lot_size = sum(demands[t:z+1])
        lot_sizes[t] = lot_size
        log(f"{label} (period {t} lot size)", lot_size, unit="units", suffix=suffix)
        
        # Move to the next period after this lot-size
        t = z + 1
    
    return setup_decision, lot_sizes


def calc_inventory_costs(demands, setup_decision, lot_sizes, setup_cost, holding_cost, label="Inventory cost", suffix=""):
    """
    Calculate total inventory costs given a lot-sizing decision.
    
    Parameters:
        demands: Array of demand values
        setup_decision: Boolean array indicating periods with setups
        lot_sizes: Array of lot sizes for each period
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Total cost
    """
    num_periods = len(demands)
    total_setup_cost = setup_cost * np.sum(setup_decision)
    
    # Calculate inventory levels and holding costs
    inventory = np.zeros(num_periods)
    inventory[0] = lot_sizes[0] - demands[0]
    for t in range(1, num_periods):
        inventory[t] = inventory[t-1] + lot_sizes[t] - demands[t]
    
    total_holding_cost = holding_cost * np.sum(inventory)
    total_cost = total_setup_cost + total_holding_cost
    
    log(f"{label} (setup cost)", total_setup_cost, suffix=suffix)
    log(f"{label} (inventory levels)", inventory, suffix=suffix)
    log(f"{label} (holding cost)", total_holding_cost, suffix=suffix)
    log(f"{label} (total cost)", total_cost, suffix=suffix)
    
    return total_cost


def least_unit_cost(demands, setup_cost, holding_cost, label="Least Unit Cost", suffix=""):
    """
    Apply the Least Unit Cost (LUC) heuristic for lot-sizing.
    
    Parameters:
        demands: Array of demand values
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Dictionary with setup_decision, lot_sizes, and total_cost
    """
    log(f"{label} (demands)", demands, suffix=suffix)
    log(f"{label} (setup cost)", setup_cost, suffix=suffix)
    log(f"{label} (holding cost)", holding_cost, suffix=suffix)
    
    setup_decision, lot_sizes = make_lot_sizing_decision(
        demands, setup_cost, holding_cost, least_unit_cost_criterion, label, suffix
    )
    
    total_cost = calc_inventory_costs(
        demands, setup_decision, lot_sizes, setup_cost, holding_cost, label, suffix
    )
    
    result = {
        "setup_decision": setup_decision,
        "lot_sizes": lot_sizes,
        "total_cost": total_cost
    }
    
    log(f"{label} (setup periods)", np.where(setup_decision)[0], suffix=suffix)
    log(f"{label} (result)", "Completed", suffix=suffix)
    
    return result


def silver_meal(demands, setup_cost, holding_cost, label="Silver-Meal", suffix=""):
    """
    Apply the Silver-Meal heuristic for lot-sizing.
    
    Parameters:
        demands: Array of demand values
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Dictionary with setup_decision, lot_sizes, and total_cost
    """
    log(f"{label} (demands)", demands, suffix=suffix)
    log(f"{label} (setup cost)", setup_cost, suffix=suffix)
    log(f"{label} (holding cost)", holding_cost, suffix=suffix)
    
    setup_decision, lot_sizes = make_lot_sizing_decision(
        demands, setup_cost, holding_cost, silver_meal_criterion, label, suffix
    )
    
    total_cost = calc_inventory_costs(
        demands, setup_decision, lot_sizes, setup_cost, holding_cost, label, suffix
    )
    
    result = {
        "setup_decision": setup_decision,
        "lot_sizes": lot_sizes,
        "total_cost": total_cost
    }
    
    log(f"{label} (setup periods)", np.where(setup_decision)[0], suffix=suffix)
    log(f"{label} (result)", "Completed", suffix=suffix)
    
    return result


def wagner_whitin(demands, setup_cost, holding_cost, label="Wagner-Whitin", suffix=""):
    """
    Apply the Wagner-Whitin algorithm for optimal lot-sizing.
    
    Parameters:
        demands: Array of demand values
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Dictionary with setup_decision, lot_sizes, and total_cost
    """
    log(f"{label} (demands)", demands, suffix=suffix)
    log(f"{label} (setup cost)", setup_cost, suffix=suffix)
    log(f"{label} (holding cost)", holding_cost, suffix=suffix)
    
    num_periods = len(demands)
    
    # Initialize cost and decision matrices
    # F[t] = minimum cost from period 0 to t
    F = np.zeros(num_periods)
    # P[t] = the period from which the production for period t occurs
    P = np.zeros(num_periods, dtype=int)
    
    # Base case: Period 0
    F[0] = setup_cost
    P[0] = 0
    
    # Fill in the rest of the DP table
    for t in range(1, num_periods):
        # Initialize with ordering in period t
        F[t] = F[t-1] + setup_cost
        P[t] = t
        
        # Consider producing for period t from period j (0 ≤ j < t)
        for j in range(t):
            # Calculate holding cost if we produce for period t in period j
            holding = 0
            for k in range(j+1, t+1):
                holding += holding_cost * demands[k] * (k - j)
            
            # Calculate cost of this policy
            candidate_cost = (F[j-1] if j > 0 else 0) + setup_cost + holding
            
            # If cheaper, update policy
            if candidate_cost < F[t]:
                F[t] = candidate_cost
                P[t] = j
    
    # Backtracking to find lot sizes
    setup_decision = np.full(num_periods, False, dtype=bool)
    lot_sizes = np.zeros(num_periods)
    
    t = num_periods - 1
    while t >= 0:
        p = P[t]  # Production period for period t
        setup_decision[p] = True
        
        # Add demand for all periods from p to t to the lot size of period p
        for k in range(p, t+1):
            lot_sizes[p] += demands[k]
        
        log(f"{label} (period {p} covers through period {t})", lot_sizes[p], unit="units", suffix=suffix)
        
        t = p - 1  # Move to the period before p
    
    # Calculate total cost
    total_cost = calc_inventory_costs(
        demands, setup_decision, lot_sizes, setup_cost, holding_cost, label, suffix
    )
    
    result = {
        "setup_decision": setup_decision,
        "lot_sizes": lot_sizes,
        "total_cost": total_cost,
        "min_cost": F[num_periods-1]
    }
    
    log(f"{label} (setup periods)", np.where(setup_decision)[0], suffix=suffix)
    log(f"{label} (dp min cost)", F[num_periods-1], suffix=suffix)
    log(f"{label} (result)", "Completed", suffix=suffix)
    
    return result


#####################################################
# Material Requirements Planning (MRP)
#####################################################


def adjusted_mrp_for_multi_echelon(retailer_reqs, wholesaler_reqs, periods_delay=1, 
                                 label="Adjusted MRP", suffix=""):
    """
    Adjust MRP for multi-echelon supply chain by incorporating retailer requirements
    into wholesaler's requirements (From Assignment 4, Question 1)
    
    Parameters:
        retailer_reqs: List of retailer net requirements
        wholesaler_reqs: List of wholesaler base gross requirements
        periods_delay: Delay in periods between retailer order and wholesaler fulfillment (default: 1)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        List of adjusted gross requirements for the wholesaler
    """
    log("Retailer net requirements", retailer_reqs, suffix=suffix)
    log("Wholesaler base requirements", wholesaler_reqs, suffix=suffix)
    log("Periods delay", periods_delay, suffix=suffix)
    
    n = len(wholesaler_reqs)
    adjusted_reqs = [0] * n
    
    for i in range(n):
        # Base wholesaler requirement
        adjusted_reqs[i] = wholesaler_reqs[i]
        
        # Add retailer requirement with delay
        if i >= periods_delay and i - periods_delay < len(retailer_reqs):
            adjusted_reqs[i] += retailer_reqs[i - periods_delay]
    
    log(f"{label} (adjusted requirements)", adjusted_reqs, suffix=suffix)
    return adjusted_reqs


def multi_echelon_mrp_analysis(retailer_gross_reqs, retailer_arrivals, retailer_inventory, retailer_safety_stock,
                             wholesaler_gross_reqs, wholesaler_arrivals, wholesaler_inventory, wholesaler_safety_stock=0,
                             label="Multi-echelon MRP", suffix=""):
    """
    Complete multi-echelon MRP analysis for a retailer-wholesaler system
    (From Assignment 4, Question 1)
    
    Parameters:
        retailer_gross_reqs: List of retailer gross requirements
        retailer_arrivals: List of scheduled retailer arrivals
        retailer_inventory: Initial retailer inventory
        retailer_safety_stock: Retailer safety stock level
        wholesaler_gross_reqs: List of wholesaler base gross requirements
        wholesaler_arrivals: List of scheduled wholesaler arrivals
        wholesaler_inventory: Initial wholesaler inventory
        wholesaler_safety_stock: Wholesaler safety stock level (default: 0)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with MRP results for both echelons
    """
    log("Retailer gross requirements", retailer_gross_reqs, suffix=suffix)
    log("Retailer arrivals", retailer_arrivals, suffix=suffix)
    log("Retailer initial inventory", retailer_inventory, suffix=suffix)
    log("Retailer safety stock", retailer_safety_stock, suffix=suffix)
    
    log("Wholesaler base requirements", wholesaler_gross_reqs, suffix=suffix)
    log("Wholesaler arrivals", wholesaler_arrivals, suffix=suffix)
    log("Wholesaler initial inventory", wholesaler_inventory, suffix=suffix)
    log("Wholesaler safety stock", wholesaler_safety_stock, suffix=suffix)
    
    # Step 1: Calculate retailer's MRP
    retailer_net_reqs = material_requirement_planning(
        retailer_gross_reqs,
        retailer_arrivals,
        retailer_inventory,
        retailer_safety_stock,
        label=f"{label} (retailer)",
        suffix=suffix
    )
    
    # Step 2: Adjust wholesaler's gross requirements based on retailer's net requirements
    wholesaler_adjusted_reqs = adjusted_mrp_for_multi_echelon(
        retailer_net_reqs,
        wholesaler_gross_reqs,
        periods_delay=1,  # Typically 1 period delay
        label=f"{label} (adjusted wholesaler reqs)",
        suffix=suffix
    )
    
    # Step 3: Calculate wholesaler's MRP based on adjusted requirements
    wholesaler_net_reqs = material_requirement_planning(
        wholesaler_adjusted_reqs,
        wholesaler_arrivals,
        wholesaler_inventory,
        wholesaler_safety_stock,
        label=f"{label} (wholesaler)",
        suffix=suffix
    )
    
    return {
        "retailer": {
            "gross_requirements": retailer_gross_reqs,
            "net_requirements": retailer_net_reqs,
            "projected_inventory": [retailer_inventory] + 
                                  [retailer_inventory - sum(retailer_gross_reqs[:i+1]) + 
                                   sum(retailer_arrivals[:i+1]) + 
                                   sum(retailer_net_reqs[:i]) 
                                   for i in range(len(retailer_gross_reqs))]
        },
        "wholesaler": {
            "gross_requirements": wholesaler_gross_reqs,
            "adjusted_requirements": wholesaler_adjusted_reqs,
            "net_requirements": wholesaler_net_reqs,
            "projected_inventory": [wholesaler_inventory] + 
                                  [wholesaler_inventory - sum(wholesaler_adjusted_reqs[:i+1]) + 
                                   sum(wholesaler_arrivals[:i+1]) + 
                                   sum(wholesaler_net_reqs[:i]) 
                                   for i in range(len(wholesaler_adjusted_reqs))]
        }
    }


#####################################################
# Multi-Echelon Inventory Control
#####################################################


def sequential_planning_approach(D, Ar, hr, Aw, hw, label="Sequential planning", suffix=""):
    """
    Sequential planning approach for a two-echelon inventory system
    (From Assignment 4, Question 3)
    
    Parameters:
        D: Annual demand
        Ar: Retailer ordering cost
        hr: Retailer holding cost
        Aw: Warehouse ordering cost
        hw: Warehouse holding cost
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
    
    Returns:
        Dictionary with optimal lot sizes and costs
    """
    log("Annual demand", D, suffix=suffix)
    log("Retailer ordering cost", Ar, suffix=suffix)
    log("Retailer holding cost", hr, suffix=suffix)
    log("Warehouse ordering cost", Aw, suffix=suffix)
    log("Warehouse holding cost", hw, suffix=suffix)
    
    # Step 1: Calculate retailer's optimal lot size using EOQ
    retailer_lot_size = math.sqrt(2 * Ar * D / hr)
    log(f"{label} (retailer lot size)", retailer_lot_size, unit="units", suffix=suffix)
    
    # Step 2: Calculate integer multiplier n
    # The theoretical value for n is sqrt(Aw*hr/(Ar*hw))
    n_theoretical = math.sqrt(Aw * hr / (Ar * hw))
    n = round(n_theoretical)
    if n < 1:
        n = 1  # Ensure n is at least 1
    
    log(f"{label} (theoretical n)", n_theoretical, suffix=suffix)
    log(f"{label} (rounded n)", n, suffix=suffix)
    
    # Step 3: Calculate warehouse lot size
    warehouse_lot_size = n * retailer_lot_size
    log(f"{label} (warehouse lot size)", warehouse_lot_size, unit="units", suffix=suffix)
    
    # Step 4: Calculate costs
    retailer_cost = (D / retailer_lot_size) * Ar + (retailer_lot_size / 2) * hr
    log(f"{label} (retailer cost)", retailer_cost, suffix=suffix)
    
    warehouse_cost = (D / warehouse_lot_size) * Aw + (retailer_lot_size / 2) * hw * (n - 1)
    log(f"{label} (warehouse cost)", warehouse_cost, suffix=suffix)
    
    total_cost = retailer_cost + warehouse_cost
    log(f"{label} (total cost)", total_cost, suffix=suffix)
    
    return {
        "retailer_lot_size": retailer_lot_size,
        "warehouse_lot_size": warehouse_lot_size,
        "n_multiplier": n,
        "retailer_cost": retailer_cost,
        "warehouse_cost": warehouse_cost,
        "total_cost": total_cost
    }


def simultaneous_planning_approach(D, Ar, hr, Aw, hw, label="Simultaneous planning", suffix=""):
    """
    Simultaneous planning approach for a two-echelon inventory system
    (From Assignment 4, Question 3)
    
    Parameters:
        D: Annual demand
        Ar: Retailer ordering cost
        hr: Retailer holding cost
        Aw: Warehouse ordering cost
        hw: Warehouse holding cost
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
    
    Returns:
        Dictionary with optimal lot sizes and costs
    """
    log("Annual demand", D, suffix=suffix)
    log("Retailer ordering cost", Ar, suffix=suffix)
    log("Retailer holding cost", hr, suffix=suffix)
    log("Warehouse ordering cost", Aw, suffix=suffix)
    log("Warehouse holding cost", hw, suffix=suffix)
    
    # Step 1: Calculate integer multiplier n
    # The theoretical value for n
    n_theoretical = math.sqrt(Aw * (hr - hw) / (Ar * hw))
    n = round(n_theoretical)
    if n < 1:
        n = 1  # Ensure n is at least 1
    
    log(f"{label} (theoretical n)", n_theoretical, suffix=suffix)
    log(f"{label} (rounded n)", n, suffix=suffix)
    
    # Step 2: Calculate retailer's lot size
    retailer_lot_size = math.sqrt(
        (2 * D * (Aw/n + Ar)) / (n * hw + hr - hw)
    )
    log(f"{label} (retailer lot size)", retailer_lot_size, unit="units", suffix=suffix)
    
    # Step 3: Calculate warehouse lot size
    warehouse_lot_size = n * retailer_lot_size
    log(f"{label} (warehouse lot size)", warehouse_lot_size, unit="units", suffix=suffix)
    
    # Step 4: Calculate total cost
    total_cost = math.sqrt(
        2 * D * (Aw/n + Ar) * ((n-1) * hw + hr)
    )
    log(f"{label} (total cost)", total_cost, suffix=suffix)
    
    # Calculate individual costs for comparison
    retailer_cost = (D / retailer_lot_size) * Ar + (retailer_lot_size / 2) * hr
    log(f"{label} (retailer cost)", retailer_cost, suffix=suffix)
    
    warehouse_cost = (D / warehouse_lot_size) * Aw + (retailer_lot_size / 2) * hw * (n - 1)
    log(f"{label} (warehouse cost)", warehouse_cost, suffix=suffix)
    
    return {
        "retailer_lot_size": retailer_lot_size,
        "warehouse_lot_size": warehouse_lot_size,
        "n_multiplier": n,
        "retailer_cost": retailer_cost,
        "warehouse_cost": warehouse_cost,
        "total_cost": total_cost
    }


def multi_echelon_comparison(D, Ar, hr, Aw, hw, label="Multi-echelon comparison", suffix=""):
    """
    Compare different planning approaches for a two-echelon inventory system
    (From Assignment 4, Question 3)
    
    Parameters:
        D: Annual demand
        Ar: Retailer ordering cost
        hr: Retailer holding cost
        Aw: Warehouse ordering cost
        hw: Warehouse holding cost
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
    
    Returns:
        Dictionary with results from all approaches
    """
    log("Annual demand", D, suffix=suffix)
    log("Retailer ordering cost", Ar, suffix=suffix)
    log("Retailer holding cost", hr, suffix=suffix)
    log("Warehouse ordering cost", Aw, suffix=suffix)
    log("Warehouse holding cost", hw, suffix=suffix)
    
    # Independent planning
    retailer_lot_size_ind = math.sqrt(2 * Ar * D / hr)
    warehouse_lot_size_ind = math.sqrt(2 * Aw * D / hw)
    
    retailer_cost_ind = (D / retailer_lot_size_ind) * Ar + (retailer_lot_size_ind / 2) * hr
    warehouse_cost_ind = (D / warehouse_lot_size_ind) * Aw + (warehouse_lot_size_ind / 2) * hw
    total_cost_ind = retailer_cost_ind + warehouse_cost_ind
    
    log(f"{label} (Independent retailer lot size)", retailer_lot_size_ind, unit="units", suffix=suffix)
    log(f"{label} (Independent warehouse lot size)", warehouse_lot_size_ind, unit="units", suffix=suffix)
    log(f"{label} (Independent total cost)", total_cost_ind, suffix=suffix)
    
    # Sequential planning
    seq_results = sequential_planning_approach(D, Ar, hr, Aw, hw, f"{label} (Sequential)", suffix)
    
    # Simultaneous planning
    sim_results = simultaneous_planning_approach(D, Ar, hr, Aw, hw, f"{label} (Simultaneous)", suffix)
    
    # Find the best approach
    costs = {
        "Independent": total_cost_ind,
        "Sequential": seq_results["total_cost"],
        "Simultaneous": sim_results["total_cost"]
    }
    best_approach = min(costs, key=costs.get)
    
    log(f"{label} (best approach)", best_approach, suffix=suffix)
    log(f"{label} (best cost)", costs[best_approach], suffix=suffix)
    
    return {
        "independent": {
            "retailer_lot_size": retailer_lot_size_ind,
            "warehouse_lot_size": warehouse_lot_size_ind,
            "retailer_cost": retailer_cost_ind,
            "warehouse_cost": warehouse_cost_ind,
            "total_cost": total_cost_ind
        },
        "sequential": seq_results,
        "simultaneous": sim_results,
        "best_approach": best_approach,
        "best_cost": costs[best_approach]
    }


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


def material_requirement_planning(
    gross_requirements,
    arrivals,
    starting_inventory,
    safety_stock=0,
    label="MRP",
    suffix="",
):
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
            log(
                f"{label} (period {i} after arrivals)",
                projected_inventory,
                suffix=suffix,
            )

            net_requirement = max(
                0, safety_stock + gross_requirements[i] - projected_inventory
            )
            log(f"{label} (period {i} net requirement)", net_requirement, suffix=suffix)
        else:
            projected_inventory = (
                projected_inventory
                - gross_requirements[i - 1]
                + arrivals[i]
                + net_requirements[-1]
            )
            log(
                f"{label} (period {i} after arrivals)",
                projected_inventory,
                suffix=suffix,
            )

            net_requirement = max(
                0, gross_requirements[i] + safety_stock - projected_inventory
            )
            log(f"{label} (period {i} net requirement)", net_requirement, suffix=suffix)

        net_requirements.append(net_requirement)

    log(f"{label} (net requirements)", net_requirements, suffix=suffix)

    return net_requirements


def serial_system_echelon_stock(
    inventory_positions, net_requirements, label="Echelon stock", suffix=""
):
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
            log(
                f"{label} (stage {i} upstream inventory)",
                upstream_inventory,
                suffix=suffix,
            )

            echelon_positions.append(inventory_positions[i] + upstream_inventory)
            log(f"{label} (stage {i} echelon)", echelon_positions[i], suffix=suffix)

    log(label, echelon_positions, suffix=suffix)

    return echelon_positions


def guaranteed_service_model(
    lead_times, holding_costs, demand_mean, std, service_level, 
    review_period=0, candidates=None, label="Guaranteed Service Model", suffix=""
):
    """
    Guaranteed Service Model for multi-echelon inventory optimization

    Parameters:
        lead_times: List of lead times
        holding_costs: List of holding costs
        demand_mean: Mean demand
        std: Standard deviation of demand
        service_level: Service level
        review_period: Review period (default 0)
        candidates: Array of allocation candidates (optional)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Dictionary with results
    """
    n_stage = len(holding_costs)
    safety_factor = inverse_cdf(service_level, label=f"{label} (safety factor)", suffix=suffix)
    
    log("Lead times", lead_times, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Mean demand", demand_mean, suffix=suffix)
    log("Standard deviation", std, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    log("Review period", review_period, suffix=suffix)
    log("Number of stages", n_stage, suffix=suffix)

    # Create allocation candidates if not provided
    if candidates is None:
        import itertools

        candidates = np.array(
            list(itertools.product([0, 1], repeat=n_stage - 1) + [(1,)])
        )
    log(f"{label} (allocation candidates)", candidates, suffix=suffix)

    # Calculate coverage time for each candidate
    def func_coverage_time(decision):
        cover_time = np.zeros(n_stage)
        cumulative_time = 0

        for i in range(n_stage - 1):
            if decision[i] == 1:
                cover_time[i] = lead_times[i] + cumulative_time
                cumulative_time = 0
            else:
                cover_time[i] = 0
                cumulative_time += lead_times[i]

        # Last stage always covers its lead time plus review period
        cover_time[n_stage - 1] = lead_times[-1] + review_period + cumulative_time
        return cover_time

    coverage_times = []
    for candidate in candidates:
        coverage_times.append(func_coverage_time(candidate))

    coverage_times = np.array(coverage_times)
    log(f"{label} (coverage times)", coverage_times, suffix=suffix)

    # Calculate safety stock for each candidate
    safety_stocks = []
    for coverage_time in coverage_times:
        ss = []
        for t in coverage_time:
            ss.append(safety_factor * std * math.sqrt(t) if t > 0 else 0)
        safety_stocks.append(ss)

    safety_stocks = np.array(safety_stocks)
    log(f"{label} (safety stocks)", safety_stocks, suffix=suffix)

    # Calculate total cost for each candidate
    total_costs = np.sum(holding_costs * safety_stocks, axis=1)
    log(f"{label} (total costs)", total_costs, suffix=suffix)

    # Find optimal candidate
    opt_idx = np.argmin(total_costs)
    log(f"{label} (optimal candidate)", candidates[opt_idx], suffix=suffix)
    log(f"{label} (optimal coverage time)", coverage_times[opt_idx], suffix=suffix)
    log(f"{label} (optimal safety stock)", safety_stocks[opt_idx], suffix=suffix)
    log(f"{label} (optimal cost)", total_costs[opt_idx], suffix=suffix)

    return {
        "candidates": candidates,
        "coverage_times": coverage_times,
        "safety_stocks": safety_stocks,
        "total_costs": total_costs,
        "optimal_candidate": candidates[opt_idx],
        "optimal_coverage_time": coverage_times[opt_idx],
        "optimal_safety_stock": safety_stocks[opt_idx],
        "optimal_cost": total_costs[opt_idx],
    }


def clark_scarf_model(lead_times, mu, sigma, holding_costs, penalty_cost, 
                     label="Clark-Scarf model", suffix=""):
    """
    Clark-Scarf model for two-stage serial system

    Parameters:
        lead_times: List of lead times [upstream, downstream]
        mu: Mean demand
        sigma: Standard deviation of demand
        holding_costs: List of holding costs [upstream, downstream]
        penalty_cost: Penalty cost for shortages
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Dictionary with optimal order-up-to levels
    """
    log("Lead times", lead_times, suffix=suffix)
    log("Mean demand", mu, suffix=suffix)
    log("Standard deviation", sigma, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Penalty cost", penalty_cost, suffix=suffix)
    
    # Optimal Solutions Stage 2 (downstream)
    critical_ratio_s2 = (holding_costs[0] + penalty_cost) / (
        holding_costs[1] + penalty_cost
    )
    log(f"{label} (critical ratio S2)", critical_ratio_s2, suffix=suffix)
    
    # Calculate mean and std for stage 2
    mu_s2 = mu * (lead_times[1] + 1)
    sigma_s2 = sigma * math.sqrt(lead_times[1] + 1)
    log(f"{label} (mean demand during S2 horizon)", mu_s2, suffix=suffix)
    log(f"{label} (std dev during S2 horizon)", sigma_s2, suffix=suffix)
    
    opt_s2 = norm.ppf(critical_ratio_s2, mu_s2, sigma_s2)
    log(f"{label} (optimal S2)", opt_s2, suffix=suffix)

    # Optimal Solutions Stage 1 (upstream)
    critical_ratio_s1 = penalty_cost / (holding_costs[1] + penalty_cost)
    log(f"{label} (critical ratio S1)", critical_ratio_s1, suffix=suffix)
    
    # Calculate mean and std for stage 1
    mu_s1 = mu * lead_times[0]
    sigma_s1 = sigma * math.sqrt(lead_times[0])
    log(f"{label} (mean demand during S1 horizon)", mu_s1, suffix=suffix)
    log(f"{label} (std dev during S1 horizon)", sigma_s1, suffix=suffix)
    
    def func_s1(s1):
        # Integrate func from 0 to opt_s2
        integral = integrate.quad(
            lambda d: norm.cdf(s1 - d, mu_s1, sigma_s1) * 
                      norm.pdf(d, mu_s2, sigma_s2),
            0,
            opt_s2,
        )[0]
        return integral - critical_ratio_s1

    opt_s1 = fsolve(func_s1, opt_s2)[0]
    log(f"{label} (optimal S1)", opt_s1, suffix=suffix)
    
    # Calculate cost
    expected_holding_cost = holding_costs[0] * opt_s1 + holding_costs[1] * opt_s2
    log(f"{label} (expected holding cost)", expected_holding_cost, suffix=suffix)

    return {
        "optimal_s2": opt_s2,
        "optimal_s1": opt_s1,
        "critical_ratio_s2": critical_ratio_s2,
        "critical_ratio_s1": critical_ratio_s1,
        "expected_holding_cost": expected_holding_cost
    }


def metric_model(demand_rate, penalty_cost, holding_costs, lead_times, 
               label="METRIC model", suffix=""):
    """
    METRIC model for two-echelon inventory system

    Parameters:
        demand_rate: Mean demand rate (λ)
        penalty_cost: Penalty cost
        holding_costs: List of holding costs [warehouse, retailer]
        lead_times: List of lead times [warehouse, retailer]
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Dictionary with results
    """
    log("Demand rate", demand_rate, suffix=suffix)
    log("Penalty cost", penalty_cost, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Lead times", lead_times, suffix=suffix)
    
    # Bounds of retailer lead time
    lt_r_min = lead_times[1]  # Minimum possible lead time = retailer lead time
    lt_r_max = sum(
        lead_times
    )  # Maximum possible lead time = retailer + warehouse lead time
    log(f"{label} (retailer min lead time)", lt_r_min, suffix=suffix)
    log(f"{label} (retailer max lead time)", lt_r_max, suffix=suffix)

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
        return exp_cost_r, exp_leftover, exp_backorder

    # Find optimal S_r for min/max lead times
    s_r_candidates = range(
        int(demand_rate * lt_r_max * 2)
    )  # Range of possible S_r values

    exp_cost_lt_min = [func_exp_cost_r(s, lt_r_min)[0] for s in s_r_candidates]
    exp_cost_lt_max = [func_exp_cost_r(s, lt_r_max)[0] for s in s_r_candidates]

    sr_min = np.argmin(exp_cost_lt_min)
    sr_max = np.argmin(exp_cost_lt_max)
    log(f"{label} (retailer min base stock)", sr_min, suffix=suffix)
    log(f"{label} (retailer max base stock)", sr_max, suffix=suffix)

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
        ret_cost, ret_leftover, ret_backorder = func_exp_cost_r(base_stock_r, exp_lt_r)
        exp_total_cost = holding_costs[0] * exp_leftover_w + ret_cost
        
        return exp_total_cost, exp_leftover_w, exp_backorder_w, exp_lt_r, ret_leftover, ret_backorder

    # Find bounds on S_w
    s_w_candidates = range(int(demand_rate * lead_times[0] * 2))

    exp_total_cost_sw_min = [func_exp_total_cost(s, sr_max)[0] for s in s_w_candidates]
    exp_total_cost_sw_max = [func_exp_total_cost(s, sr_min)[0] for s in s_w_candidates]

    sw_min = np.argmin(exp_total_cost_sw_min)
    sw_max = np.argmin(exp_total_cost_sw_max)
    log(f"{label} (warehouse min base stock)", sw_min, suffix=suffix)
    log(f"{label} (warehouse max base stock)", sw_max, suffix=suffix)

    # Enumerate all combinations within bounds
    total_costs = np.zeros((sw_max - sw_min + 1, sr_max - sr_min + 1))

    for i in range(sw_max - sw_min + 1):
        for j in range(sr_max - sr_min + 1):
            sw_val = sw_min + i
            sr_val = sr_min + j
            total_costs[i, j] = func_exp_total_cost(sw_val, sr_val)[0]
            log(f"{label} (S_w={sw_val}, S_r={sr_val} cost)", total_costs[i, j], suffix=suffix)

    # Find optimal combination
    min_idx = np.unravel_index(np.argmin(total_costs), total_costs.shape)
    opt_sw = sw_min + min_idx[0]
    opt_sr = sr_min + min_idx[1]
    opt_cost = total_costs[min_idx]
    
    # Get detailed metrics for optimal solution
    _, opt_leftover_w, opt_backorder_w, opt_lt_r, opt_leftover_r, opt_backorder_r = \
        func_exp_total_cost(opt_sw, opt_sr)
    
    log(f"{label} (optimal warehouse base stock)", opt_sw, suffix=suffix)
    log(f"{label} (optimal retailer base stock)", opt_sr, suffix=suffix)
    log(f"{label} (optimal cost)", opt_cost, suffix=suffix)
    log(f"{label} (effective retailer lead time)", opt_lt_r, suffix=suffix)
    log(f"{label} (warehouse leftover inventory)", opt_leftover_w, suffix=suffix)
    log(f"{label} (warehouse backorders)", opt_backorder_w, suffix=suffix)
    log(f"{label} (retailer leftover inventory)", opt_leftover_r, suffix=suffix)
    log(f"{label} (retailer backorders)", opt_backorder_r, suffix=suffix)

    return {
        "optimal_sw": opt_sw,
        "optimal_sr": opt_sr,
        "optimal_cost": opt_cost,
        "sr_bounds": (sr_min, sr_max),
        "sw_bounds": (sw_min, sw_max),
        "total_costs": total_costs,
        "effective_lead_time": opt_lt_r,
        "warehouse_leftover": opt_leftover_w,
        "warehouse_backorder": opt_backorder_w,
        "retailer_leftover": opt_leftover_r,
        "retailer_backorder": opt_backorder_r
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
        log(f"{label} (CV)", float("inf"), suffix=suffix)
        log(f"{label} (classification)", "Z", suffix=suffix)
        return {"classification": "Z", "cv": float("inf")}

    # Calculate coefficient of variation (CV)
    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    cv = std_sales / mean_sales if mean_sales != 0 else float("inf")

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
        "std": std_sales,
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


def economic_lot_scheduling_problem(
    demand_rates, production_rates, setup_costs, holding_costs, label="ELSP", suffix=""
):
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
        log(
            f"{label} (product {i} production time)",
            production_times[i],
            unit="periods",
            suffix=suffix,
        )

    # Calculate cost for each product
    costs = []
    for i in range(n):
        ordering_cost = (demand_rates[i] / Q[i]) * setup_costs[i]
        holding_cost = (
            (holding_costs[i] / 2)
            * (production_rates[i] - demand_rates[i])
            * (Q[i] / production_rates[i])
        )
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
        "min_cycle_time": min(T),
    }

    log(f"{label} (total cost)", result["total_cost"], suffix=suffix)
    log(
        f"{label} (total production time)",
        result["total_production_time"],
        suffix=suffix,
    )
    log(f"{label} (min cycle time)", result["min_cycle_time"], suffix=suffix)

    return result


def common_cycle_approach(
    demand_rates,
    production_rates,
    setup_costs,
    holding_costs,
    setup_times=None,
    label="Common cycle approach",
    suffix="",
):
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
    denominator = sum(
        [
            h * d * (1 - d / p)
            for h, d, p in zip(holding_costs, demand_rates, production_rates)
        ]
    )
    T_unconstrained = math.sqrt(numerator / denominator)

    log(f"{label} (unconstrained numerator)", numerator, suffix=suffix)
    log(f"{label} (unconstrained denominator)", denominator, suffix=suffix)
    log(
        f"{label} (unconstrained cycle time)",
        T_unconstrained,
        unit="periods",
        suffix=suffix,
    )

    # Calculate capacity constraint (if setup times provided)
    if setup_times:
        numerator_c = sum(setup_times)
        denominator_c = 1 - sum([d / p for d, p in zip(demand_rates, production_rates)])
        T_constrained = numerator_c / denominator_c

        log(f"{label} (constrained numerator)", numerator_c, suffix=suffix)
        log(f"{label} (constrained denominator)", denominator_c, suffix=suffix)
        log(
            f"{label} (constrained cycle time)",
            T_constrained,
            unit="periods",
            suffix=suffix,
        )

        T_optimal = max(T_unconstrained, T_constrained)
        log(
            f"{label} (binding constraint)",
            "Capacity" if T_constrained > T_unconstrained else "Cost",
            suffix=suffix,
        )
    else:
        T_optimal = T_unconstrained

    log(f"{label} (optimal cycle time)", T_optimal, unit="periods", suffix=suffix)

    # Calculate lot sizes and costs
    Q_optimal = [d * T_optimal for d in demand_rates]
    costs = []

    for i in range(len(demand_rates)):
        setup_term = setup_costs[i] / T_optimal
        holding_term = (
            (holding_costs[i] / 2)
            * (production_rates[i] - demand_rates[i])
            * (demand_rates[i] / production_rates[i])
            * T_optimal
        )
        total_product_cost = setup_term + holding_term
        costs.append(total_product_cost)

        log(
            f"{label} (product {i} lot size)", Q_optimal[i], unit="units", suffix=suffix
        )
        log(f"{label} (product {i} setup cost)", setup_term, suffix=suffix)
        log(f"{label} (product {i} holding cost)", holding_term, suffix=suffix)
        log(f"{label} (product {i} total cost)", total_product_cost, suffix=suffix)

    total_cost = sum(costs)
    log(f"{label} (total cost)", total_cost, suffix=suffix)

    return {
        "T_optimal": T_optimal,
        "Q_optimal": Q_optimal,
        "costs": costs,
        "total_cost": total_cost,
    }


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
    denominator = r * (1 - d / p)

    log(f"{label} (numerator)", numerator, suffix=suffix)
    log(f"{label} (denominator)", denominator, suffix=suffix)

    emq = math.sqrt(numerator / denominator)
    log(label, emq, unit="units", suffix=suffix)

    return emq


def capacity_constrained_lotsize(
    d, A, h, a, W, label="Capacity constrained lotsize", suffix=""
):
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
    Q_unconstrained = [
        math.sqrt((2 * d_i * A_i) / h_i) for d_i, A_i, h_i in zip(d, A, h)
    ]
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
            "constrained": False,
        }

    # If unconstrained solution is not feasible, find Lagrange multiplier
    log(f"{label} (is constrained)", "Yes", suffix=suffix)

    def equation(lambda_value):
        return (
            sum(
                [
                    a_i * math.sqrt(2 * d_i * A_i / (h_i + 2 * lambda_value * a_i))
                    for d_i, A_i, h_i, a_i in zip(d, A, h, a)
                ]
            )
            - W
        )

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
        "lambda": lambda_value,
    }


def joint_replenishment(d, A, h, A0, label="Joint replenishment", suffix=""):
    """
    Joint Replenishment Problem

    Solves the joint replenishment problem for multiple items with shared setup cost.
    Uses an iterative approach to find the optimal base cycle time and multipliers.

    Parameters:
        d: List of demand rates
        A: List of individual setup costs
        h: List of holding costs
        A0: Major setup cost (shared across items)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios

    Returns:
        Dictionary with results including:
        - base_cycle: optimal base cycle time
        - n_values: integer multipliers for each item
        - individual_cycles: cycle time for each item
        - order_quantities: order quantity for each item
    """
    log("Demand rates", d, suffix=suffix)
    log("Individual setup costs", A, suffix=suffix)
    log("Holding costs", h, suffix=suffix)
    log("Major setup cost", A0, suffix=suffix)

    num_items = len(d)
    log("Number of items", num_items, suffix=suffix)

    # Calculate individual optimal cycle times
    T_i = [math.sqrt(2 * A_i / (h_i * d_i)) for A_i, h_i, d_i in zip(A, h, d)]
    log(f"{label} (individual cycle times)", T_i, suffix=suffix)

    # Find product with minimum cycle time
    min_index = T_i.index(min(T_i))
    log(f"{label} (item with min cycle time)", min_index, suffix=suffix)

    # Initialize n values
    n_values = [
        math.sqrt(A_i * h[min_index] * d[min_index] / (h_i * d_i * (A0 + A[min_index])))
        for A_i, h_i, d_i in zip(A, h, d)
    ]
    n_values = [round(value) for value in n_values]
    log(f"{label} (initial n values)", n_values, suffix=suffix)

    # Iterative improvement
    iteration = 0
    max_iterations = 20  # Prevent infinite loop

    while iteration < max_iterations:
        iteration += 1
        log(f"{label} (iteration)", iteration, suffix=suffix)

        # Calculate base cycle time
        sum_A_div_n = A0 + sum([A_i / n_i for A_i, n_i in zip(A, n_values)])
        sum_h_d_n = sum([h_i * d_i * n_i for h_i, d_i, n_i in zip(h, d, n_values)])
        T = math.sqrt(2 * sum_A_div_n / sum_h_d_n)

        log(f"{label} (setup term)", sum_A_div_n, suffix=suffix)
        log(f"{label} (holding term)", sum_h_d_n, suffix=suffix)
        log(f"{label} (base cycle time)", T, unit="periods", suffix=suffix)

        # Store old n values
        old_n = n_values.copy()

        # Update n values
        for i in range(len(n_values)):
            old_ni = n_values[i]
            while True:
                if n_values[i] * (n_values[i] + 1) >= 2 * A[i] / (h[i] * d[i] * T**2):
                    break
                n_values[i] = n_values[i] + 1

            if old_ni != n_values[i]:
                log(
                    f"{label} (item {i} n value updated)",
                    f"from {old_ni} to {n_values[i]}",
                    suffix=suffix,
                )

        # Recalculate base cycle time
        sum_A_div_n = A0 + sum([A_i / n_i for A_i, n_i in zip(A, n_values)])
        sum_h_d_n = sum([h_i * d_i * n_i for h_i, d_i, n_i in zip(h, d, n_values)])
        T = math.sqrt(2 * sum_A_div_n / sum_h_d_n)

        log(f"{label} (updated base cycle time)", T, unit="periods", suffix=suffix)

        # Check convergence
        if old_n == n_values:
            log(f"{label} (convergence reached)", "Yes", suffix=suffix)
            break

    individual_cycles = [n_i * T for n_i in n_values]
    order_quantities = [d_i * n_i * T for d_i, n_i in zip(d, n_values)]

    for i in range(num_items):
        log(f"{label} (item {i} n value)", n_values[i], suffix=suffix)
        log(
            f"{label} (item {i} cycle time)",
            individual_cycles[i],
            unit="periods",
            suffix=suffix,
        )
        log(
            f"{label} (item {i} order quantity)",
            order_quantities[i],
            unit="units",
            suffix=suffix,
        )

    return {
        "base_cycle": T,
        "n_values": n_values,
        "individual_cycles": individual_cycles,
        "order_quantities": order_quantities,
    }


#####################################################
# Risk Pooling and Correlation
#####################################################


def risk_pooling_color_products(silver_demand, black_demand, price, cost, salvage, 
                              common_cost=None, color_specific_cost=None, 
                              label="Risk pooling for color products", suffix=""):
    """
    Risk pooling analysis for products with different colors
    (From Assignment 4, Question 4)
    
    Parameters:
        silver_demand: List of historical demand for silver product
        black_demand: List of historical demand for black product
        price: Selling price per unit
        cost: Procurement cost per unit for separate ordering
        salvage: Salvage value per unit
        common_cost: Common cost component for pooled ordering (optional)
        color_specific_cost: Color-specific cost component for pooled ordering (optional)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimal order quantities and expected profits
    """
    import numpy as np
    
    # Calculate correlation
    corr = coefficient_of_correlation(silver_demand, black_demand, 
                                     label=f"{label} (correlation)", suffix=suffix)
    
    # Calculate statistics for each product
    silver_mean = np.mean(silver_demand)
    silver_std = np.std(silver_demand, ddof=1)
    black_mean = np.mean(black_demand)
    black_std = np.std(black_demand, ddof=1)
    
    log(f"{label} (silver mean)", silver_mean, suffix=suffix)
    log(f"{label} (silver std)", silver_std, suffix=suffix)
    log(f"{label} (black mean)", black_mean, suffix=suffix)
    log(f"{label} (black std)", black_std, suffix=suffix)
    
    # Separate newsvendor analysis
    # Critical ratio
    separate_CR = newsvendor_critical_ratio(price, cost, salvage, 
                                          label=f"{label} (separate CR)", suffix=suffix)
    
    # Optimal order quantities
    silver_Q = newsvendor_normal(silver_mean, silver_std, separate_CR, 
                               label=f"{label} (silver order quantity)", suffix=suffix)
    black_Q = newsvendor_normal(black_mean, black_std, separate_CR, 
                              label=f"{label} (black order quantity)", suffix=suffix)
    
    # Calculate expected profits
    # First calculate G(z)
    z = inverse_cdf(separate_CR, label=f"{label} (z-value)", suffix=suffix)
    g_z = G_z(z, label=f"{label} (G(z))", suffix=suffix)
    
    # Calculate expected lost sales and expected sales
    silver_ELS = silver_std * g_z
    silver_ES = silver_mean - silver_ELS
    silver_ELO = silver_Q - silver_ES  # Expected leftover inventory
    
    black_ELS = black_std * g_z
    black_ES = black_mean - black_ELS
    black_ELO = black_Q - black_ES
    
    log(f"{label} (silver expected lost sales)", silver_ELS, suffix=suffix)
    log(f"{label} (silver expected sales)", silver_ES, suffix=suffix)
    log(f"{label} (silver expected leftover)", silver_ELO, suffix=suffix)
    
    log(f"{label} (black expected lost sales)", black_ELS, suffix=suffix)
    log(f"{label} (black expected sales)", black_ES, suffix=suffix)
    log(f"{label} (black expected leftover)", black_ELO, suffix=suffix)
    
    # Calculate expected profits
    silver_profit = -cost * silver_Q + price * silver_ES + salvage * silver_ELO
    black_profit = -cost * black_Q + price * black_ES + salvage * black_ELO
    total_separate_profit = silver_profit + black_profit
    
    log(f"{label} (silver expected profit)", silver_profit, suffix=suffix)
    log(f"{label} (black expected profit)", black_profit, suffix=suffix)
    log(f"{label} (total separate profit)", total_separate_profit, suffix=suffix)
    
    # Pooled analysis (if common_cost and color_specific_cost are provided)
    pooled_results = None
    if common_cost is not None and color_specific_cost is not None:
        # Calculate combined demand
        combined_demand = [s + b for s, b in zip(silver_demand, black_demand)]
        combined_mean = np.mean(combined_demand)
        combined_std = np.std(combined_demand, ddof=1)
        
        log(f"{label} (combined mean)", combined_mean, suffix=suffix)
        log(f"{label} (combined std)", combined_std, suffix=suffix)
        
        # Calculate new critical ratio based on pooled cost structure
        total_cost = common_cost + color_specific_cost
        underage_cost = price - total_cost
        overage_cost = common_cost - salvage  # Only common cost is recovered in salvage
        
        pooled_CR = newsvendor_critical_ratio(underage_cost, overage_cost, 0, 
                                           label=f"{label} (pooled CR)", suffix=suffix)
        
        # Calculate optimal order quantity
        pooled_Q = newsvendor_normal(combined_mean, combined_std, pooled_CR, 
                                   label=f"{label} (pooled order quantity)", suffix=suffix)
        
        # Calculate expected profit
        pooled_z = inverse_cdf(pooled_CR, label=f"{label} (pooled z-value)", suffix=suffix)
        pooled_g_z = G_z(pooled_z, label=f"{label} (pooled G(z))", suffix=suffix)
        
        pooled_ELS = combined_std * pooled_g_z
        pooled_ES = combined_mean - pooled_ELS
        pooled_ELO = pooled_Q - pooled_ES
        
        log(f"{label} (pooled expected lost sales)", pooled_ELS, suffix=suffix)
        log(f"{label} (pooled expected sales)", pooled_ES, suffix=suffix)
        log(f"{label} (pooled expected leftover)", pooled_ELO, suffix=suffix)
        
        # Calculate expected profit
        pooled_profit = -total_cost * pooled_Q + price * pooled_ES + salvage * pooled_ELO
        log(f"{label} (pooled expected profit)", pooled_profit, suffix=suffix)
        
        # Benefit of risk pooling
        profit_benefit = pooled_profit - total_separate_profit
        profit_benefit_percent = (profit_benefit / abs(total_separate_profit)) * 100
        
        log(f"{label} (profit benefit)", profit_benefit, suffix=suffix)
        log(f"{label} (profit benefit percentage)", profit_benefit_percent, unit="%", suffix=suffix)
        
        pooled_results = {
            "combined_mean": combined_mean,
            "combined_std": combined_std,
            "pooled_CR": pooled_CR,
            "pooled_Q": pooled_Q,
            "pooled_profit": pooled_profit,
            "profit_benefit": profit_benefit,
            "profit_benefit_percent": profit_benefit_percent
        }
    
    return {
        "correlation": corr,
        "silver": {
            "mean": silver_mean,
            "std": silver_std,
            "order_quantity": silver_Q,
            "expected_profit": silver_profit
        },
        "black": {
            "mean": black_mean,
            "std": black_std,
            "order_quantity": black_Q,
            "expected_profit": black_profit
        },
        "total_separate_profit": total_separate_profit,
        "pooled": pooled_results
    }


def risk_pooling_effect_with_correlation(mean1, std1, mean2, std2, correlation, service_level,
                                       label="Risk pooling with correlation", suffix=""):
    """
    Calculate risk pooling effect with correlation for two products
    
    Parameters:
        mean1: Mean demand of first product
        std1: Standard deviation of first product
        mean2: Mean demand of second product
        std2: Standard deviation of second product
        correlation: Correlation coefficient between demands
        service_level: Service level target
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with safety stock comparison
    """
    import math
    
    log("Mean demand 1", mean1, suffix=suffix)
    log("Std dev 1", std1, suffix=suffix)
    log("Mean demand 2", mean2, suffix=suffix)
    log("Std dev 2", std2, suffix=suffix)
    log("Correlation", correlation, suffix=suffix)
    log("Service level", service_level, suffix=suffix)
    
    # Calculate z-value for service level
    z = inverse_cdf(service_level, label=f"{label} (z-value)", suffix=suffix)
    
    # Calculate individual safety stocks
    ss1 = z * std1
    ss2 = z * std2
    total_individual_ss = ss1 + ss2
    
    log(f"{label} (SS product 1)", ss1, suffix=suffix)
    log(f"{label} (SS product 2)", ss2, suffix=suffix)
    log(f"{label} (total individual SS)", total_individual_ss, suffix=suffix)
    
    # Calculate pooled safety stock
    pooled_variance = std1**2 + std2**2 + 2 * correlation * std1 * std2
    pooled_std = math.sqrt(pooled_variance)
    pooled_ss = z * pooled_std
    
    log(f"{label} (pooled variance)", pooled_variance, suffix=suffix)
    log(f"{label} (pooled std dev)", pooled_std, suffix=suffix)
    log(f"{label} (pooled SS)", pooled_ss, suffix=suffix)
    
    # Calculate risk pooling effect
    ss_reduction = total_individual_ss - pooled_ss
    ss_reduction_percent = (ss_reduction / total_individual_ss) * 100
    
    log(f"{label} (SS reduction)", ss_reduction, suffix=suffix)
    log(f"{label} (SS reduction percentage)", ss_reduction_percent, unit="%", suffix=suffix)
    
    # Calculate coefficient of variation
    cv1 = std1 / mean1
    cv2 = std2 / mean2
    pooled_mean = mean1 + mean2
    pooled_cv = pooled_std / pooled_mean
    
    log(f"{label} (CV product 1)", cv1, suffix=suffix)
    log(f"{label} (CV product 2)", cv2, suffix=suffix)
    log(f"{label} (pooled CV)", pooled_cv, suffix=suffix)
    
    return {
        "z_value": z,
        "individual_ss": [ss1, ss2],
        "total_individual_ss": total_individual_ss,
        "pooled_std": pooled_std,
        "pooled_ss": pooled_ss,
        "ss_reduction": ss_reduction,
        "ss_reduction_percent": ss_reduction_percent,
        "cv": [cv1, cv2, pooled_cv]
    }


def risk_pooling_benefit(
    demand_means,
    demand_stds,
    correlation_matrix,
    service_level,
    label="Risk pooling benefit",
    suffix="",
):
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
        "relative_benefit": relative_benefit,
    }


def risk_pooling_correlation(
    mu_A,
    sigma_A,
    mu_B,
    sigma_B,
    rho,
    p,
    c,
    g=0,
    label="Risk pooling correlation",
    suffix="",
):
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
        "CV_joint": CV_joint,
    }


def coefficient_of_correlation(
    data1, data2, label="Correlation coefficient", suffix=""
):
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


def bullwhip_effect(
    orders_variance, demand_variance, label="Bullwhip effect", suffix=""
):
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


def compare_scenarios(
    value1, value2, label="Comparison", unit="", suffix1="A", suffix2="B"
):
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


#####################################################
# Statistical Estimation Methods
#####################################################


def sample_mean(data, label="Sample mean", suffix=""):
    """
    Calculate sample mean estimator: μ = (1/T) * sum(dt) from t=1 to T
    
    Parameters:
        data: Array of data points
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Sample mean
    """
    mean = np.mean(data)
    log(label, mean, suffix=suffix)
    return mean


def sample_std_dev(data, label="Sample standard deviation", suffix=""):
    """
    Calculate sample standard deviation estimator: 
    σ = sqrt((1/(T-1)) * sum((dt - μ)^2))
    
    Parameters:
        data: Array of data points
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Sample standard deviation
    """
    std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation
    log(label, std_dev, suffix=suffix)
    return std_dev


def estimate_gamma_params(data, label="Gamma parameters", suffix=""):
    """
    Estimate gamma distribution parameters:
    α = μ^2/σ^2, β = σ^2/μ
    
    Parameters:
        data: Array of data points
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with alpha (shape) and beta (scale) parameters
    """
    mu = sample_mean(data, label=f"{label} (mean)", suffix=suffix)
    sigma = sample_std_dev(data, label=f"{label} (std dev)", suffix=suffix)
    
    alpha = mu**2 / sigma**2  # Shape parameter
    beta = sigma**2 / mu      # Scale parameter
    
    log(f"{label} (alpha)", alpha, suffix=suffix)
    log(f"{label} (beta)", beta, suffix=suffix)
    
    return {"alpha": alpha, "beta": beta}


def compound_poisson_estimators(data, T, label="Compound Poisson", suffix=""):
    """
    Estimate parameters for compound Poisson distribution:
    λ = -ln(n0/T), μc = μ/λ
    
    Parameters:
        data: Array of observed demand values (can contain zeros)
        T: Total number of time periods observed
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with lambda (arrival rate) and mu_c (individual demand size mean)
    """
    # Calculate n0 (number of periods with zero demand)
    n0 = np.sum(np.array(data) == 0)
    log(f"{label} (zero demand periods)", n0, suffix=suffix)
    
    # Calculate lambda (arrival rate)
    if n0 == 0:
        lam = T  # Maximum likelihood value when n0=0
    elif n0 == T:
        lam = 0.01  # Minimum value when all periods have zero demand
    else:
        lam = -math.log(n0/T)
    log(f"{label} (lambda)", lam, suffix=suffix)
    
    # Calculate overall mean
    mu = sample_mean(data, label=f"{label} (overall mean)", suffix=suffix)
    
    # Calculate mu_c (mean individual demand size)
    if lam == 0:
        mu_c = 0  # No arrivals
    else:
        mu_c = mu / lam
    log(f"{label} (mu_c)", mu_c, suffix=suffix)
    
    return {"lambda": lam, "mu_c": mu_c}


#####################################################
# Regression Methods
#####################################################


def demand_price_regression(price, demand, label="Demand-price regression", suffix=""):
    """
    Linear regression model for price-demand relationship
    
    Parameters:
        price: Array of price values
        demand: Array of corresponding demand values
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with regression results
    """
    log("Price values", price, suffix=suffix)
    log("Demand values", demand, suffix=suffix)
    
    # Calculate regression coefficients manually
    X = np.array(price).reshape(-1, 1)
    y = np.array(demand)
    
    n = len(price)
    mean_x = np.mean(price)
    mean_y = np.mean(demand)
    
    # Calculate slope (coefficient)
    numerator = sum((price[i] - mean_x) * (demand[i] - mean_y) for i in range(n))
    denominator = sum((price[i] - mean_x)**2 for i in range(n))
    
    coefficient = numerator / denominator if denominator != 0 else 0
    intercept = mean_y - coefficient * mean_x
    
    log(f"{label} (intercept)", intercept, suffix=suffix)
    log(f"{label} (price coefficient)", coefficient, suffix=suffix)
    
    # Calculate predictions and R²
    y_pred = intercept + coefficient * X.flatten()
    
    # Calculate R²
    ss_total = sum((y - mean_y)**2)
    ss_residual = sum((y - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    log(f"{label} (R² score)", r2, suffix=suffix)
    
    # Interpretation
    if coefficient > 0:
        interpretation = "Positive relationship (higher price -> higher demand)"
    else:
        interpretation = "Negative relationship (higher price -> lower demand)"
        
    log(f"{label} (interpretation)", interpretation, suffix=suffix)
    
    return {
        "intercept": intercept,
        "coefficient": coefficient,
        "r2": r2,
        "predicted_values": y_pred,
        "formula": f"Demand = {intercept:.2f} + {coefficient:.2f} × Price",
        "interpretation": interpretation
    }


#####################################################
# Statistical Tests
#####################################################


def chi_squared_test(observed, expected, alpha=0.05, label="Chi-squared test", suffix=""):
    """
    Perform Chi-squared goodness of fit test
    
    Parameters:
        observed: Observed frequencies
        expected: Expected frequencies
        alpha: Significance level (default: 0.05)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with test statistics and results
    """
    from scipy.stats import chi2
    
    log("Observed values", observed, suffix=suffix)
    log("Expected values", expected, suffix=suffix)
    log("Significance level", alpha, suffix=suffix)
    
    # Calculate chi-squared statistic
    chi_sq = sum((n_j - s_j)**2 / s_j for n_j, s_j in zip(observed, expected) if s_j > 0)
    log(f"{label} (statistic)", chi_sq, suffix=suffix)
    
    # Degrees of freedom (k-s-1)
    # k = number of categories, s = number of parameters estimated (typically 1)
    # Default is for 1 parameter estimated (s=1)
    df = len(observed) - 1 - 1
    log(f"{label} (degrees of freedom)", df, suffix=suffix)
    
    # Critical value
    critical_value = chi2.ppf(1 - alpha, df)
    log(f"{label} (critical value)", critical_value, suffix=suffix)
    
    # p-value
    p_value = 1 - chi2.cdf(chi_sq, df)
    log(f"{label} (p-value)", p_value, suffix=suffix)
    
    # Test result
    reject_h0 = p_value < alpha
    log(f"{label} (reject H0)", "Yes" if reject_h0 else "No", suffix=suffix)
    
    result = {
        "chi_squared": chi_sq,
        "df": df,
        "critical_value": critical_value,
        "p_value": p_value,
        "reject_h0": reject_h0,
        "interpretation": "Reject H0: Data does not fit the expected distribution" if reject_h0 else
                        "Fail to reject H0: Data fits the expected distribution"
    }
    
    log(f"{label} (interpretation)", result["interpretation"], suffix=suffix)
    
    return result


#####################################################
# Demand Modeling and Forecasting Functions
#####################################################


def constant_model(data, label="Constant model", suffix=""):
    """
    Fit a constant model (y_t = a + ε_t) to time series data
    
    Parameters:
        data: Time series data
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with model parameters and forecasts
    """
    log("Data length", len(data), suffix=suffix)
    
    # Fit constant model (simple mean)
    constant = np.mean(data)
    log(f"{label} (constant)", constant, suffix=suffix)
    
    # Calculate forecasts (same value for all periods)
    forecasts = np.full_like(data, constant, dtype=float)
    
    # Calculate errors
    errors = np.array(data) - forecasts
    
    # Calculate performance metrics
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    
    log(f"{label} (MAE)", mae, suffix=suffix)
    log(f"{label} (MSE)", mse, suffix=suffix)
    log(f"{label} (RMSE)", rmse, suffix=suffix)
    
    return {
        "constant": constant,
        "forecasts": forecasts,
        "errors": errors,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }


def winters_method(data, season_length, alpha=0.2, beta=0.1, gamma=0.1, num_seasons=3, 
                 label="Winters method", suffix=""):
    """
    Winter's triple exponential smoothing method for seasonality with trend
    
    Parameters:
        data: Historical time series data
        season_length: Length of seasonal cycle
        alpha: Level smoothing parameter (0-1)
        beta: Trend smoothing parameter (0-1)
        gamma: Seasonal smoothing parameter (0-1)
        num_seasons: Number of seasons to forecast
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with model components and forecasts
    """
    log("Data length", len(data), suffix=suffix)
    log("Season length", season_length, suffix=suffix)
    log("Alpha (level)", alpha, suffix=suffix)
    log("Beta (trend)", beta, suffix=suffix)
    log("Gamma (season)", gamma, suffix=suffix)
    
    data = np.array(data)
    n = len(data)
    
    # Initialize level, trend, and seasonal components
    level = np.zeros(n)
    trend = np.zeros(n)
    season = np.zeros(n)
    fitted = np.zeros(n)
    
    # Initialize seasonal indices
    season_averages = np.zeros(season_length)
    
    # Calculate initial seasonal indices
    for i in range(season_length):
        indices = range(i, n, season_length)
        if len(indices) > 0:
            season_averages[i] = np.mean(data[indices])
    
    global_average = np.mean(season_averages)
    
    for i in range(season_length):
        season[i] = season_averages[i] / global_average if global_average != 0 else 1.0
    
    # Initialize level and trend
    level[0] = data[0] / season[0] if season[0] != 0 else data[0]
    trend[0] = 0  # Start with no trend
    
    # Apply Winter's method
    for t in range(1, n):
        # Calculate indices considering seasonality
        season_idx = t % season_length
        
        if t < season_length:
            level[t] = alpha * (data[t] / season[season_idx]) + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
            season[season_idx] = gamma * (data[t] / level[t]) + (1 - gamma) * season[season_idx]
        else:
            level[t] = alpha * (data[t] / season[season_idx]) + (1 - alpha) * (level[t-1] + trend[t-1])
            trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
            season[season_idx] = gamma * (data[t] / level[t]) + (1 - gamma) * season[season_idx]
        
        # Calculate fitted values
        fitted[t] = (level[t-1] + trend[t-1]) * season[season_idx]
    
    # Generate forecasts for future periods
    forecasts = []
    for i in range(1, num_seasons * season_length + 1):
        forecast_level = level[-1] + i * trend[-1]
        forecast_season = season[(n + i - 1) % season_length]
        forecast = forecast_level * forecast_season
        forecasts.append(forecast)
    
    # Normalize seasonal components
    season_sum = np.sum(season[-season_length:])
    normalized_season = season * season_length / season_sum if season_sum != 0 else season
    
    # Calculate errors
    errors = data[1:] - fitted[1:]  # Skip first value which has no forecast
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    
    log(f"{label} (final level)", level[-1], suffix=suffix)
    log(f"{label} (final trend)", trend[-1], suffix=suffix)
    log(f"{label} (seasonal factors)", normalized_season[-season_length:], suffix=suffix)
    log(f"{label} (MAE)", mae, suffix=suffix)
    log(f"{label} (RMSE)", rmse, suffix=suffix)
    log(f"{label} (forecasts)", forecasts, suffix=suffix)
    
    return {
        "level": level,
        "trend": trend,
        "season": normalized_season,
        "fitted": fitted,
        "forecasts": forecasts,
        "errors": errors,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }


def croston_sba(demand, alpha=0.1, beta=0.1, label="Croston SBA", suffix=""):
    """
    Croston's method with Syntetos-Boylan Approximation (SBA) for intermittent demand
    
    Parameters:
        demand: Historical demand time series
        alpha: Smoothing parameter for demand size (0-1)
        beta: Smoothing parameter for intervals (0-1)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with forecasts and components
    """
    log("Data length", len(demand), suffix=suffix)
    log("Alpha (demand size)", alpha, suffix=suffix)
    log("Beta (intervals)", beta, suffix=suffix)
    
    demand = np.array(demand)
    n = len(demand)
    
    # Initialize variables
    z = np.zeros(n)      # Smoothed demand size
    x = np.zeros(n)      # Smoothed interval
    forecast = np.zeros(n)  # Croston's forecast
    forecast_sba = np.zeros(n)  # SBA forecast
    last_demand_period = 0
    
    # Find the first non-zero demand
    first_nonzero = np.nonzero(demand)[0]
    if len(first_nonzero) == 0:
        log(f"{label} (warning)", "No non-zero demands found", suffix=suffix)
        return {
            "croston_forecasts": forecast,
            "sba_forecasts": forecast_sba,
            "demand_size": z,
            "intervals": x
        }
    
    first_nonzero = first_nonzero[0]
    
    # Initialize with first non-zero demand
    z[first_nonzero] = demand[first_nonzero]
    x[first_nonzero] = 1  # First interval
    last_demand_period = first_nonzero
    
    # Apply Croston's method
    for t in range(first_nonzero + 1, n):
        if demand[t] > 0:
            # Update interval
            interval = t - last_demand_period
            x[t] = beta * interval + (1 - beta) * x[last_demand_period]
            
            # Update demand size
            z[t] = alpha * demand[t] + (1 - alpha) * z[last_demand_period]
            
            last_demand_period = t
        else:
            # Carry forward previous estimates
            z[t] = z[last_demand_period]
            x[t] = x[last_demand_period]
        
        # Standard Croston forecast
        if x[t] > 0:
            forecast[t] = z[t] / x[t]
        
        # Syntetos-Boylan Approximation (SBA)
        if x[t] > 0:
            forecast_sba[t] = (1 - beta/2) * z[t] / x[t]
    
    # Calculate errors (only for periods with actual demand)
    demand_periods = demand > 0
    errors_croston = demand[demand_periods] - forecast[demand_periods]
    errors_sba = demand[demand_periods] - forecast_sba[demand_periods]
    
    # Calculate performance metrics
    mae_croston = np.mean(np.abs(errors_croston))
    mse_croston = np.mean(errors_croston**2)
    rmse_croston = np.sqrt(mse_croston)
    
    mae_sba = np.mean(np.abs(errors_sba))
    mse_sba = np.mean(errors_sba**2)
    rmse_sba = np.sqrt(mse_sba)
    
    log(f"{label} (Croston MAE)", mae_croston, suffix=suffix)
    log(f"{label} (Croston RMSE)", rmse_croston, suffix=suffix)
    log(f"{label} (SBA MAE)", mae_sba, suffix=suffix)
    log(f"{label} (SBA RMSE)", rmse_sba, suffix=suffix)
    
    return {
        "croston_forecasts": forecast,
        "sba_forecasts": forecast_sba,
        "demand_size": z,
        "intervals": x,
        "mae_croston": mae_croston,
        "mse_croston": mse_croston,
        "rmse_croston": rmse_croston,
        "mae_sba": mae_sba,
        "mse_sba": mse_sba,
        "rmse_sba": rmse_sba
    }


#####################################################
# Warehouse Scheduling and Multi-Item Inventory Models
#####################################################


def warehouse_scheduling_dedicated_capacity(demands, setup_costs, holding_costs, item_sizes, warehouse_capacity, 
                                          label="Warehouse scheduling (dedicated)", suffix=""):
    """
    Solve the warehouse scheduling problem with dedicated capacity constraint
    
    min ∑(h_i * d_i/Q_i * A_i + h_i/2 * Q_i)
    s.t. ∑(a_i * Q_i) ≤ W
         Q_i ≥ 0 for i = 1, 2, ..., N
    
    Parameters:
        demands: List of demand rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        item_sizes: List of item sizes (a_i)
        warehouse_capacity: Total warehouse capacity (W)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimal order quantities and costs
    """
    n = len(demands)
    log("Number of items", n, suffix=suffix)
    log("Demands", demands, suffix=suffix)
    log("Setup costs", setup_costs, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Item sizes", item_sizes, suffix=suffix)
    log("Warehouse capacity", warehouse_capacity, suffix=suffix)
    
    # Calculate EOQ values (unconstrained)
    unconstrained_Q = [eoq(demands[i], setup_costs[i], holding_costs[i], 
                          label=f"{label} (EOQ {i+1})", suffix=suffix) for i in range(n)]
    
    # Check if capacity constraint is active
    total_space = sum(item_sizes[i] * unconstrained_Q[i] for i in range(n))
    log(f"{label} (total space required)", total_space, suffix=suffix)
    
    if total_space <= warehouse_capacity:
        log(f"{label} (constraint)", "Inactive - using EOQ values", suffix=suffix)
        total_cost = sum(
            demands[i] * setup_costs[i] / unconstrained_Q[i] + holding_costs[i] * unconstrained_Q[i] / 2
            for i in range(n)
        )
        log(f"{label} (total cost)", total_cost, suffix=suffix)
        
        return {
            "order_quantities": unconstrained_Q,
            "total_cost": total_cost,
            "lambda": 0,  # Lagrange multiplier
            "constrained": False
        }
    
    # If constrained, solve using Lagrangian approach
    log(f"{label} (constraint)", "Active - using Lagrangian approach", suffix=suffix)
    
    # Define function to find lambda
    def capacity_equation(lambda_val):
        if lambda_val <= 0:
            return float('inf')  # Force positive lambda
        
        Q_vals = [
            math.sqrt(2 * demands[i] * setup_costs[i] / (holding_costs[i] + 2 * lambda_val * item_sizes[i]))
            for i in range(n)
        ]
        
        return sum(item_sizes[i] * Q_vals[i] for i in range(n)) - warehouse_capacity
    
    # Find lambda that satisfies capacity constraint
    lambda_bounds = [1e-10, 1000]  # Bounds for lambda
    
    # Test a range of lambda values to ensure we have proper bounds
    while capacity_equation(lambda_bounds[1]) > 0:
        lambda_bounds[1] *= 10
    
    # Use brentq for root finding (more robust than fsolve for this problem)
    try:
        lambda_val = brentq(capacity_equation, lambda_bounds[0], lambda_bounds[1])
        log(f"{label} (lambda)", lambda_val, suffix=suffix)
        
        # Calculate optimal Q values using lambda
        constrained_Q = [
            math.sqrt(2 * demands[i] * setup_costs[i] / (holding_costs[i] + 2 * lambda_val * item_sizes[i]))
            for i in range(n)
        ]
        
        # Verify capacity constraint
        total_space_used = sum(item_sizes[i] * constrained_Q[i] for i in range(n))
        log(f"{label} (total space used)", total_space_used, suffix=suffix)
        
        # Calculate total cost
        total_cost = sum(
            demands[i] * setup_costs[i] / constrained_Q[i] + holding_costs[i] * constrained_Q[i] / 2
            for i in range(n)
        )
        log(f"{label} (total cost)", total_cost, suffix=suffix)
        
        # Log individual order quantities
        for i in range(n):
            log(f"{label} (Q_{i+1})", constrained_Q[i], suffix=suffix)
            
        return {
            "order_quantities": constrained_Q,
            "total_cost": total_cost,
            "lambda": lambda_val,
            "constrained": True
        }
        
    except Exception as e:
        log(f"{label} (error)", str(e), suffix=suffix)
        return {
            "error": str(e),
            "order_quantities": unconstrained_Q,  # Return unconstrained as fallback
            "total_cost": None,
            "lambda": None,
            "constrained": True
        }


def warehouse_scheduling_average_utilization(demands, setup_costs, holding_costs, item_sizes, warehouse_capacity, 
                                           label="Warehouse scheduling (avg utilization)", suffix=""):
    """
    Solve the warehouse scheduling problem with average utilization constraint
    
    min ∑(h_i * d_i/Q_i * A_i + h_i/2 * Q_i)
    s.t. ∑(0.5 * a_i * Q_i) ≤ W
         Q_i ≥ 0 for i = 1, 2, ..., N
    
    Parameters:
        demands: List of demand rates
        setup_costs: List of setup costs
        holding_costs: List of holding costs
        item_sizes: List of item sizes (a_i)
        warehouse_capacity: Total warehouse capacity (W)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimal order quantities and costs
    """
    n = len(demands)
    log("Number of items", n, suffix=suffix)
    log("Demands", demands, suffix=suffix)
    log("Setup costs", setup_costs, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Item sizes", item_sizes, suffix=suffix)
    log("Warehouse capacity", warehouse_capacity, suffix=suffix)
    
    # Calculate EOQ values (unconstrained)
    unconstrained_Q = [eoq(demands[i], setup_costs[i], holding_costs[i], 
                          label=f"{label} (EOQ {i+1})", suffix=suffix) for i in range(n)]
    
    # Check if capacity constraint is active (using average utilization = 0.5 * a_i * Q_i)
    total_space = sum(0.5 * item_sizes[i] * unconstrained_Q[i] for i in range(n))
    log(f"{label} (total average space required)", total_space, suffix=suffix)
    
    if total_space <= warehouse_capacity:
        log(f"{label} (constraint)", "Inactive - using EOQ values", suffix=suffix)
        total_cost = sum(
            demands[i] * setup_costs[i] / unconstrained_Q[i] + holding_costs[i] * unconstrained_Q[i] / 2
            for i in range(n)
        )
        log(f"{label} (total cost)", total_cost, suffix=suffix)
        
        return {
            "order_quantities": unconstrained_Q,
            "total_cost": total_cost,
            "lambda": 0,  # Lagrange multiplier
            "constrained": False
        }
    
    # If constrained, solve using Lagrangian approach
    log(f"{label} (constraint)", "Active - using Lagrangian approach", suffix=suffix)
    
    # Define function to find lambda
    def capacity_equation(lambda_val):
        if lambda_val <= 0:
            return float('inf')  # Force positive lambda
        
        Q_vals = [
            math.sqrt(2 * demands[i] * setup_costs[i] / (holding_costs[i] + lambda_val * item_sizes[i]))
            for i in range(n)
        ]
        
        return sum(0.5 * item_sizes[i] * Q_vals[i] for i in range(n)) - warehouse_capacity
    
    # Find lambda that satisfies capacity constraint
    lambda_bounds = [1e-10, 1000]  # Bounds for lambda
    
    # Test a range of lambda values to ensure we have proper bounds
    while capacity_equation(lambda_bounds[1]) > 0:
        lambda_bounds[1] *= 10
    
    # Use brentq for root finding (more robust than fsolve for this problem)
    try:
        lambda_val = brentq(capacity_equation, lambda_bounds[0], lambda_bounds[1])
        log(f"{label} (lambda)", lambda_val, suffix=suffix)
        
        # Calculate optimal Q values using lambda
        constrained_Q = [
            math.sqrt(2 * demands[i] * setup_costs[i] / (holding_costs[i] + lambda_val * item_sizes[i]))
            for i in range(n)
        ]
        
        # Verify capacity constraint
        total_space_used = sum(0.5 * item_sizes[i] * constrained_Q[i] for i in range(n))
        log(f"{label} (total average space used)", total_space_used, suffix=suffix)
        
        # Calculate total cost
        total_cost = sum(
            demands[i] * setup_costs[i] / constrained_Q[i] + holding_costs[i] * constrained_Q[i] / 2
            for i in range(n)
        )
        log(f"{label} (total cost)", total_cost, suffix=suffix)
        
        # Log individual order quantities
        for i in range(n):
            log(f"{label} (Q_{i+1})", constrained_Q[i], suffix=suffix)
            
        return {
            "order_quantities": constrained_Q,
            "total_cost": total_cost,
            "lambda": lambda_val,
            "constrained": True
        }
        
    except Exception as e:
        log(f"{label} (error)", str(e), suffix=suffix)
        return {
            "error": str(e),
            "order_quantities": unconstrained_Q,  # Return unconstrained as fallback
            "total_cost": None,
            "lambda": None,
            "constrained": True
        }


def rotation_common_cycle_with_capacity(demands, setup_costs, holding_costs, item_sizes, warehouse_capacity, 
                                       label="Rotation common cycle", suffix=""):
    """
    Solve the rotation common cycle problem with capacity constraints
    
    Parameters:
        demands: List of demand rates (d_i)
        setup_costs: List of setup costs (A_i)
        holding_costs: List of holding costs (h_i)
        item_sizes: List of item sizes (a_i)
        warehouse_capacity: Total warehouse capacity (W)
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimal cycle time, order quantities, and costs
    """
    n = len(demands)
    log("Number of items", n, suffix=suffix)
    log("Demands", demands, suffix=suffix)
    log("Setup costs", setup_costs, suffix=suffix)
    log("Holding costs", holding_costs, suffix=suffix)
    log("Item sizes", item_sizes, suffix=suffix)
    log("Warehouse capacity", warehouse_capacity, suffix=suffix)
    
    # Calculate capacity balance times (t_i)
    sum_ad = sum(item_sizes[i] * demands[i] for i in range(n))
    log(f"{label} (sum a_i * d_i)", sum_ad, suffix=suffix)
    
    t_values = [0]  # t_0 = 0
    for i in range(1, n+1):
        sum_partial = sum(item_sizes[j] * demands[j] for j in range(i))
        t_i = sum_partial / sum_ad
        t_values.append(t_i)
        log(f"{label} (t_{i})", t_i, suffix=suffix)
    
    # Calculate capacity requirement term
    cap_term = 0
    for i in range(n):
        for j in range(i+1):
            cap_term += item_sizes[i] * item_sizes[j] * demands[i] * demands[j]
    
    cap_term = cap_term / sum_ad
    log(f"{label} (capacity term)", cap_term, suffix=suffix)
    
    # Calculate unconstrained optimal cycle time
    cost_sum_A = sum(setup_costs)
    log(f"{label} (sum A_i)", cost_sum_A, suffix=suffix)
    
    cost_sum_hd = sum(holding_costs[i] * demands[i] for i in range(n))
    log(f"{label} (sum h_i * d_i)", cost_sum_hd, suffix=suffix)
    
    T_unconstrained = math.sqrt(2 * cost_sum_A / cost_sum_hd)
    log(f"{label} (unconstrained T*)", T_unconstrained, suffix=suffix)
    
    # Calculate capacity constrained cycle time
    T_constrained = warehouse_capacity * sum_ad / cap_term
    log(f"{label} (capacity constrained T*)", T_constrained, suffix=suffix)
    
    # Choose the minimum (binding constraint)
    T_optimal = min(T_unconstrained, T_constrained)
    log(f"{label} (optimal T*)", T_optimal, suffix=suffix)
    
    is_capacity_binding = (T_optimal == T_constrained)
    log(f"{label} (capacity binding?)", "Yes" if is_capacity_binding else "No", suffix=suffix)
    
    # Calculate optimal order quantities and costs
    Q_optimal = [demands[i] * T_optimal for i in range(n)]
    
    # Calculate total cost
    total_cost = sum(setup_costs) / T_optimal + T_optimal * sum(holding_costs[i] * demands[i] for i in range(n)) / 2
    log(f"{label} (total cost)", total_cost, suffix=suffix)
    
    # Log individual order quantities
    for i in range(n):
        log(f"{label} (Q_{i+1})", Q_optimal[i], suffix=suffix)
        
    # Calculate peak inventory and verify capacity constraint
    peak_inventory = 0
    for i in range(n):
        item_peak = Q_optimal[i] - demands[i] * (T_optimal - t_values[i])
        peak_inventory += item_sizes[i] * item_peak
        
    log(f"{label} (peak inventory space)", peak_inventory, suffix=suffix)
    log(f"{label} (capacity constraint satisfied?)", "Yes" if peak_inventory <= warehouse_capacity else "No", suffix=suffix)
    
    return {
        "cycle_time": T_optimal,
        "order_quantities": Q_optimal,
        "total_cost": total_cost,
        "capacity_binding": is_capacity_binding,
        "capacity_requirement": peak_inventory,
        "time_points": t_values
    }


#####################################################
# Heuristic Models
#####################################################

def average_cost_s_q_policy(D, A, h, p, Q, z, sigma_L, stockout_type="occasion", label="Average cost (s,Q)", suffix=""):
    """
    Calculate average cost for (s,Q) policy with normally distributed demand
    
    Parameters:
        D: Annual demand rate
        A: Setup cost
        h: Holding cost per unit per year
        p: Penalty cost (per stockout occasion or per unit short)
        Q: Order quantity
        z: Safety factor
        sigma_L: Standard deviation of demand during lead time
        stockout_type: Type of stockout cost - "occasion" or "unit"
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Average cost
    """
    log("Annual demand", D, suffix=suffix)
    log("Setup cost", A, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Penalty cost", p, suffix=suffix)
    log("Order quantity", Q, suffix=suffix)
    log("Safety factor (z)", z, suffix=suffix)
    log("Std dev lead time demand", sigma_L, suffix=suffix)
    log("Stockout type", stockout_type, suffix=suffix)
    
    # Calculate ordering cost
    ordering_cost = (D / Q) * A
    log(f"{label} (ordering cost)", ordering_cost, suffix=suffix)
    
    # Calculate holding cost (includes cycle stock and safety stock)
    holding_cost_value = h * (Q/2 + z * sigma_L)
    log(f"{label} (holding cost)", holding_cost_value, suffix=suffix)
    
    # Calculate stockout cost
    if stockout_type == "occasion":
        # Penalty cost per stockout occasion
        phi_z = phi_z(z, label="", suffix="")
        stockout_prob = 1 - Phi_z(z, label="", suffix="")
        stockout_cost = (D/Q) * p * stockout_prob
        log(f"{label} (stockout probability)", stockout_prob, suffix=suffix)
    else:
        # Penalty cost per unit short
        g_z = G_z(z, label="", suffix="")
        stockout_cost = (D/Q) * p * sigma_L * g_z
        log(f"{label} (expected shortage)", sigma_L * g_z, suffix=suffix)
    
    log(f"{label} (stockout cost)", stockout_cost, suffix=suffix)
    
    # Calculate total average cost
    average_cost = ordering_cost + holding_cost_value + stockout_cost
    log(label, average_cost, suffix=suffix)
    
    return average_cost


def joint_optimization_s_q_algorithm(D, A, h, p, mu_L, sigma_L, max_iterations=10, 
                                   stockout_type="occasion", label="Joint optimization", suffix=""):
    """
    Joint optimization algorithm for (s,Q) policy
    
    Parameters:
        D: Annual demand
        A: Setup cost
        h: Holding cost per unit per year
        p: Penalty/stockout cost
        mu_L: Mean demand during lead time
        sigma_L: Standard deviation of demand during lead time
        max_iterations: Maximum number of iterations
        stockout_type: Type of stockout cost - "occasion" or "unit"
        label: Optional label for logging output
        suffix: Optional suffix to differentiate scenarios
        
    Returns:
        Dictionary with optimal s, Q and cost
    """
    log("Annual demand", D, suffix=suffix)
    log("Setup cost", A, suffix=suffix)
    log("Holding cost", h, suffix=suffix)
    log("Penalty cost", p, suffix=suffix)
    log("Mean lead time demand", mu_L, suffix=suffix)
    log("Std dev lead time demand", sigma_L, suffix=suffix)
    log("Stockout type", stockout_type, suffix=suffix)
    
    # Initialize with EOQ
    Q = eoq(D, A, h, label=f"{label} (initial Q)", suffix=suffix)
    
    for i in range(max_iterations):
        log(f"{label} (iteration {i+1})", "", suffix=suffix)
        
        # Step 1: Determine z(Q)
        if stockout_type == "occasion":
            # Formula for penalty cost per stockout occasion
            term = D * p / (math.sqrt(2 * math.pi) * Q * h * sigma_L)
            log(f"{label} (term)", term, suffix=suffix)
            
            if term >= 1:
                z = math.sqrt(2 * math.log(term))
                log(f"{label} (z value)", z, suffix=suffix)
            else:
                z = 0
                log(f"{label} (z value - term < 1)", z, suffix=suffix)
                
        else:
            # Formula for penalty cost per unit short
            term = h * Q / (p * D)
            log(f"{label} (term)", term, suffix=suffix)
            
            if term < 1:
                z = inverse_cdf(1 - term, label=f"{label} (z value)", suffix=suffix)
            else:
                z = 0
                log(f"{label} (z value - term ≥ 1)", z, suffix=suffix)
        
        # Step 2: Determine Q(z)
        if stockout_type == "occasion":
            stockout_prob = 1 - Phi_z(z, label="", suffix="")
            Q_new = math.sqrt(2 * D * (A + p * stockout_prob) / h)
        else:
            g_z = G_z(z, label="", suffix="")
            Q_new = math.sqrt(2 * D * (A + p * sigma_L * g_z) / h)
        
        log(f"{label} (updated Q)", Q_new, suffix=suffix)
        
        # Check convergence
        if abs(Q_new - Q) < 0.01 * Q:
            log(f"{label} (converged)", "Yes", suffix=suffix)
            Q = Q_new
            break
            
        Q = Q_new
    
    # Calculate reorder point s
    s = mu_L + z * sigma_L
    log(f"{label} (reorder point s)", s, suffix=suffix)
    
    # Calculate cost
    cost = average_cost_s_q_policy(D, A, h, p, Q, z, sigma_L, 
                                 stockout_type=stockout_type, 
                                 label=f"{label} (final cost)", 
                                 suffix=suffix)
    
    return {
        "order_quantity": Q,
        "reorder_point": s,
        "safety_factor": z,
        "safety_stock": z * sigma_L,
        "total_cost": cost
    }


def part_period_balancing(demands, setup_cost, holding_cost, label="Part Period Balancing", suffix=""):
    """
    Apply the Part Period Balancing (PPB) heuristic for lot-sizing.
    
    This heuristic tries to balance setup costs against holding costs by using
    the Economic Part Period (EPP) concept.
    
    Parameters:
        demands: Array of demand values
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Dictionary with setup_decision, lot_sizes, and total_cost
    """
    log(f"{label} (demands)", demands, suffix=suffix)
    log(f"{label} (setup cost)", setup_cost, suffix=suffix)
    log(f"{label} (holding cost)", holding_cost, suffix=suffix)
    
    num_periods = len(demands)
    setup_decision = np.full(num_periods, False, dtype=bool)
    lot_sizes = np.zeros(num_periods)
    
    # Economic Part Period (EPP) = Setup Cost / Holding Cost
    epp = setup_cost / holding_cost
    log(f"{label} (Economic Part Period)", epp, suffix=suffix)
    
    t = 0
    while t < num_periods:
        setup_decision[t] = True
        z = t
        holding_parts = 0  # Cumulative part-periods of holding
        
        # Look ahead until holding cost exceeds EPP or end of horizon
        while z < num_periods - 1:
            next_holding = holding_parts + demands[z+1] * (z+1 - t)
            log(f"{label} (period {t}-{z+1} holding part-periods)", next_holding, suffix=suffix)
            
            if next_holding > epp:
                # Check if adding this period gets closer to EPP than excluding it
                if abs(next_holding - epp) < abs(holding_parts - epp):
                    z += 1
                break
            
            holding_parts = next_holding
            z += 1
        
        # Set the lot size for this period
        lot_sizes[t] = sum(demands[t:z+1])
        log(f"{label} (period {t} lot size)", lot_sizes[t], unit="units", suffix=suffix)
        log(f"{label} (covers periods {t} through {z})", "", suffix=suffix)
        
        # Move to the next uncovered period
        t = z + 1
    
    # Calculate total cost
    total_cost = calc_inventory_costs(
        demands, setup_decision, lot_sizes, setup_cost, holding_cost, label, suffix
    )
    
    result = {
        "setup_decision": setup_decision,
        "lot_sizes": lot_sizes,
        "total_cost": total_cost,
        "epp": epp
    }
    
    log(f"{label} (setup periods)", np.where(setup_decision)[0], suffix=suffix)
    log(f"{label} (result)", "Completed", suffix=suffix)
    
    return result


def compare_lot_sizing_methods(demands, setup_cost, holding_cost, label="Lot sizing comparison", suffix=""):
    """
    Compare different lot-sizing methods on the same problem.
    
    Parameters:
        demands: Array of demand values
        setup_cost: Setup cost
        holding_cost: Holding cost per unit per period
        label: Optional label for logging
        suffix: Optional suffix for scenario tracking/logging
    
    Returns:
        Dictionary with results from each method and comparison summary
    """
    log(f"{label} (demands)", demands, suffix=suffix)
    log(f"{label} (setup cost)", setup_cost, suffix=suffix)
    log(f"{label} (holding cost)", holding_cost, suffix=suffix)
    
    # Run all methods
    luc_result = least_unit_cost(demands, setup_cost, holding_cost, f"{label} (LUC)", suffix)
    sm_result = silver_meal(demands, setup_cost, holding_cost, f"{label} (SM)", suffix)
    ppb_result = part_period_balancing(demands, setup_cost, holding_cost, f"{label} (PPB)", suffix)
    ww_result = wagner_whitin(demands, setup_cost, holding_cost, f"{label} (WW)", suffix)
    
    # Collect costs for comparison
    costs = {
        "LUC": luc_result["total_cost"],
        "Silver-Meal": sm_result["total_cost"],
        "Part Period Balancing": ppb_result["total_cost"],
        "Wagner-Whitin": ww_result["total_cost"]
    }
    
    # Find best method
    best_method = min(costs, key=costs.get)
    
    # Calculate optimality gaps
    optimal_cost = costs["Wagner-Whitin"]
    gaps = {}
    for method, cost in costs.items():
        if method != "Wagner-Whitin":
            gap = ((cost - optimal_cost) / optimal_cost) * 100
            gaps[method] = gap
            log(f"{label} ({method} optimality gap)", gap, unit="%", suffix=suffix)
    
    log(f"{label} (best method)", best_method, suffix=suffix)
    log(f"{label} (best cost)", costs[best_method], suffix=suffix)
    
    result = {
        "LUC": luc_result,
        "Silver-Meal": sm_result,
        "Part Period Balancing": ppb_result,
        "Wagner-Whitin": ww_result,
        "costs": costs,
        "optimality_gaps": gaps,
        "best_method": best_method
    }
    
    return result

import math
import numpy as np
from scipy.stats import norm, poisson, gamma

# EOQ (Economic Order Quantity)
def eoq(D, S, h):
    return math.sqrt((2 * D * S) / h)

# EOQ with price breaks (returns EOQ, adjusted for min/max Q)
def eoq_price_break(D, S, i, price, q_min=0, q_max=float("inf")):
    h = i * price
    Q = math.sqrt((2 * D * S) / h)
    Q = max(q_min, min(Q, q_max))
    return Q

# Total annual cost for EOQ
def total_annual_cost(D, S, h, Q, price):
    purchase_cost = D * price
    ordering_cost = S * (D / Q)
    holding_cost = h * (Q / 2)
    return purchase_cost + ordering_cost + holding_cost

# Reorder Point (ROP)
def reorder_point(D, lead_time_months):
    monthly_demand = D / 12
    return monthly_demand * lead_time_months

# Newsvendor critical ratio
def newsvendor_critical_ratio(p, c, g=0):
    return (p - c) / (p - g)

# Newsvendor order quantity (Normal)
def newsvendor_normal(mu, sigma, CR):
    z = norm.ppf(CR)
    Q = mu + z * sigma
    return Q

# Safety stock calculation
def safety_stock(z, std_dev, lead_time, review_period):
    return z * std_dev * math.sqrt(lead_time + review_period)

# Exponential smoothing forecast
def exp_smoothing(alpha, demand, initial_forecast):
    forecasts = [initial_forecast]
    for t in range(len(demand)):
        new_forecast = alpha * demand[t] + (1 - alpha) * forecasts[-1]
        forecasts.append(new_forecast)
    return forecasts[1:]

# Order-up-to level (S)
def order_up_to_level(forecast, safety_stock):
    return np.array(forecast) + safety_stock

# Variance calculation
def sample_variance(data):
    return np.var(data, ddof=1)

# Common replenishment cycle (T*)
def common_cycle(A_list, h_list, d_list):
    total_ordering = 2 * sum(A_list)
    total_holding = sum([h * d for h, d in zip(h_list, d_list)])
    return math.sqrt(total_ordering / total_holding)

# XYZ classification (coefficient of variation)
def xyz_classification(sales):
    mean_sales = np.mean(sales)
    std_sales = np.std(sales)
    cv = std_sales / mean_sales if mean_sales != 0 else float('inf')
    if cv <= 0.5:
        return "X"
    elif cv <= 1.25:
        return "Y"
    else:
        return "Z"
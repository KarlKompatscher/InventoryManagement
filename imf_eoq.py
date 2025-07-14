"""
Inventory Management Formulas - Economic Order Quantity (EOQ) Module

This module provides a unified framework for EOQ calculations and variants.
"""

import math
from imf_utils import log


class EOQ:
    """
    Economic Order Quantity calculation framework.
    
    This class provides methods for calculating various EOQ model variants
    including basic EOQ, quantity discounts, and production models.
    """
    
    @staticmethod
    def basic(demand, setup_cost, holding_cost, label="EOQ", suffix=""):
        """
        Calculate basic Economic Order Quantity
        
        Parameters:
            demand: Annual demand
            setup_cost: Setup cost per order
            holding_cost: Holding cost per unit per year
            label: Optional label for logging output (default: "EOQ")
            suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)
            
        Returns:
            Economic Order Quantity
        """
        log("Annual demand", demand, suffix=suffix)
        log("Setup cost", setup_cost, suffix=suffix)
        log("Holding cost", holding_cost, suffix=suffix)
        
        # Formula: Q* = sqrt(2*D*A/h)
        eoq = math.sqrt(2 * demand * setup_cost / holding_cost)
        log(label, eoq, unit="units", suffix=suffix)
        
        # Calculate related metrics
        cycle_time = eoq / demand
        log("Cycle time", cycle_time, unit="years", suffix=suffix)
        
        avg_inventory = eoq / 2
        log("Average inventory", avg_inventory, unit="units", suffix=suffix)
        
        total_cost = setup_cost * demand / eoq + holding_cost * eoq / 2
        log("Total annual cost", total_cost, unit="currency", suffix=suffix)
        
        return eoq
    
    @staticmethod
    def all_unit_quantity_discount(demand, setup_cost, unit_costs, holding_rate, 
                                  quantity_breaks, label="All-unit quantity discount", suffix=""):
        """
        Calculate EOQ with all-unit quantity discount
        
        Parameters:
            demand: Annual demand
            setup_cost: Setup cost per order
            unit_costs: List of unit costs for different quantity ranges
            holding_rate: Holding cost as a fraction of unit cost
            quantity_breaks: List of quantity break points
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with optimal order quantity and cost
        """
        log("Annual demand", demand, suffix=suffix)
        log("Setup cost", setup_cost, suffix=suffix)
        log("Unit costs", unit_costs, suffix=suffix)
        log("Holding rate", holding_rate, suffix=suffix)
        log("Quantity breaks", quantity_breaks, suffix=suffix)
        
        # Calculate EOQ for each price point
        candidates = []
        
        # First price point (consider EOQ)
        holding_cost = holding_rate * unit_costs[0]
        unconstrained_eoq = math.sqrt(2 * demand * setup_cost / holding_cost)
        log(f"{label} (unconstrained EOQ for price {unit_costs[0]})", 
            unconstrained_eoq, unit="units", suffix=suffix)
        
        if unconstrained_eoq < quantity_breaks[0]:
            # EOQ is feasible
            candidates.append({
                "quantity": unconstrained_eoq,
                "unit_cost": unit_costs[0],
                "holding_cost": holding_cost
            })
        else:
            # EOQ is not feasible, consider the quantity break
            candidates.append({
                "quantity": quantity_breaks[0],
                "unit_cost": unit_costs[0],
                "holding_cost": holding_cost
            })
        
        # Remaining price points
        for i in range(1, len(unit_costs)):
            holding_cost = holding_rate * unit_costs[i]
            unconstrained_eoq = math.sqrt(2 * demand * setup_cost / holding_cost)
            log(f"{label} (unconstrained EOQ for price {unit_costs[i]})", 
                unconstrained_eoq, unit="units", suffix=suffix)
            
            lower_bound = quantity_breaks[i-1]
            upper_bound = quantity_breaks[i] if i < len(quantity_breaks) else float('inf')
            
            if lower_bound <= unconstrained_eoq < upper_bound:
                # EOQ is feasible
                candidates.append({
                    "quantity": unconstrained_eoq,
                    "unit_cost": unit_costs[i],
                    "holding_cost": holding_cost
                })
            else:
                # EOQ is not feasible, consider the lower bound
                candidates.append({
                    "quantity": lower_bound,
                    "unit_cost": unit_costs[i],
                    "holding_cost": holding_cost
                })
        
        # Calculate total cost for each candidate
        min_cost = float('inf')
        optimal = None
        
        for candidate in candidates:
            q = candidate["quantity"]
            c = candidate["unit_cost"]
            h = candidate["holding_cost"]
            
            # Total cost = ordering cost + holding cost + purchase cost
            total_cost = setup_cost * demand / q + h * q / 2 + c * demand
            
            log(f"{label} (q={q:.2f}, c={c}, total cost)", total_cost, suffix=suffix)
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal = {
                    "order_quantity": q,
                    "unit_cost": c,
                    "total_cost": total_cost,
                    "ordering_cost": setup_cost * demand / q,
                    "holding_cost": h * q / 2,
                    "purchase_cost": c * demand
                }
        
        log(f"{label} (optimal quantity)", optimal["order_quantity"], unit="units", suffix=suffix)
        log(f"{label} (optimal unit cost)", optimal["unit_cost"], unit="currency/unit", suffix=suffix)
        log(f"{label} (optimal total cost)", optimal["total_cost"], unit="currency", suffix=suffix)
        
        return optimal
    
    @staticmethod
    def incremental_quantity_discount(demand, setup_cost, unit_costs, holding_rate, 
                                    quantity_breaks, label="Incremental quantity discount", suffix=""):
        """
        Calculate EOQ with incremental quantity discount
        
        Parameters:
            demand: Annual demand
            setup_cost: Setup cost per order
            unit_costs: List of unit costs for different quantity ranges
            holding_rate: Holding cost as a fraction of unit cost
            quantity_breaks: List of quantity break points
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with optimal order quantity and cost
        """
        log("Annual demand", demand, suffix=suffix)
        log("Setup cost", setup_cost, suffix=suffix)
        log("Unit costs", unit_costs, suffix=suffix)
        log("Holding rate", holding_rate, suffix=suffix)
        log("Quantity breaks", quantity_breaks, suffix=suffix)
        
        # Prepare the quantity breaks with 0 as the first break point
        breaks = [0] + quantity_breaks
        
        # Calculate candidates for each price range
        candidates = []
        
        for j in range(1, len(unit_costs) + 1):
            # Calculate R_j (sum of terms independent of Q if q_j-1 ≤ Q < q_j)
            r_j = 0
            for i in range(1, j):
                r_j += unit_costs[i-1] * (breaks[i] - breaks[i-1])
            
            # Calculate optimal quantity for this range
            term = (r_j - unit_costs[j-1] * breaks[j-1] + setup_cost) * demand
            holding_cost = holding_rate * unit_costs[j-1]
            
            q_j = math.sqrt(2 * term / holding_cost)
            log(f"{label} (q_{j} unconstrained)", q_j, unit="units", suffix=suffix)
            
            # Check if q_j is in the valid range
            lower_bound = breaks[j-1]
            upper_bound = breaks[j] if j < len(breaks) else float('inf')
            
            if lower_bound <= q_j < upper_bound:
                # q_j is feasible
                candidates.append({
                    "quantity": q_j,
                    "price_range": j
                })
            elif q_j < lower_bound and j > 1:
                # If q_j < lower_bound, it's not feasible for this range
                pass
            else:
                # Consider the lower bound
                candidates.append({
                    "quantity": lower_bound,
                    "price_range": j
                })
        
        # Calculate total cost for each candidate
        min_cost = float('inf')
        optimal = None
        
        for candidate in candidates:
            q = candidate["quantity"]
            j = candidate["price_range"]
            
            # Calculate purchase cost with incremental pricing
            purchase_cost = 0
            
            for i in range(1, j):
                purchase_cost += unit_costs[i-1] * (breaks[i] - breaks[i-1])
            
            purchase_cost += unit_costs[j-1] * (q - breaks[j-1])
            purchase_cost *= demand / q
            
            # Calculate holding cost (more complex for incremental pricing)
            inventory_value = 0
            
            for i in range(1, j):
                inventory_value += unit_costs[i-1] * (breaks[i] - breaks[i-1])
            
            inventory_value += unit_costs[j-1] * (q - breaks[j-1])
            
            holding_cost = holding_rate * inventory_value / 2
            
            # Calculate ordering cost
            ordering_cost = setup_cost * demand / q
            
            # Total cost
            total_cost = ordering_cost + holding_cost + purchase_cost
            
            log(f"{label} (q={q:.2f}, j={j}, total cost)", total_cost, suffix=suffix)
            
            if total_cost < min_cost:
                min_cost = total_cost
                optimal = {
                    "order_quantity": q,
                    "price_range": j,
                    "total_cost": total_cost,
                    "ordering_cost": ordering_cost,
                    "holding_cost": holding_cost,
                    "purchase_cost": purchase_cost
                }
        
        log(f"{label} (optimal quantity)", optimal["order_quantity"], unit="units", suffix=suffix)
        log(f"{label} (optimal price range)", optimal["price_range"], suffix=suffix)
        log(f"{label} (optimal total cost)", optimal["total_cost"], unit="currency", suffix=suffix)
        
        return optimal
    
    @staticmethod
    def production_quantity(demand, setup_cost, holding_cost, production_rate, 
                          label="EPQ", suffix=""):
        """
        Calculate Economic Production Quantity (EPQ)
        
        Parameters:
            demand: Annual demand rate
            setup_cost: Setup cost per production run
            holding_cost: Holding cost per unit per year
            production_rate: Production rate (units per year)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Economic Production Quantity
        """
        log("Annual demand", demand, suffix=suffix)
        log("Setup cost", setup_cost, suffix=suffix)
        log("Holding cost", holding_cost, suffix=suffix)
        log("Production rate", production_rate, suffix=suffix)
        
        # Validate production rate > demand
        if production_rate <= demand:
            log(f"{label} (warning)", "Production rate must exceed demand rate", suffix=suffix)
            return None
        
        # Formula: EPQ = sqrt(2*D*A / [h*(1-D/P)])
        epq = math.sqrt(2 * demand * setup_cost / (holding_cost * (1 - demand / production_rate)))
        log(label, epq, unit="units", suffix=suffix)
        
        # Calculate related metrics
        cycle_time = epq / demand
        log("Cycle time", cycle_time, unit="years", suffix=suffix)
        
        production_time = epq / production_rate
        log("Production time", production_time, unit="years", suffix=suffix)
        
        max_inventory = epq * (1 - demand / production_rate)
        log("Maximum inventory", max_inventory, unit="units", suffix=suffix)
        
        avg_inventory = max_inventory / 2
        log("Average inventory", avg_inventory, unit="units", suffix=suffix)
        
        total_cost = setup_cost * demand / epq + holding_cost * avg_inventory
        log("Total annual cost", total_cost, unit="currency", suffix=suffix)
        
        return epq
    
    @staticmethod
    def cost_penalty(actual_quantity, optimal_quantity, label="Cost penalty", suffix=""):
        """
        Calculate cost penalty for deviating from the optimal EOQ
        
        Parameters:
            actual_quantity: Actual order quantity being used
            optimal_quantity: Optimal EOQ
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Cost penalty as a percentage
        """
        log("Actual quantity", actual_quantity, unit="units", suffix=suffix)
        log("Optimal quantity", optimal_quantity, unit="units", suffix=suffix)
        
        # Formula: PCP = 0.5 * [(Q/Q*)+(Q*/Q)-2] * 100%
        penalty = 0.5 * ((actual_quantity / optimal_quantity) + 
                        (optimal_quantity / actual_quantity) - 2) * 100
        
        log(label, penalty, unit="%", suffix=suffix)
        return penalty


# For backward compatibility, expose the class methods as functions
eoq = EOQ.basic
eoq_all_unit_quantity_discount = EOQ.all_unit_quantity_discount
eoq_incremental_quantity_discount = EOQ.incremental_quantity_discount
economic_manufacturing_quantity = EOQ.production_quantity
cost_penalty = EOQ.cost_penalty
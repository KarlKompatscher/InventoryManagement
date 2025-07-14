"""
Inventory Management Formulas - Inventory Policy Module

This module provides a unified framework for inventory policy calculations.
"""

import math
from imf_utils import log, NormalDistribution


class InventoryPolicy:
    """
    Inventory policy calculation framework.
    
    This class provides methods for calculating key inventory policy parameters
    such as safety stock, reorder points, order-up-to levels, and cycle service levels.
    """
    
    @staticmethod
    def safety_stock(sigma, z, lead_time=1, review_period=0, label="Safety stock", suffix=""):
        """
        Calculate safety stock for a given safety factor and lead time
        
        Parameters:
            sigma: Standard deviation of demand per period
            z: Safety factor
            lead_time: Lead time in periods (default: 1)
            review_period: Review period in periods (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Safety stock quantity
        """
        log("Standard deviation", sigma, suffix=suffix)
        log("Safety factor", z, suffix=suffix)
        log("Lead time", lead_time, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        
        ss = z * sigma * math.sqrt(lead_time + review_period)
        log(label, ss, unit="units", suffix=suffix)
        
        return ss
    
    @staticmethod
    def reorder_point(demand_rate, lead_time, safety_stock=0, time_unit="period", 
                    label="Reorder point", suffix=""):
        """
        Calculate reorder point (ROP) for a given demand rate and lead time
        
        Parameters:
            demand_rate: Demand rate per period
            lead_time: Lead time in periods
            safety_stock: Safety stock quantity (default: 0)
            time_unit: Time unit for calculations (default: "period")
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Reorder point quantity
        """
        log("Demand rate", demand_rate, suffix=suffix)
        log("Lead time", lead_time, unit=time_unit, suffix=suffix)
        log("Safety stock", safety_stock, suffix=suffix)
        
        # Calculate lead time demand
        lead_time_demand = demand_rate * lead_time
        log("Lead time demand", lead_time_demand, suffix=suffix)
        
        # Calculate reorder point
        rop = lead_time_demand + safety_stock
        log(label, rop, unit="units", suffix=suffix)
        
        return rop
    
    @staticmethod
    def order_up_to_level(demand_rate, lead_time, safety_stock=0, review_period=0,
                         label="Order-up-to level", suffix=""):
        """
        Calculate order-up-to level (S) for periodic review system
        
        Parameters:
            demand_rate: Demand rate per period
            lead_time: Lead time in periods
            safety_stock: Safety stock quantity (default: 0)
            review_period: Review period in periods (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Order-up-to level quantity
        """
        log("Demand rate", demand_rate, suffix=suffix)
        log("Lead time", lead_time, suffix=suffix)
        log("Safety stock", safety_stock, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        
        # Calculate demand during protection interval (lead time + review period)
        protection_demand = demand_rate * (lead_time + review_period)
        log("Protection interval demand", protection_demand, suffix=suffix)
        
        # Calculate order-up-to level
        s = protection_demand + safety_stock
        log(label, s, unit="units", suffix=suffix)
        
        return s
    
    @staticmethod
    def service_level_safety_stock(mu, sigma, lead_time, service_level, review_period=0,
                                 label="Service level safety stock", suffix=""):
        """
        Calculate safety stock for a given service level
        
        Parameters:
            mu: Mean demand per period
            sigma: Standard deviation of demand per period
            lead_time: Lead time in periods
            service_level: Desired service level (probability)
            review_period: Review period in periods (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with safety stock and reorder point
        """
        log("Mean demand", mu, suffix=suffix)
        log("Standard deviation", sigma, suffix=suffix)
        log("Lead time", lead_time, suffix=suffix)
        log("Service level", service_level, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        
        # Calculate safety factor
        z = NormalDistribution.safety_factor_from_service_level(service_level, label="Safety factor", suffix=suffix)
        
        # Calculate safety stock
        ss = InventoryPolicy.safety_stock(sigma, z, lead_time, review_period, label=label, suffix=suffix)
        
        # Calculate lead time demand
        lead_time_demand = mu * (lead_time + review_period)
        log("Lead time demand", lead_time_demand, suffix=suffix)
        
        # Calculate reorder point
        rop = lead_time_demand + ss
        log("Reorder point", rop, unit="units", suffix=suffix)
        
        return {
            "safety_stock": ss,
            "reorder_point": rop,
            "safety_factor": z
        }
    
    @staticmethod
    def stochastic_lead_time(mu, sigma, mean_lead_time, std_lead_time, service_level,
                           review_period=0, label="Stochastic lead time", suffix=""):
        """
        Calculate safety stock and reorder point with stochastic lead time
        
        Parameters:
            mu: Mean demand per period
            sigma: Standard deviation of demand per period
            mean_lead_time: Mean lead time
            std_lead_time: Standard deviation of lead time
            service_level: Desired service level
            review_period: Review period in periods (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with safety stock and reorder point
        """
        log("Mean demand", mu, suffix=suffix)
        log("Standard deviation", sigma, suffix=suffix)
        log("Mean lead time", mean_lead_time, suffix=suffix)
        log("Std lead time", std_lead_time, suffix=suffix)
        log("Service level", service_level, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        
        # Calculate safety factor
        z = NormalDistribution.safety_factor_from_service_level(service_level, label="Safety factor", suffix=suffix)
        
        # Calculate variance due to demand variation
        var_demand = (mean_lead_time + review_period) * sigma**2
        log("Variance from demand", var_demand, suffix=suffix)
        
        # Calculate variance due to lead time variation
        var_leadtime = mu**2 * std_lead_time**2
        log("Variance from lead time", var_leadtime, suffix=suffix)
        
        # Calculate total standard deviation
        total_std = math.sqrt(var_demand + var_leadtime)
        log("Total standard deviation", total_std, suffix=suffix)
        
        # Calculate safety stock
        ss = z * total_std
        log(f"{label} (safety stock)", ss, unit="units", suffix=suffix)
        
        # Calculate reorder point
        rop = mu * (mean_lead_time + review_period) + ss
        log(f"{label} (reorder point)", rop, unit="units", suffix=suffix)
        
        return {
            "safety_stock": ss,
            "reorder_point": rop,
            "safety_factor": z,
            "total_std_dev": total_std
        }
    
    @staticmethod
    def fill_rate_safety_stock(mu, sigma, lead_time, fill_rate, review_period=0,
                             label="Fill rate safety stock", suffix=""):
        """
        Calculate safety stock to achieve a target fill rate
        
        Parameters:
            mu: Mean demand per period
            sigma: Standard deviation of demand per period
            lead_time: Lead time in periods
            fill_rate: Target fill rate
            review_period: Review period in periods (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with safety stock and reorder point
        """
        log("Mean demand", mu, suffix=suffix)
        log("Standard deviation", sigma, suffix=suffix)
        log("Lead time", lead_time, suffix=suffix)
        log("Target fill rate", fill_rate, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        
        # Calculate safety factor for fill rate
        z = NormalDistribution.safety_factor_from_fill_rate(fill_rate, label="Safety factor", suffix=suffix)
        
        # Calculate standard deviation during lead time + review period
        lt_sigma = sigma * math.sqrt(lead_time + review_period)
        log("Lead time standard deviation", lt_sigma, suffix=suffix)
        
        # Calculate safety stock
        ss = z * lt_sigma
        log(f"{label} (safety stock)", ss, unit="units", suffix=suffix)
        
        # Calculate reorder point
        rop = mu * (lead_time + review_period) + ss
        log(f"{label} (reorder point)", rop, unit="units", suffix=suffix)
        
        # Calculate service level (non-stockout probability)
        service_level = NormalDistribution.service_level_from_safety_factor(z, label="Service level", suffix=suffix)
        
        return {
            "safety_stock": ss,
            "reorder_point": rop,
            "safety_factor": z,
            "service_level": service_level
        }
    
    @staticmethod
    def base_stock_policy(mu, sigma, lead_time, service_level, review_period=0,
                        label="Base stock policy", suffix=""):
        """
        Calculate base stock policy parameters (S) with service level
        
        Parameters:
            mu: Mean demand per period
            sigma: Standard deviation of demand per period
            lead_time: Lead time in periods
            service_level: Target service level
            review_period: Review period in periods (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with base stock level and safety stock
        """
        log("Mean demand", mu, suffix=suffix)
        log("Standard deviation", sigma, suffix=suffix)
        log("Lead time", lead_time, suffix=suffix)
        log("Service level", service_level, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        
        # Calculate safety factor
        z = NormalDistribution.safety_factor_from_service_level(service_level, label="Safety factor", suffix=suffix)
        
        # Calculate standard deviation during lead time + review period
        lt_sigma = sigma * math.sqrt(lead_time + review_period)
        log("Lead time standard deviation", lt_sigma, suffix=suffix)
        
        # Calculate safety stock
        ss = z * lt_sigma
        log("Safety stock", ss, unit="units", suffix=suffix)
        
        # Calculate base stock level
        base_stock = mu * (lead_time + review_period) + ss
        log(f"{label} (base stock level)", base_stock, unit="units", suffix=suffix)
        
        return {
            "base_stock_level": base_stock,
            "safety_stock": ss,
            "safety_factor": z
        }
    
    @staticmethod
    def inventory_policy(demand_params, lead_time_params, service_measure, 
                       target_level, review_period=0, order_quantity=None, system_type="continuous",
                       label="Inventory policy", suffix=""):
        """
        Generalized inventory policy calculation
        
        Parameters:
            demand_params: Dictionary with demand parameters (mu, sigma)
            lead_time_params: Dictionary with lead time parameters (mean, std)
            service_measure: Service measure type ('cycle_service_level', 'fill_rate')
            target_level: Target level for the selected service measure
            review_period: Review period in periods (default: 0)
            order_quantity: Order quantity for (s,Q) policy (default: None)
            system_type: Inventory system type ('continuous', 'periodic')
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with inventory policy parameters
        """
        # Extract demand parameters
        mu = demand_params.get('mu', 0)
        sigma = demand_params.get('sigma', 0)
        
        log("Mean demand", mu, suffix=suffix)
        log("Demand std dev", sigma, suffix=suffix)
        
        # Extract lead time parameters
        mean_lt = lead_time_params.get('mean', 0)
        std_lt = lead_time_params.get('std', 0)
        
        log("Mean lead time", mean_lt, suffix=suffix)
        log("Lead time std dev", std_lt, suffix=suffix)
        log("Review period", review_period, suffix=suffix)
        log(f"Service measure ({service_measure})", target_level, suffix=suffix)
        
        # Calculate safety factor based on service measure
        if service_measure == 'cycle_service_level':
            z = NormalDistribution.safety_factor_from_service_level(target_level, label="Safety factor", suffix=suffix)
        elif service_measure == 'fill_rate':
            z = NormalDistribution.safety_factor_from_fill_rate(target_level, label="Safety factor", suffix=suffix)
        else:
            raise ValueError(f"Unsupported service measure: {service_measure}")
        
        # Calculate standard deviation accounting for lead time variation
        if std_lt > 0:
            # Stochastic lead time
            var_demand = (mean_lt + review_period) * sigma**2
            var_leadtime = mu**2 * std_lt**2
            total_std = math.sqrt(var_demand + var_leadtime)
        else:
            # Deterministic lead time
            total_std = sigma * math.sqrt(mean_lt + review_period)
            
        log("Total standard deviation", total_std, suffix=suffix)
        
        # Calculate safety stock
        ss = z * total_std
        log("Safety stock", ss, unit="units", suffix=suffix)
        
        # Calculate expected demand during protection interval
        expected_demand = mu * (mean_lt + review_period)
        log("Expected demand during protection", expected_demand, suffix=suffix)
        
        # Calculate policy parameters based on system type
        if system_type == 'continuous':
            # (s,Q) policy
            reorder_point = expected_demand + ss
            log("Reorder point (s)", reorder_point, unit="units", suffix=suffix)
            
            result = {
                "policy_type": "(s,Q)",
                "reorder_point": reorder_point,
                "safety_stock": ss,
                "safety_factor": z
            }
            
            # Add order quantity if provided
            if order_quantity is not None:
                result["order_quantity"] = order_quantity
                
        else:  # periodic
            # (R,S) policy
            order_up_to = expected_demand + ss
            log("Order-up-to level (S)", order_up_to, unit="units", suffix=suffix)
            
            result = {
                "policy_type": "(R,S)",
                "review_period": review_period,
                "order_up_to_level": order_up_to,
                "safety_stock": ss,
                "safety_factor": z
            }
        
        return result


# For backward compatibility, expose the class methods as functions
safety_stock = InventoryPolicy.safety_stock
reorder_point = InventoryPolicy.reorder_point
order_up_to_level = InventoryPolicy.order_up_to_level
service_level_safety_stock = InventoryPolicy.service_level_safety_stock
variable_lead_time_safety_stock = InventoryPolicy.stochastic_lead_time
fill_rate_safety_stock = InventoryPolicy.fill_rate_safety_stock
base_stock_policy = InventoryPolicy.base_stock_policy
inventory_policy = InventoryPolicy.inventory_policy
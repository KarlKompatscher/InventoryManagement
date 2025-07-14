"""
Inventory Management Formulas - Statistical Utility Module

This module provides statistical distribution utility functions for inventory management calculations.
"""

import math
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar


# Basic logging utility
def log(label, value, unit="", suffix=""):
    """
    Simple logging utility for labeled output.

    Parameters:
        label: Description of the value
        value: The value to log
        unit: Optional unit of measurement
        suffix: Optional suffix to differentiate scenarios (A/B, 1/2, etc.)

    Returns:
        None
    """
    if suffix:
        suffix = f" ({suffix})"
    
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) > 1000:
            value_str = f"{value:.2e}"
        else:
            value_str = f"{value:.2f}"
    else:
        value_str = str(value)
        
    if unit:
        unit = f" {unit}"
        
    print(f"[INFO] {label}{suffix}: {value_str}{unit}")


class NormalDistribution:
    """
    Utility class for normal distribution calculations commonly used in inventory management.
    """
    
    @staticmethod
    def pdf(z, label="Normal PDF", suffix=""):
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
    
    @staticmethod
    def cdf(z, label="Normal CDF", suffix=""):
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
    
    @staticmethod
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
        if p <= 0:
            log(f"{label} (warning)", f"Invalid probability value {p}, using 0.001", suffix=suffix)
            p = 0.001
        elif p >= 1:
            log(f"{label} (warning)", f"Invalid probability value {p}, using 0.999", suffix=suffix)
            p = 0.999
            
        result = norm.ppf(p)
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def loss_function(z, label="Loss function G(z)", suffix=""):
        """
        Standard normal loss function G(z) = φ(z) - z·(1-Φ(z))
        
        Parameters:
            z: Standard normal quantile value
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Value of the loss function G(z)
        """
        # Calculate directly without calling separate functions for efficiency
        result = norm.pdf(z) - z * (1 - norm.cdf(z))
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def inverse_loss_function(G_target, min=-1000, max=1000, label="Inverse G(z)", suffix=""):
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
        # Use a local function to avoid logging during root finding
        def g_z(z):
            return norm.pdf(z) - z * (1 - norm.cdf(z)) - G_target
        
        result = brentq(g_z, min, max)
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def safety_factor_from_service_level(service_level, label="Safety factor z", suffix=""):
        """
        Calculate safety factor z corresponding to a given service level
        
        Parameters:
            service_level: Target service level (probability of no stockout)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Safety factor z
        """
        return NormalDistribution.inverse_cdf(service_level, label=label, suffix=suffix)
    
    @staticmethod
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
        return NormalDistribution.cdf(z, label=label, suffix=suffix)
    
    @staticmethod
    def fill_rate_from_safety_factor(z, label="Fill rate", suffix=""):
        """
        Calculate fill rate given a safety factor z, assuming normally distributed demand
        
        Parameters:
            z: Safety factor
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Fill rate (fraction of demand satisfied)
        """
        if z <= 0:
            result = 0
        else:
            g_z = NormalDistribution.loss_function(z, label="", suffix="")
            result = 1 - g_z / z
            
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def safety_factor_from_fill_rate(fill_rate, label="Safety factor z", suffix=""):
        """
        Calculate safety factor z for a given fill rate, assuming normally distributed demand
        
        Parameters:
            fill_rate: Target fill rate (0-1)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Safety factor z
        """
        if fill_rate >= 1:
            log(f"{label} (warning)", f"Invalid fill rate {fill_rate}, using 0.9999", suffix=suffix)
            fill_rate = 0.9999
        elif fill_rate <= 0:
            log(f"{label} (warning)", f"Invalid fill rate {fill_rate}, using 0.0001", suffix=suffix)
            fill_rate = 0.0001
            
        # Define the objective function
        def objective(z):
            if z <= 0:
                return float('inf')
            g_z = norm.pdf(z) - z * (1 - norm.cdf(z))
            return abs(1 - g_z / z - fill_rate)
        
        # Solve using minimize_scalar
        result = minimize_scalar(objective, method='golden', bounds=(0.01, 10)).x
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def expected_shortage(sigma, z, label="Expected shortage", suffix=""):
        """
        Calculate expected shortage for a given safety factor z and normal distribution
        
        Parameters:
            sigma: Standard deviation of demand
            z: Safety factor
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Expected shortage quantity
        """
        g_z = NormalDistribution.loss_function(z, label="", suffix="")
        result = sigma * g_z
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def convert_mad_to_stdev(mad, label="Standard deviation", suffix=""):
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
    
    @staticmethod
    def convert_stdev_to_mad(sigma, label="MAD", suffix=""):
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
    
    @staticmethod
    def standardize(x, mu, sigma, label="Standardized value", suffix=""):
        """
        Standardize a value to z-score: z = (x - μ) / σ
        
        Parameters:
            x: Value to standardize
            mu: Mean of the distribution
            sigma: Standard deviation of the distribution
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Standardized value (z-score)
        """
        result = (x - mu) / sigma
        log(label, result, suffix=suffix)
        return result
    
    @staticmethod
    def unstandardize(z, mu, sigma, label="Unstandardized value", suffix=""):
        """
        Convert z-score back to original scale: x = μ + z·σ
        
        Parameters:
            z: Standardized value (z-score)
            mu: Mean of the distribution
            sigma: Standard deviation of the distribution
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Value in original scale
        """
        result = mu + z * sigma
        log(label, result, suffix=suffix)
        return result


class InventoryUtils:
    """
    Utility class for common inventory calculations.
    """
    
    @staticmethod
    def calculate_service_measures(mean_demand, std_demand, reorder_point=None, safety_stock=None, 
                                 lead_time=1, order_quantity=None, safety_factor=None):
        """
        Calculate various service measures for an inventory system
        
        Parameters:
            mean_demand: Mean demand per period
            std_demand: Standard deviation of demand per period
            reorder_point: Reorder point (s)
            safety_stock: Safety stock level
            lead_time: Lead time in periods
            order_quantity: Order quantity (Q) - needed for fill rate calculation
            safety_factor: Safety factor (z) - alternative to providing reorder_point
            
        Returns:
            Dictionary with service measures
        """
        # Calculate lead time demand parameters
        lt_mean = mean_demand * lead_time
        lt_std = std_demand * math.sqrt(lead_time)
        
        # Determine safety factor z
        if safety_factor is not None:
            z = safety_factor
        elif reorder_point is not None:
            z = (reorder_point - lt_mean) / lt_std
        elif safety_stock is not None:
            z = safety_stock / lt_std
        else:
            raise ValueError("Either reorder_point, safety_stock, or safety_factor must be provided")
        
        # Calculate service level (non-stockout probability)
        service_level = NormalDistribution.service_level_from_safety_factor(z, label="")
        
        # Calculate expected shortage
        expected_short = NormalDistribution.expected_shortage(lt_mean, lt_std, z, label="")
        
        # Calculate fill rate if order quantity is provided
        fill_rate = None
        if order_quantity is not None:
            fill_rate = 1 - expected_short / order_quantity
        
        return {
            "safety_factor": z,
            "service_level": service_level,
            "expected_shortage": expected_short,
            "fill_rate": fill_rate
        }


# For backward compatibility, expose the class methods as functions
phi_z = NormalDistribution.pdf
Phi_z = NormalDistribution.cdf
G_z = NormalDistribution.loss_function
inverse_G = NormalDistribution.inverse_loss_function
inverse_cdf = NormalDistribution.inverse_cdf
safety_factor_from_service_level = NormalDistribution.safety_factor_from_service_level
service_level_from_safety_factor = NormalDistribution.service_level_from_safety_factor
fill_rate_from_safety_factor = NormalDistribution.fill_rate_from_safety_factor
expected_shortage = NormalDistribution.expected_shortage
mad_to_stdev = NormalDistribution.convert_mad_to_stdev
stdev_to_mad = NormalDistribution.convert_stdev_to_mad
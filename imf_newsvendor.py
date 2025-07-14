"""
Inventory Management Formulas - Newsvendor Module

This module provides a unified framework for newsvendor calculations with different distributions.
"""

import math
from scipy.stats import poisson, gamma, t
from imf_utils import log, NormalDistribution


class Newsvendor:
    """
    Newsvendor model calculation framework.
    
    This class provides methods for calculating optimal order quantities 
    for the newsvendor problem with various demand distributions.
    """
    
    @staticmethod
    def critical_ratio(price, cost, salvage=0, label="Critical ratio", suffix=""):
        """
        Calculate newsvendor critical ratio (profit-maximizing service level)
        
        Parameters:
            price: Selling price per unit
            cost: Cost per unit
            salvage: Salvage value per unit (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Critical ratio (optimal service level)
        """
        log("Selling price", price, suffix=suffix)
        log("Unit cost", cost, suffix=suffix)
        log("Salvage value", salvage, suffix=suffix)
        
        if price <= cost:
            log(f"{label} (warning)", "Price must exceed cost", suffix=suffix)
            return 0
        
        if salvage > price:
            log(f"{label} (warning)", "Salvage value should not exceed price", suffix=suffix)
            salvage = price
            
        # Formula: CR = (p - c) / (p - g)
        cr = (price - cost) / (price - salvage)
        log(label, cr, suffix=suffix)
        
        return cr
    
    @staticmethod
    def critical_fractile(co, cu, label="Critical fractile", suffix=""):
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
        
        # Formula: CF = cu / (co + cu)
        cf = cu / (co + cu)
        log(label, cf, suffix=suffix)
        
        return cf
    
    @staticmethod
    def normal(mean, std, critical_ratio, label="Newsvendor normal", suffix=""):
        """
        Calculate newsvendor order quantity for normally distributed demand
        
        Parameters:
            mean: Mean demand
            std: Standard deviation of demand
            critical_ratio: Critical ratio or critical fractile
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Optimal order quantity
        """
        log("Mean demand", mean, suffix=suffix)
        log("Standard deviation", std, suffix=suffix)
        log("Critical ratio", critical_ratio, suffix=suffix)
        
        # Calculate optimal z-value
        z = NormalDistribution.inverse_cdf(critical_ratio, label="z-value", suffix=suffix)
        
        # Calculate optimal order quantity
        optimal_q = mean + z * std
        log(label, optimal_q, unit="units", suffix=suffix)
        
        # Calculate expected profit components
        expected_shortage = NormalDistribution.expected_shortage(std, z, label="Expected shortage", suffix=suffix)
        expected_overage = expected_shortage + (optimal_q - mean)
        log("Expected overage", expected_overage, suffix=suffix)
        
        return optimal_q
    
    @staticmethod
    def with_estimated_params(sample_mean, sample_std, sample_size, price, cost, 
                            salvage=0, label="Newsvendor with estimated parameters", suffix=""):
        """
        Calculate newsvendor order quantity with parameter uncertainty using t-distribution
        
        Parameters:
            sample_mean: Sample mean of demand
            sample_std: Sample standard deviation of demand
            sample_size: Sample size used for estimation
            price: Selling price per unit
            cost: Cost per unit
            salvage: Salvage value per unit (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Optimal order quantity accounting for parameter uncertainty
        """
        log("Sample mean", sample_mean, suffix=suffix)
        log("Sample std dev", sample_std, suffix=suffix)
        log("Sample size", sample_size, suffix=suffix)
        log("Selling price", price, suffix=suffix)
        log("Unit cost", cost, suffix=suffix)
        log("Salvage value", salvage, suffix=suffix)
        
        # Calculate critical ratio
        cr = Newsvendor.critical_ratio(price, cost, salvage, label=f"{label} (critical ratio)", suffix=suffix)
        
        # Calculate t-value for given critical ratio and degrees of freedom
        degrees_of_freedom = sample_size - 1
        t_value = t.ppf(cr, df=degrees_of_freedom)
        log(f"{label} (t-value)", t_value, suffix=suffix)
        
        # Calculate correction factor
        correction_factor = math.sqrt(1 + 1/sample_size)
        log(f"{label} (correction factor)", correction_factor, suffix=suffix)
        
        # Calculate optimal order quantity
        optimal_q = sample_mean + t_value * sample_std * correction_factor
        log(label, optimal_q, unit="units", suffix=suffix)
        
        # Compare with standard normal solution
        standard_q = Newsvendor.normal(sample_mean, sample_std, cr, 
                                      label=f"{label} (without correction)", suffix=suffix)
        
        # Calculate difference
        difference = optimal_q - standard_q
        percent_difference = (difference / standard_q) * 100 if standard_q != 0 else 0
        
        log(f"{label} (absolute difference)", difference, unit="units", suffix=suffix)
        log(f"{label} (percentage difference)", percent_difference, unit="%", suffix=suffix)
        
        return optimal_q
    
    @staticmethod
    def uniform(lower_bound, upper_bound, critical_ratio, label="Newsvendor uniform", suffix=""):
        """
        Calculate newsvendor order quantity for uniformly distributed demand
        
        Parameters:
            lower_bound: Lower bound of the uniform distribution
            upper_bound: Upper bound of the uniform distribution
            critical_ratio: Critical ratio or critical fractile
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Optimal order quantity
        """
        log("Lower bound", lower_bound, suffix=suffix)
        log("Upper bound", upper_bound, suffix=suffix)
        log("Critical ratio", critical_ratio, suffix=suffix)
        
        # Calculate optimal order quantity
        scale = upper_bound - lower_bound
        optimal_q = lower_bound + critical_ratio * scale
        log(label, optimal_q, unit="units", suffix=suffix)
        
        return optimal_q
    
    @staticmethod
    def poisson(lambda_param, critical_ratio, label="Newsvendor Poisson", suffix=""):
        """
        Calculate newsvendor order quantity for Poisson distributed demand
        
        Parameters:
            lambda_param: Lambda parameter (mean) of the Poisson distribution
            critical_ratio: Critical ratio or critical fractile
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Optimal order quantity
        """
        log("Lambda parameter", lambda_param, suffix=suffix)
        log("Critical ratio", critical_ratio, suffix=suffix)
        
        # Calculate optimal order quantity using the inverse CDF
        optimal_q = poisson.ppf(critical_ratio, lambda_param)
        log(label, optimal_q, unit="units", suffix=suffix)
        
        return optimal_q
    
    @staticmethod
    def gamma(mean, std, critical_ratio, label="Newsvendor gamma", suffix=""):
        """
        Calculate newsvendor order quantity for gamma distributed demand
        
        Parameters:
            mean: Mean demand
            std: Standard deviation of demand
            critical_ratio: Critical ratio or critical fractile
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Optimal order quantity
        """
        log("Mean", mean, suffix=suffix)
        log("Standard deviation", std, suffix=suffix)
        log("Critical ratio", critical_ratio, suffix=suffix)
        
        # Calculate gamma parameters
        var = std**2
        alpha = mean**2 / var  # shape parameter
        theta = var / mean     # scale parameter
        
        log("Gamma alpha", alpha, suffix=suffix)
        log("Gamma theta", theta, suffix=suffix)
        
        # Calculate optimal order quantity
        optimal_q = gamma.ppf(critical_ratio, a=alpha, scale=theta)
        log(label, optimal_q, unit="units", suffix=suffix)
        
        return optimal_q
    
    @staticmethod
    def general(distribution, params, price, cost, salvage=0, label="Newsvendor general", suffix=""):
        """
        Generalized newsvendor calculation for different distributions
        
        Parameters:
            distribution: Distribution type ('normal', 'uniform', 'poisson', 'gamma')
            params: Dictionary of distribution parameters
            price: Selling price per unit
            cost: Cost per unit
            salvage: Salvage value per unit (default: 0)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with optimal order quantity and metrics
        """
        log("Distribution", distribution, suffix=suffix)
        log("Distribution parameters", params, suffix=suffix)
        log("Selling price", price, suffix=suffix)
        log("Unit cost", cost, suffix=suffix)
        log("Salvage value", salvage, suffix=suffix)
        
        # Calculate critical ratio
        cr = Newsvendor.critical_ratio(price, cost, salvage, label=f"{label} (critical ratio)", suffix=suffix)
        
        # Calculate optimal order quantity based on distribution
        if distribution == 'normal':
            optimal_q = Newsvendor.normal(params['mean'], params['std'], cr, 
                                        label=f"{label} (optimal Q)", suffix=suffix)
            
            # Calculate z-value
            z = (optimal_q - params['mean']) / params['std']
            
            # Calculate expected shortage and other metrics
            expected_shortage = params['std'] * NormalDistribution.loss_function(z, label="", suffix="")
            expected_leftover = optimal_q - params['mean'] + expected_shortage
            expected_sales = params['mean'] - expected_shortage
            stockout_prob = 1 - NormalDistribution.cdf(z, label="", suffix="")
            fill_rate = 1 - expected_shortage / params['mean']
            
        elif distribution == 'uniform':
            optimal_q = Newsvendor.uniform(params['lower'], params['upper'], cr, 
                                         label=f"{label} (optimal Q)", suffix=suffix)
            
            # Calculate expected shortage and other metrics
            if optimal_q >= params['upper']:
                expected_shortage = 0
                expected_sales = params['mean']
                stockout_prob = 0
                fill_rate = 1
            else:
                expected_shortage = ((params['upper'] - optimal_q) ** 2) / (2 * (params['upper'] - params['lower']))
                expected_sales = params['mean'] - expected_shortage
                stockout_prob = (params['upper'] - optimal_q) / (params['upper'] - params['lower'])
                fill_rate = expected_sales / params['mean']
                
            expected_leftover = optimal_q - expected_sales
            
        elif distribution == 'poisson':
            optimal_q = Newsvendor.poisson(params['lambda'], cr, 
                                         label=f"{label} (optimal Q)", suffix=suffix)
            
            # Calculate expected shortage and other metrics
            expected_shortage = 0
            for d in range(int(optimal_q) + 1, int(params['lambda'] * 3) + 1):  # Use 3*lambda as upper bound
                expected_shortage += (d - optimal_q) * poisson.pmf(d, params['lambda'])
                
            expected_sales = params['lambda'] - expected_shortage
            expected_leftover = optimal_q - expected_sales
            stockout_prob = 1 - poisson.cdf(optimal_q - 1, params['lambda'])
            fill_rate = expected_sales / params['lambda']
            
        elif distribution == 'gamma':
            # Calculate gamma parameters
            var = params['std']**2
            alpha = params['mean']**2 / var
            theta = var / params['mean']
            
            optimal_q = Newsvendor.gamma(params['mean'], params['std'], cr, 
                                       label=f"{label} (optimal Q)", suffix=suffix)
            
            # Calculate expected shortage (numerical approximation)
            expected_shortage = 0
            upper_bound = params['mean'] + 4 * params['std']  # Use mean + 4*std as upper bound
            step = params['std'] / 20
            
            x = optimal_q
            while x <= upper_bound:
                expected_shortage += step * (x - optimal_q) * gamma.pdf(x, a=alpha, scale=theta)
                x += step
                
            expected_sales = params['mean'] - expected_shortage
            expected_leftover = optimal_q - expected_sales
            stockout_prob = 1 - gamma.cdf(optimal_q, a=alpha, scale=theta)
            fill_rate = expected_sales / params['mean']
            
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        # Calculate expected profit
        expected_profit = price * expected_sales + salvage * expected_leftover - cost * optimal_q
        
        # Log results
        log(f"{label} (optimal order quantity)", optimal_q, unit="units", suffix=suffix)
        log(f"{label} (expected profit)", expected_profit, suffix=suffix)
        log(f"{label} (expected sales)", expected_sales, unit="units", suffix=suffix)
        log(f"{label} (expected leftover)", expected_leftover, unit="units", suffix=suffix)
        log(f"{label} (expected shortage)", expected_shortage, unit="units", suffix=suffix)
        log(f"{label} (stockout probability)", stockout_prob, suffix=suffix)
        log(f"{label} (fill rate)", fill_rate, suffix=suffix)
        
        return {
            "optimal_order_quantity": optimal_q,
            "critical_ratio": cr,
            "expected_profit": expected_profit,
            "expected_sales": expected_sales,
            "expected_leftover": expected_leftover,
            "expected_shortage": expected_shortage,
            "stockout_probability": stockout_prob,
            "fill_rate": fill_rate
        }
    
    @staticmethod
    def with_costs(unit_cost, price, holding_cost, stockout_cost, mean=None, std=None,
                 continuous=True, label="Newsvendor with costs", suffix=""):
        """
        Calculate newsvendor solution with explicit holding and stockout costs
        
        Parameters:
            unit_cost: Cost per unit
            price: Selling price per unit
            holding_cost: Holding cost per unit per period
            stockout_cost: Stockout cost per unit
            mean: Mean demand (for normal distribution)
            std: Standard deviation of demand (for normal distribution)
            distribution: Distribution type (for non-normal distributions)
            distribution_params: Parameters for non-normal distributions
            continuous: Whether to treat demand as continuous (True) or discrete (False)
            label: Optional label for logging output
            suffix: Optional suffix to differentiate scenarios
            
        Returns:
            Dictionary with optimal order quantity and metrics
        """
        log("Unit cost", unit_cost, suffix=suffix)
        log("Selling price", price, suffix=suffix)
        log("Holding cost", holding_cost, suffix=suffix)
        log("Stockout cost", stockout_cost, suffix=suffix)
        
        # Calculate overage and underage costs
        overage_cost = unit_cost - 0  # Assuming zero salvage value
        underage_cost = price - unit_cost + stockout_cost
        
        log("Overage cost", overage_cost, suffix=suffix)
        log("Underage cost", underage_cost, suffix=suffix)
        
        # Calculate critical fractile
        critical_fractile = Newsvendor.critical_fractile(
            overage_cost, underage_cost, 
            label=f"{label} (critical fractile)", 
            suffix=suffix
        )
        
        # Handle different distribution types
        if mean is not None and std is not None:
            # Normal distribution
            if continuous:
                optimal_q = Newsvendor.normal(
                    mean, std, critical_fractile,
                    label=f"{label} (optimal Q)",
                    suffix=suffix
                )
                
                # Calculate expected costs using normal distribution
                z = (optimal_q - mean) / std
                expected_shortage = std * NormalDistribution.loss_function(z, label="", suffix="")
                expected_leftover = optimal_q - mean + expected_shortage
                
                # Calculate expected costs
                expected_holding_cost = holding_cost * expected_leftover
                expected_stockout_cost = stockout_cost * expected_shortage
                expected_total_cost = expected_holding_cost + expected_stockout_cost
                
                log(f"{label} (expected holding cost)", expected_holding_cost, suffix=suffix)
                log(f"{label} (expected stockout cost)", expected_stockout_cost, suffix=suffix)
                log(f"{label} (expected total cost)", expected_total_cost, suffix=suffix)
                
                return {
                    "optimal_order_quantity": optimal_q,
                    "critical_fractile": critical_fractile,
                    "expected_holding_cost": expected_holding_cost,
                    "expected_stockout_cost": expected_stockout_cost,
                    "expected_total_cost": expected_total_cost
                }
            else:
                # Discrete normal approximation
                # Find the smallest integer q such that Φ((q-0.5-μ)/σ) ≥ critical_fractile
                q = mean - 0.5  # Start at mean - 0.5
                while NormalDistribution.cdf((q - 0.5 - mean) / std, label="", suffix="") < critical_fractile:
                    q += 1
                    
                optimal_q = q
                log(f"{label} (optimal Q - discrete)", optimal_q, unit="units", suffix=suffix)
                
                # Calculate expected costs for discrete normal
                expected_holding_cost = 0
                expected_stockout_cost = 0
                
                # Use normal approximation for calculations
                for d in range(int(mean + 4*std) + 1):
                    prob_d = NormalDistribution.pdf((d - mean) / std, label="", suffix="") / std
                    
                    if d < optimal_q:
                        # Holding cost for excess inventory
                        expected_holding_cost += holding_cost * (optimal_q - d) * prob_d
                    else:
                        # Stockout cost for shortage
                        expected_stockout_cost += stockout_cost * (d - optimal_q) * prob_d
                
                expected_total_cost = expected_holding_cost + expected_stockout_cost
                
                log(f"{label} (expected holding cost)", expected_holding_cost, suffix=suffix)
                log(f"{label} (expected stockout cost)", expected_stockout_cost, suffix=suffix)
                log(f"{label} (expected total cost)", expected_total_cost, suffix=suffix)
                
                return {
                    "optimal_order_quantity": optimal_q,
                    "critical_fractile": critical_fractile,
                    "expected_holding_cost": expected_holding_cost,
                    "expected_stockout_cost": expected_stockout_cost,
                    "expected_total_cost": expected_total_cost
                }
        else:
            raise ValueError("Mean and standard deviation must be provided")


# For backward compatibility, expose the class methods as functions
newsvendor_critical_ratio = Newsvendor.critical_ratio
newsvendor_critical_fractile = Newsvendor.critical_fractile
newsvendor_normal = Newsvendor.normal
newsvendor_with_estimated_params = Newsvendor.with_estimated_params
newsvendor_uniform = Newsvendor.uniform
newsvendor_poisson = Newsvendor.poisson
newsvendor_gamma = Newsvendor.gamma
newsvendor_general = Newsvendor.general
newsvendor_with_costs = Newsvendor.with_costs
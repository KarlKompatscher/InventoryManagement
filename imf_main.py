"""
Inventory Management Formulas - Main Module

This module provides a unified interface to all inventory management formulas
by importing and exposing functions from the specialized modules.
"""

# Import utility functions
from imf_utils import (
    log,
    NormalDistribution,
    InventoryUtils,
    # Backward compatibility for frequently used functions
    phi_z,
    Phi_z,
    G_z,
    inverse_G,
    inverse_cdf,
    safety_factor_from_service_level,
    service_level_from_safety_factor,
    fill_rate_from_safety_factor,
    expected_shortage,
    mad_to_stdev,
    stdev_to_mad
)

# Import EOQ functions
from imf_eoq import (
    EOQ,
    # Backward compatibility functions
    eoq,
    eoq_all_unit_quantity_discount,
    eoq_incremental_quantity_discount,
    economic_manufacturing_quantity,
    cost_penalty
)

# Import Newsvendor functions
from imf_newsvendor import (
    Newsvendor,
    # Backward compatibility functions
    newsvendor_critical_ratio,
    newsvendor_critical_fractile,
    newsvendor_normal,
    newsvendor_with_estimated_params,
    newsvendor_uniform,
    newsvendor_poisson,
    newsvendor_gamma,
    newsvendor_general,
    newsvendor_with_costs
)

# Import Inventory Policy functions
from imf_inventory import (
    InventoryPolicy,
    # Backward compatibility functions
    safety_stock,
    reorder_point,
    order_up_to_level,
    service_level_safety_stock,
    variable_lead_time_safety_stock,
    fill_rate_safety_stock,
    base_stock_policy,
    inventory_policy
)

# Version information
__version__ = "2.0.0"
__author__ = "Inventory Management Team"
__description__ = "Comprehensive inventory management formula library"


def version_info():
    """Return library version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__
    }


# Define a dictionary of all available functions for documentation purposes
available_functions = {
    "Statistical Distribution Utilities": [
        "phi_z - Standard normal PDF",
        "Phi_z - Standard normal CDF",
        "G_z - Normal loss function",
        "inverse_G - Inverse normal loss function",
        "inverse_cdf - Inverse standard normal CDF",
        "safety_factor_from_service_level - Convert service level to safety factor",
        "service_level_from_safety_factor - Convert safety factor to service level",
        "fill_rate_from_safety_factor - Calculate fill rate from safety factor",
        "expected_shortage - Calculate expected shortage",
        "mad_to_stdev - Convert MAD to standard deviation",
        "stdev_to_mad - Convert standard deviation to MAD"
    ],
    
    "EOQ Models": [
        "eoq - Basic Economic Order Quantity",
        "eoq_all_unit_quantity_discount - EOQ with all-unit quantity discount",
        "eoq_incremental_quantity_discount - EOQ with incremental quantity discount",
        "economic_manufacturing_quantity - Economic Production Quantity",
        "cost_penalty - Calculate cost penalty for non-optimal order quantity"
    ],
    
    "Newsvendor Models": [
        "newsvendor_critical_ratio - Calculate critical ratio",
        "newsvendor_critical_fractile - Calculate critical fractile",
        "newsvendor_normal - Newsvendor with normal distribution",
        "newsvendor_with_estimated_params - Newsvendor with estimated parameters",
        "newsvendor_uniform - Newsvendor with uniform distribution",
        "newsvendor_poisson - Newsvendor with Poisson distribution",
        "newsvendor_gamma - Newsvendor with gamma distribution",
        "newsvendor_general - Generalized newsvendor calculation",
        "newsvendor_with_costs - Newsvendor with explicit costs"
    ],
    
    "Inventory Policy": [
        "safety_stock - Calculate safety stock",
        "reorder_point - Calculate reorder point",
        "order_up_to_level - Calculate order-up-to level",
        "service_level_safety_stock - Safety stock for a given service level",
        "variable_lead_time_safety_stock - Safety stock with variable lead time",
        "fill_rate_safety_stock - Safety stock for a given fill rate",
        "base_stock_policy - Calculate base stock policy parameters",
        "inventory_policy - Generalized inventory policy calculation"
    ]
}


# Function to list all available functions in a category
def list_functions(category=None):
    """
    List all available functions in the library, optionally filtered by category.
    
    Parameters:
        category: Optional category to filter by
        
    Returns:
        List of function descriptions
    """
    if category is not None and category in available_functions:
        return available_functions[category]
    elif category is not None:
        return f"Category '{category}' not found. Available categories: {list(available_functions.keys())}"
    else:
        result = []
        for category, functions in available_functions.items():
            result.append(f"\n{category}:")
            result.extend([f"  - {func}" for func in functions])
        return "\n".join(result)


# If this file is run directly, print version information
if __name__ == "__main__":
    version = version_info()
    print(f"{version['description']} v{version['version']}")
    print(f"Author: {version['author']}")
    print("\nAvailable function categories:")
    for category in available_functions:
        print(f"- {category}")
    print("\nUse list_functions() to see all available functions")
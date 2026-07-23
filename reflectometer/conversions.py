# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 16:26:10 2025

@author: aszta
"""


def from_unit_to_centi(value):
    """
    Converts from unit to centi.
    
    Args:
        value (float or numpy.ndarray): Wave frequency in Hz
        
    Returns:
        float or numpy.ndarray: values from unit to centi by x 100
        
    """
    return value * 100

def from_centi_to_unit(value):
    """
    Converts from centi to unit.
    
    Args:
        value (float or numpy.ndarray): Wave frequency in Hz
        
    Returns:
        float or numpy.ndarray: values from centi to unit by x 0.01
        
    """
    return value * 0.01

def antenna_pos_to_unit(antenna_pos, y0, ny, dx):
    """
    Convert a physical antenna Y-position (in meters) to a grid row index,
    using the same convention as the Gaussian wave source.
    """
    return int(ny - (antenna_pos - y0) // dx)

def meter_to_unit(distance_m, dx):
    """
    Convert a physical distance (in meters) to a grid row index,
    """
    return int(distance_m // dx)
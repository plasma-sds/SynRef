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
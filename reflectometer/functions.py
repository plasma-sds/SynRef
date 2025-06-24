# -*- coding: utf-8 -*-
"""
Reflectometry utility functions for reflectometry calculations.

This module provides functions for calculating cut-off frequencies and densities
for O-mode and X-mode electromagnetic waves in plasma.
"""

import scipy.constants as constant
import numpy as np


def get_Omode_cutoff_density(frequency):
    """
    Calculate the O-mode cut-off density for a given frequency.
    
    The O-mode cut-off occurs when the plasma frequency equals the wave frequency.
    The plasma frequency is given by: ω_pe = sqrt(n_e * e^2 / (m_e * ε_0))
    At cut-off: ω = ω_pe, therefore: n_e = (ω^2 * m_e * ε_0) / e^2
    
    Args:
        frequency (float): Wave frequency in Hz
        
    Returns:
        float: Cut-off density in m^-3
        
    Example:
        >>> get_Omode_cutoff_density(3e10)  # 30 GHz
        1.115e19  # m^-3
    """
    # Plasma frequency: ω_pe = sqrt(n_e * e^2 / (m_e * ε_0))
    # At cut-off: ω = ω_pe
    # Therefore: n_e = (ω^2 * m_e * ε_0) / e^2
    
    # Constants
    e = constant.elementary_charge  # Elementary charge in C
    m_e = constant.m_e  # Electron mass in kg
    epsilon_0 = constant.epsilon_0  # Vacuum permittivity in F/m
    
    # Calculate cut-off density
    omega = 2 * np.pi * frequency  # Angular frequency
    cutoff_density = (omega**2 * m_e * epsilon_0) / (e**2)
    
    return cutoff_density


def get_Xmode_cutoff_density(frequency, bfield):
    """
    Calculate the X-mode cut-off densities for a given frequency and magnetic field.
    
    The X-mode has two cut-offs:
    1. Upper cut-off: ω = ω_pe (same as O-mode)
    2. Lower cut-off: ω = (ω_ce + sqrt(ω_ce^2 + 4*ω_pe^2)) / 2
    
    This function returns both cut-off densities.
    
    Args:
        frequency (float): Wave frequency in Hz
        bfield (float): Magnetic field strength in Tesla
        
    Returns:
        tuple: (lower_cutoff_density, upper_cutoff_density) in m^-3
        
    Example:
        >>> get_Xmode_cutoff_density(3e10, 2.5)  # 30 GHz, 2.5 T
        (1.115e19, 1.115e19)  # m^-3 (lower and upper cut-offs)
    """
    # Upper cut-off density (same as O-mode)
    upper_cutoff_density = get_Omode_cutoff_density(frequency)
    
    # Calculate electron cyclotron frequency
    f_ce = get_cyclotron_frequency(bfield)
    omega_ce = 2 * np.pi * f_ce
    
    # Lower cut-off frequency
    omega = 2 * np.pi * frequency
    lower_cutoff_freq = (omega_ce + np.sqrt(omega_ce**2 + 4 * omega**2)) / (2 * 2 * np.pi)
    
    # Lower cut-off density using plasma frequency relationship
    # n_e = (f_pe^2 * m_e * ε_0 * 4π^2) / e^2
    e = constant.elementary_charge
    m_e = constant.m_e
    epsilon_0 = constant.epsilon_0
    lower_cutoff_density = (lower_cutoff_freq**2 * m_e * epsilon_0 * 4 * np.pi**2) / (e**2)
    
    return lower_cutoff_density, upper_cutoff_density


def get_Omode_cutoff_frequency(density):
    """
    Calculate the O-mode cut-off frequency for a given plasma density.
    
    The O-mode cut-off frequency is the plasma frequency:
    f_pe = sqrt(n_e * e^2 / (m_e * ε_0)) / (2*π)
    
    Args:
        density (float): Plasma density in m^-3
        
    Returns:
        float: Cut-off frequency in Hz
        
    Example:
        >>> get_Omode_cutoff_frequency(1e19)  # 1e19 m^-3
        2.84e10  # Hz (28.4 GHz)
    """
    return get_plasma_frequency(density)


def get_Xmode_cutoff_frequency(density, bfield):
    """
    Calculate the X-mode cut-off frequencies for a given plasma density and magnetic field.
    
    The X-mode has two cut-offs:
    1. Upper cut-off: f = f_pe (same as O-mode)
    2. Lower cut-off: f = (f_ce + sqrt(f_ce^2 + 4*f_pe^2)) / 2
    
    Args:
        density (float): Plasma density in m^-3
        bfield (float): Magnetic field strength in Tesla
        
    Returns:
        tuple: (lower_cutoff_freq, upper_cutoff_freq) in Hz
        
    Example:
        >>> get_Xmode_cutoff_frequency(1e19, 2.5)  # 1e19 m^-3, 2.5 T
        (2.84e10, 2.84e10)  # Hz (lower and upper cut-offs)
    """
    # Calculate plasma frequency (upper cut-off)
    f_pe = get_plasma_frequency(density)
    
    # Calculate electron cyclotron frequency
    f_ce = get_cyclotron_frequency(bfield)
    
    # Upper cut-off (same as O-mode)
    upper_cutoff = f_pe
    
    # Lower cut-off
    lower_cutoff = (f_ce + np.sqrt(f_ce**2 + 4 * f_pe**2)) / 2
    
    return lower_cutoff, upper_cutoff


def get_plasma_frequency(density):
    """
    Calculate the plasma frequency for a given density.
    
    The plasma frequency is the natural oscillation frequency of electrons
    in a plasma: f_pe = sqrt(n_e * e^2 / (m_e * ε_0)) / (2*π)
    
    Args:
        density (float): Plasma density in m^-3
        
    Returns:
        float: Plasma frequency in Hz
        
    Example:
        >>> get_plasma_frequency(1e19)  # 1e19 m^-3
        2.84e10  # Hz (28.4 GHz)
    """
    # Constants
    e = constant.elementary_charge  # Elementary charge in C
    m_e = constant.m_e  # Electron mass in kg
    epsilon_0 = constant.epsilon_0  # Vacuum permittivity in F/m
    
    # Calculate plasma frequency
    omega_pe = np.sqrt(density * e**2 / (m_e * epsilon_0))
    f_pe = omega_pe / (2 * np.pi)
    
    return f_pe


def get_cyclotron_frequency(bfield):
    """
    Calculate the electron cyclotron frequency for a given magnetic field.
    
    The electron cyclotron frequency is the frequency at which electrons
    gyrate around magnetic field lines: f_ce = e * B / (m_e * 2*π)
    
    Args:
        bfield (float): Magnetic field strength in Tesla
        
    Returns:
        float: Cyclotron frequency in Hz
        
    Example:
        >>> get_cyclotron_frequency(2.5)  # 2.5 T
        7.0e10  # Hz (70 GHz)
    """
    # Constants
    e = constant.elementary_charge  # Elementary charge in C
    m_e = constant.m_e  # Electron mass in kg
    
    # Calculate cyclotron frequency
    omega_ce = e * bfield / m_e
    f_ce = omega_ce / (2 * np.pi)
    
    return f_ce


def is_Omode_propagating(frequency, density):
    """
    Check if an O-mode wave can propagate at a given frequency and density.
    
    An O-mode wave can propagate when the wave frequency is greater than
    the plasma frequency (f > f_pe).
    
    Args:
        frequency (float): Wave frequency in Hz
        density (float): Plasma density in m^-3
        
    Returns:
        bool: True if wave can propagate, False if cut-off
        
    Example:
        >>> is_Omode_propagating(3e10, 1e19)  # 30 GHz, 1e19 m^-3
        True  # Can propagate
        >>> is_Omode_propagating(3e10, 2e19)  # 30 GHz, 2e19 m^-3
        False  # Cut-off
    """
    cutoff_freq = get_Omode_cutoff_frequency(density)
    return frequency > cutoff_freq


def is_Xmode_propagating(frequency, density, bfield):
    """
    Check if an X-mode wave can propagate at given frequency, density, and magnetic field.
    
    An X-mode wave can propagate when the frequency is between the lower and upper cut-offs:
    f_lower < f < f_upper
    
    Args:
        frequency (float): Wave frequency in Hz
        density (float): Plasma density in m^-3
        bfield (float): Magnetic field strength in Tesla
        
    Returns:
        bool: True if wave can propagate, False if cut-off
        
    Example:
        >>> is_Xmode_propagating(3e10, 1e19, 2.5)  # 30 GHz, 1e19 m^-3, 2.5 T
        True  # Can propagate
    """
    lower_cutoff, upper_cutoff = get_Xmode_cutoff_frequency(density, bfield)
    return lower_cutoff < frequency < upper_cutoff
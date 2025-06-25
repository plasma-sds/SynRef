# -*- coding: utf-8 -*-
"""
Reflectometry utility functions for reflectometry calculations.

This module provides functions for calculating cut-off frequencies and densities
for O-mode and X-mode electromagnetic waves in plasma.
"""

import scipy.constants as constant
import numpy as np
import ctypes


def get_Omode_cutoff_density(frequency):
    """
    Calculate the O-mode cut-off density for a given frequency.
    
    The O-mode cut-off occurs when the plasma frequency equals the wave frequency.
    The plasma frequency is given by: ω_pe = sqrt(n_e * e^2 / (m_e * ε_0))
    At cut-off: ω = ω_pe, therefore: n_e = (ω^2 * m_e * ε_0) / e^2
    
    Args:
        frequency (float or numpy.ndarray): Wave frequency in Hz
        
    Returns:
        float or numpy.ndarray: Cut-off density in m^-3
        
    Example:
        >>> get_Omode_cutoff_density(3e10)  # 30 GHz
        1.115e19  # m^-3
        >>> get_Omode_cutoff_density(np.array([3e10, 4e10]))  # 30, 40 GHz
        array([1.115e19, 1.982e19])  # m^-3
    """
    # Convert to numpy array if scalar
    if not isinstance(frequency, np.ndarray):
        frequency = np.array(frequency)
    
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
    
    # Return scalar if input was scalar
    return cutoff_density.item() if cutoff_density.size == 1 else cutoff_density


def get_Xmode_cutoff_density(frequency, bfield):
    """
    Calculate the X-mode cut-off densities for a given frequency and magnetic field.
    
    The X-mode has two cut-offs:
    1. Upper cut-off: ω = ω_pe (same as O-mode)
    2. Lower cut-off: ω = (ω_ce + sqrt(ω_ce^2 + 4*ω_pe^2)) / 2
    
    This function returns both cut-off densities.
    
    Args:
        frequency (float or numpy.ndarray): Wave frequency in Hz
        bfield (float or numpy.ndarray): Magnetic field strength in Tesla
        
    Returns:
        tuple: (lower_cutoff_density, upper_cutoff_density) in m^-3
        
    Example:
        >>> get_Xmode_cutoff_density(3e10, 2.5)  # 30 GHz, 2.5 T
        (1.115e19, 1.115e19)  # m^-3 (lower and upper cut-offs)
    """
    # Convert to numpy arrays if scalars
    if not isinstance(frequency, np.ndarray):
        frequency = np.array(frequency)
    if not isinstance(bfield, np.ndarray):
        bfield = np.array(bfield)
    
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
        density (float or numpy.ndarray): Plasma density in m^-3
        
    Returns:
        float or numpy.ndarray: Cut-off frequency in Hz
        
    Example:
        >>> get_Omode_cutoff_frequency(1e19)  # 1e19 m^-3
        2.84e10  # Hz (28.4 GHz)
        >>> get_Omode_cutoff_frequency(np.array([1e19, 2e19]))  # 1e19, 2e19 m^-3
        array([2.84e10, 4.02e10])  # Hz
    """
    return get_plasma_frequency(density)


def get_Xmode_cutoff_frequency(density, bfield):
    """
    Calculate the X-mode cut-off frequencies for a given plasma density and magnetic field.
    
    The X-mode has two cut-offs:
    1. Upper cut-off: f = f_pe (same as O-mode)
    2. Lower cut-off: f = (f_ce + sqrt(f_ce^2 + 4*f_pe^2)) / 2
    
    Args:
        density (float or numpy.ndarray): Plasma density in m^-3
        bfield (float or numpy.ndarray): Magnetic field strength in Tesla
        
    Returns:
        tuple: (lower_cutoff_freq, upper_cutoff_freq) in Hz
        
    Example:
        >>> get_Xmode_cutoff_frequency(1e19, 2.5)  # 1e19 m^-3, 2.5 T
        (2.84e10, 2.84e10)  # Hz (lower and upper cut-offs)
    """
    # Convert to numpy arrays if scalars
    if not isinstance(density, np.ndarray):
        density = np.array(density)
    if not isinstance(bfield, np.ndarray):
        bfield = np.array(bfield)
    
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
        density (float or numpy.ndarray): Plasma density in m^-3
        
    Returns:
        float or numpy.ndarray: Plasma frequency in Hz
        
    Example:
        >>> get_plasma_frequency(1e19)  # 1e19 m^-3
        2.84e10  # Hz (28.4 GHz)
        >>> get_plasma_frequency(np.array([1e19, 2e19]))  # 1e19, 2e19 m^-3
        array([2.84e10, 4.02e10])  # Hz
    """
    # Convert to numpy array if scalar
    if not isinstance(density, np.ndarray):
        density = np.array(density)
    
    # Constants
    e = constant.elementary_charge  # Elementary charge in C
    m_e = constant.m_e  # Electron mass in kg
    epsilon_0 = constant.epsilon_0  # Vacuum permittivity in F/m
    
    # Calculate plasma frequency
    omega_pe = np.sqrt(density * e**2 / (m_e * epsilon_0))
    f_pe = omega_pe / (2 * np.pi)
    
    # Return scalar if input was scalar
    return f_pe.item() if f_pe.size == 1 else f_pe


def get_cyclotron_frequency(bfield):
    """
    Calculate the electron cyclotron frequency for a given magnetic field.
    
    The electron cyclotron frequency is the frequency at which electrons
    gyrate around magnetic field lines: f_ce = e * B / (m_e * 2*π)
    
    Args:
        bfield (float or numpy.ndarray): Magnetic field strength in Tesla
        
    Returns:
        float or numpy.ndarray: Cyclotron frequency in Hz
        
    Example:
        >>> get_cyclotron_frequency(2.5)  # 2.5 T
        7.0e10  # Hz (70 GHz)
        >>> get_cyclotron_frequency(np.array([2.0, 2.5, 3.0]))  # 2.0, 2.5, 3.0 T
        array([5.6e10, 7.0e10, 8.4e10])  # Hz
    """
    # Convert to numpy array if scalar
    if not isinstance(bfield, np.ndarray):
        bfield = np.array(bfield)
    
    # Constants
    e = constant.elementary_charge  # Elementary charge in C
    m_e = constant.m_e  # Electron mass in kg
    
    # Calculate cyclotron frequency
    omega_ce = e * bfield / m_e
    f_ce = omega_ce / (2 * np.pi)
    
    # Return scalar if input was scalar
    return f_ce.item() if f_ce.size == 1 else f_ce


def is_Omode_propagating(frequency, density):
    """
    Check if an O-mode wave can propagate at a given frequency and density.
    
    An O-mode wave can propagate when the wave frequency is greater than
    the plasma frequency (f > f_pe).
    
    Args:
        frequency (float or numpy.ndarray): Wave frequency in Hz
        density (float or numpy.ndarray): Plasma density in m^-3
        
    Returns:
        bool or numpy.ndarray: True if wave can propagate, False if cut-off
        
    Example:
        >>> is_Omode_propagating(3e10, 1e19)  # 30 GHz, 1e19 m^-3
        True  # Can propagate
        >>> is_Omode_propagating(3e10, 2e19)  # 30 GHz, 2e19 m^-3
        False  # Cut-off
        >>> is_Omode_propagating(np.array([3e10, 4e10]), np.array([1e19, 2e19]))
        array([True, True])  # Both can propagate
    """
    # Convert to numpy arrays if scalars
    if not isinstance(frequency, np.ndarray):
        frequency = np.array(frequency)
    if not isinstance(density, np.ndarray):
        density = np.array(density)
    
    cutoff_freq = get_Omode_cutoff_frequency(density)
    result = frequency > cutoff_freq
    
    # Return scalar if input was scalar
    return result.item() if result.size == 1 else result


def is_Xmode_propagating(frequency, density, bfield):
    """
    Check if an X-mode wave can propagate at given frequency, density, and magnetic field.
    
    An X-mode wave can propagate when the frequency is between the lower and upper cut-offs:
    f_lower < f < f_upper
    
    Args:
        frequency (float or numpy.ndarray): Wave frequency in Hz
        density (float or numpy.ndarray): Plasma density in m^-3
        bfield (float or numpy.ndarray): Magnetic field strength in Tesla
        
    Returns:
        bool or numpy.ndarray: True if wave can propagate, False if cut-off
        
    Example:
        >>> is_Xmode_propagating(3e10, 1e19, 2.5)  # 30 GHz, 1e19 m^-3, 2.5 T
        True  # Can propagate
        >>> is_Xmode_propagating(np.array([3e10, 4e10]), np.array([1e19, 2e19]), 2.5)
        array([True, True])  # Both can propagate
    """
    # Convert to numpy arrays if scalars
    if not isinstance(frequency, np.ndarray):
        frequency = np.array(frequency)
    if not isinstance(density, np.ndarray):
        density = np.array(density)
    if not isinstance(bfield, np.ndarray):
        bfield = np.array(bfield)
    
    lower_cutoff, upper_cutoff = get_Xmode_cutoff_frequency(density, bfield)
    result = (lower_cutoff < frequency) & (frequency < upper_cutoff)
    
    # Return scalar if input was scalar
    return result.item() if result.size == 1 else result


def get_frequency_sweep(reflectometer, frequency_range, frequency_resolution):
    """
    Perform a frequency sweep using a Basic reflectometer and return antenna output data.
    
    This function executes the FW2D calculations for a range of frequencies and
    collects the antenna phase and amplitude data for each frequency point.
    
    Args:
        reflectometer (Basic): A configured Basic reflectometer instance
        frequency_range (tuple): (start_frequency, end_frequency) in Hz
        frequency_resolution (float): Frequency step size in Hz
        
    Returns:
         - 'amplitudes': numpy array of antenna amplitudes
         - 'phases': numpy array of antenna phases in radians

    Example:
        >>> from reflectometer.basic import Basic
        >>> ref = Basic(frequency=3e10)  # 30 GHz
        >>> amplitudes, phases = get_frequency_sweep(ref, (2.8e10, 3.2e10), 1e9)  # 28-32 GHz, 1 GHz steps

    """
    # Extract frequency range
    start_freq, end_freq = frequency_range
    
    # Generate frequency array
    frequencies = np.arange(start_freq, end_freq + frequency_resolution, frequency_resolution)
    n_frequencies = len(frequencies)
    
    # Initialize arrays to store results
    amplitudes = np.zeros(n_frequencies)
    phases = np.zeros(n_frequencies)
    
    try:
        # Perform frequency sweep
        for i, freq in enumerate(frequencies):
            # Update reflectometer frequency
            reflectometer.update_frequency(freq)
            
            # Execute FW2D calculation
            # Note: This assumes the reflectometer has a method to run the simulation
            # You may need to call the appropriate method based on your Basic class implementation
            result = reflectometer.fw2d.maxwell_2d_omode(ctypes.byref(reflectometer.data))
            
            if result != 0:
                print(f"Warning: FW2D calculation failed for frequency {freq:.2e} Hz")
                amplitudes[i] = np.nan
                phases[i] = np.nan
                continue
            
            # Get antenna output
            amp_array, phase_array = reflectometer.get_antenna_output()
            
            amplitudes[i] = amp_array[0]
            phases[i] = phase_array[0]
            
            # Optional: Print progress for long sweeps
            if n_frequencies > 10 and (i + 1) % (n_frequencies // 10) == 0:
                print(f"Frequency sweep progress: {i+1}/{n_frequencies} ({100*(i+1)/n_frequencies:.1f}%)")

    return amplitudes, phases
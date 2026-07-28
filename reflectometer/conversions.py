# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 16:26:10 2025

@author: aszta
"""

import ctypes

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

def _build_extended_incident_arrays(self, ampl_phys, phase_phys, TFSF):
    """
    Build EXTENDED grid arrays (amplitude & phase) to match the C-side layout.

    The physical grid (length ny_phys+1) is embedded in an extended grid that
    adds TFSF extra points on each side (for the TFSF/PML boundary region).
    Points outside the physical region are set to zero.

    Parameters
    ----------
    ampl_phys : sequence of float
        Amplitude values on the physical grid, indexed 0..ny_phys.
    phase_phys : sequence of float
        Phase values on the physical grid, indexed 0..ny_phys.
    TFSF : int
        Number of padding points added on each side of the physical grid.

    Returns
    -------
    (ampl_2d, phase_2d) : tuple of ctypes arrays
        POINTER(c_double) arrays of length ny_ext+1, also assigned to
        self.data.ampl_inc / self.data.phase_inc as a side effect.
    """
    ny_phys = self.data.ny
    ny_ext = ny_phys - 1 + 2 * TFSF

    ampl_2d = (ctypes.POINTER(ctypes.c_double) * (ny_ext + 1))()
    phase_2d = (ctypes.POINTER(ctypes.c_double) * (ny_ext + 1))()
    self._ampl_inc_rows = []
    self._phase_inc_rows = []

    for j_ext in range(ny_ext + 1):
        row_ampl = (ctypes.c_double * 1)()
        row_phase = (ctypes.c_double * 1)()

        # map extended index j_ext to physical index j_phys = j_ext - TFSF
        if TFSF <= j_ext <= TFSF + ny_phys:
            j_p = j_ext - TFSF
            row_ampl[0] = float(ampl_phys[j_p])
            row_phase[0] = float(phase_phys[j_p])
        else:
            # in PML / outside physical region → zero
            row_ampl[0] = 0.0
            row_phase[0] = 0.0

        ampl_2d[j_ext] = row_ampl
        phase_2d[j_ext] = row_phase
        self._ampl_inc_rows.append(row_ampl)
        self._phase_inc_rows.append(row_phase)

    # keep top-level pointers alive
    self._bufs['ampl_inc'] = ampl_2d
    self._bufs['phase_inc'] = phase_2d
    self.data.ampl_inc = self._bufs['ampl_inc']
    self.data.phase_inc = self._bufs['phase_inc']

    return ampl_2d, phase_2d
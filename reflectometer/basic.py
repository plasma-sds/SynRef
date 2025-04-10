# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:51:15 2025

@author: asztalos
"""

import ctypes

class InputData(ctypes.Structure):                                  #Input data structure for Basic FW2D
    _fields_ = [
        ("f0", ctypes.c_double),                                            # Inpute wave frequency [Hz]
        ("nt", ctypes.c_int),                                               # Number of tenporal iterations [-] 
        ("nx", ctypes.c_int),                                               # Number of points along x axis [-]
        ("ny", ctypes.c_int),                                               # Number of points along y axis [-] 
        ("dx", ctypes.c_double),                                            # Spatail resolution [-]
        ("yante", ctypes.c_int),                                            # Position of the antenna
        ("waist", ctypes.c_int),                                            # Beam waist in mesh point numbers [-]
        ("angle", ctypes.c_double),                                         # Angle of propagation in [deg]
        ("b0", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),            # Magnetic field
        ("ne", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),            # Plasma density field
        ("ampl_ant", ctypes.POINTER(ctypes.c_double)),                      # E amplitude at the antenna
        ("fase_ant", ctypes.POINTER(ctypes.c_double)),                      # Phase at the antenna
    ]
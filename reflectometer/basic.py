# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:51:15 2025

@author: asztalos
"""

import ctypes
import numpy
import scipy.constants as constant

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
    
    
class Basic():
    def __init__(self, wavemode='O', solver='basic', frequency=3e10, 
                 density='default', b_field='default', x='default', y='default',
                 antenna_pos='default', beam_waist='default', angle=0, 
                 expected_reflection_distance='default'):
        self.data = InputData()
        self.__set_frequency(frequency)
        self.__set_frequency_dependence()
        self.__set_density_field(x=x, y=y, density=density)
  
        
    def __set_frequency(self, frequency):
        self.data.f0 = frequency
        self.frequency = frequency
        
    def __set_dt(self):
        self.dt = 1 / self.frequency / 40
        
    def __set_wavelength(self):
        self.wavelength = constant.c / self.frequency
                
    def __set_dx(self):
        self.dx = self.wave_length / 20
        self.data.dx = self.dx 
        
    def __set_frequency_dependence(self):
        self.__set_dt()
        self.__set_wavelength()
        self.__set_dx()
        
    def __set_density_field(self, x, y, density):
        if isinstance(density, str):
            self.__make_default_density()
        elif isinstance(density, numpy.ndarray):
            self.__fit_density_to_grid(x=x, y=y, density=density)
        else:
            raise(ValueError('Expected a numpy ndarray data type. Input datatype does not match'))            
        
    def __set_spatial_resolutions(self):
        self.nx = self.x_range // self.dx
        self.ny = self.y_range // self.dx
        self.data.nx = self.nx
        self.data.ny = self.ny
            
    def __make_default_density(self):
        self.x_range = 0.1 # in m
        self.y_range = 0.1 # in m
        self.__set_spatail_resolutions()
        
        
    def __fit_density_to_grid(self, x, y, density):
        self.x_range = x[-1] - x[0]
        self.y_range = y[-1] - y[0]
        self.__set_spatial_resolutions()
        
    
    def update_frequency(self, frequency):
        pass
        
    def update_density(self, density, x, y):
        pass
    
    def update_angle(self, angle):
        pass
    
    def update_antenna(self, antenna):
        pass
    
    def update_waist(self, waist):
        pass
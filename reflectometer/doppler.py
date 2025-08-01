# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:55:33 2025

@author: akosk
"""
import numpy
import sys
import os
sys.path.append(os.path.dirname(__file__))

from reflectometer.basic import Basic

class Doppler(Basic):
    
    setup_library = {
        "ASDEX":
            {"wavemode": 'O', "solver": 'basic',
             "frequency": 3e10, "reflection_distance": 'default',
             "X":(-1840e-4, -1740e-3), "Y":(-150e-3, 150e-3),
             "antenna_pos": 150e-3, "beam_waist": 'default', "angle": -20},
        "JET":
            {"wavemode": 'O', "solver": 'basic',
             "frequency": 3e10, "reflection_distance": 'default',
             "X":(-1840e-4, -1740e-3), "Y":(-150e-3, 150e-3),
             "antenna_pos": 150e-3, "beam_waist": 'default', "angle": -20},
        "COMPASS":
            {"wavemode": 'O', "solver": 'basic',
             "frequency": 3e10, "reflection_distance": 'default',
             "X":(-1840e-4, -1740e-3), "Y":(-150e-3, 150e-3),
             "antenna_pos": 150e-3, "beam_waist": 'default', "angle": -20},
        "TEST": 
            {"wavemode": 'O', "solver": 'basic',
             "frequency": 3e10, "reflection_distance": 'default',
             "X":(-1840e-4, -1740e-3), "Y":(-150e-3, 150e-3),
             "antenna_pos": 150e-3, "beam_waist": 'default', "angle": -20} }
    
    def __init__(self, version, x='default', y='default',
                 density_evolution = "default", ez_field = "default"):
        
        self.version = version
        lib = self.setup_library[self.version]
        
        Basic.__init__(self, wavemode=lib["wavemode"], solver=lib["solver"], 
                       frequency = lib["frequency"], 
                       reflection_distance = lib["reflection_distance"],
                       antenna_pos = lib["antenna_pos"], 
                       beam_waist = lib["beam_waist"], angle = lib["angle"],
                       
                       x = x, y = y,
                       density = density_evolution[0], b_field = ez_field)
        
        
        self.__axis_reallocation()
    
    
    def __axis_reallocation(self):
        lib = self.setup_library[self.version]
        self.X = numpy.linspace(*lib["X"], self.nx)
        self.Y = numpy.linspace(*lib["Y"], self.ny)
        
    # def frequency_sweep():
        
    # def density_sweep():
        
    # def full_sweep():
        
    # def plot_section():
    
    # def plot_wave():
        
            
            
    # _________________________________________________________________________
    # update the properties, referenced in a dictionary
    # def update_parameters(self, dictionary: dict):
    #     for key, value in dictionary.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else: raise(ValueError('Referenced attribute does not exist'))
    














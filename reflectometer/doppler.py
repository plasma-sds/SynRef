# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:55:33 2025

@author: akosk
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

from reflectometer.basic import Basic
import reflectometer.file_management as fima
import reflectometer.functions as func

class Doppler(Basic):
    
    setup_library = {
        "W7X QMR-V1":
            {"wavemode": 'O', "solver": 'basic', 
             "reflection_distance": 'default',
             "frequency":    np.arange(50, 70 + 0.1, 1) *1e9, # [GHz]
             "field_width":  np.array([0, 200e-3]), 
             "field_height": np.array([-150e-3, 150e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015}, 
        "W7X QMR-E1":
            {"wavemode": 'O', "solver": 'basic', 
             "reflection_distance": 'default',
             "frequency":    np.arange(60, 90 + 0.1, 1) *1e9, # [GHz]
             "field_width":  np.array([0, 200e-3]), 
             "field_height": np.array([-150e-3, 150e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015}}
    
    def __init__(self, library = "W7X QMR-V1",
                 density_evolution = "default", ez_evolution = "default"):
        
        if   isinstance(library, dict):
            self.library = library
        elif isinstance(library, str):
            self.library = self.setup_library[library]
        # else: raise ValueError
        
        self.__create_config()
        self.__initialize(density_evolution, ez_evolution)
        
        self.__axis_reallocation()
        
        
    def __axis_reallocation(self):
        lib = self.library
        self.X = np.linspace(-lib["field_width"][1], -lib["field_width"][0],
                             self.nx) + lib["antenna_pos"][0]
        self.Y = np.linspace(*lib["field_height"], 
                             self.ny) + lib["antenna_pos"][2]
        
    # def __angle_conversion(self):
    #     r=0
    
    def __create_config(self, ref = "default"):
        if (ref == "default"):
            lib = self.library
            self.config ={
                'wavemode': lib["wavemode"], 'solver': lib["solver"],
                'frequency': lib["frequency"][0], 
                'antenna_pos': -1 * lib["field_height"][0],
                'beam_waist': lib["beam_waist"], 
                'angle': lib["LOS"][0], 
                'reflection_distance': lib["reflection_distance"]}
        else:
            self.config = {
                'wavemode': ref.wavemode, 'solver': ref.solver,
                'frequency': ref.frequency, 
                'antenna_pos': ref.antenna_pos,
                'beam_waist': ref.beam_waist_si, 
                'angle': ref.angle, 
                'reflection_distance': ref.reflection_distance}
            
    def __relative_axis(self):
        lib = self.library
        
    def __initialize(self, density_evolution, ez_field):
        lib = self.library
        Basic.__init__(self,
            x = np.linspace(0, lib["field_width"][1] - lib["field_width"][0],
                            density_evolution.shape[2]),
            y = np.linspace(0, lib["field_height"][1]-lib["field_height"][0],
                            density_evolution.shape[1]),
            density = density_evolution[0], b_field = ez_field, **self.config)
        
    def get_libraries(self): return self.setup_library
        
    
    

        
    # def frequency_sweep(self):
        
        
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
    










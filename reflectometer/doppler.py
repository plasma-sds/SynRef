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
        
        self.__positional_correction()
        
    def __positional_correction(self):
        lib = self.library
        Lx, Ly = lib["field_width"], lib["field_height"]
        x,y,z = lib["antenna_pos"]
        r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
        
        x_abs = np.linspace(r - Lx[0] , r - Lx[1],  len(self.x) )
        y_abs = np.linspace( *(Ly + z),             len(self.y) )
        
        X_abs = np.linspace(r - Lx[0] , r - Lx[1],      self.nx )
        Y_abs = np.linspace( *(Ly + z),                 self.ny )
        
        x = np.linspace(0, Lx[1] - Lx[0], len(self.x) )
        y = np.linspace(0, Ly[1] - Ly[0], len(self.y) )
        
        X = np.linspace(0, Lx[1] - Lx[0],    self.nx  )
        Y = np.linspace(0, Ly[1] - Ly[0],    self.ny  )
        
        setup_map = {"x": x, "y": y, "x_abs": x_abs, "y_abs": y_abs,
                     "X": X, "Y": Y, "X_abs": X_abs, "Y_abs": Y_abs,
                     "antenna_XYZ": np.array([x,y,z]),
                     "antenna_RTZ": np.array([r, theta, z])}
    
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
        
    
    def frequency_sweep(self, frequency_indices = "default",
                        con_filename = False, path = ''):
        
        if frequency_indices == "default": 
            frequencies = self.library["frequency"]
        else: frequencies = self.library["frequency"][frequency_indices]
        print(frequencies)
        
        if con_filename == True: func.get_frequency_sweep(self, frequencies)
        else: func.get_frequency_sweep(self, frequencies,
                                       con_filename=con_filename, path=path)
            
        
    def density_sweep(self, density_indices = "default",
                      con_filename = False, path = ''):
        
        if density_indices == "default": 
            density_indices = np.arange(self.density_evolution.shape[0])
        
        if con_filename == True: 
            func.get_density_sweep(self, self.density_evolution, 
                                   self.x, self.y, density_indices)
        else: func.get_density_sweep(self, self.density_evolution, 
                                     self.x, self.y, density_indices,
                                     con_filename=con_filename, path=path)
        
    
    def full_sweep(self, frequency_indices = "default", 
                   density_indices = "default",
                   con_filename = False, path = ''):
        
        if frequency_indices == "default": 
            frequencies = self.library["frequency"]
        else: frequencies = self.library["frequency"][frequency_indices]
        
        if density_indices == "default": 
            density_indices = np.arange(self.density_evolution.shape[0])
        
        if con_filename == True: 
            func.get_full_sweep(self, self.density_evolution, self.x, self.y,
                                frequencies, density_indices)
        else: func.get_full_sweep(self, self.density_evolution, self.x, self.y,
                                  frequencies, density_indices,
                                  con_filename=con_filename, path=path)
    
    
    # def plot_section():
    
    # def plot_wave():
        
            
            
    # _________________________________________________________________________
    # update the properties, referenced in a dictionary
    # def update_parameters(self, dictionary: dict):
    #     for key, value in dictionary.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else: raise(ValueError('Referenced attribute does not exist'))
    










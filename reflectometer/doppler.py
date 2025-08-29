# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:55:33 2025

@author: akosk
"""
import numpy as np
import sys
from os.path import dirname
from os.path import join
from os import mkdir
sys.path.append(dirname(__file__))

from reflectometer.basic import Basic
import reflectometer.functions as func

class Doppler(Basic):
    
    setup_library = {
        "W7X QMR-V1":
            {"wavemode": 'O', "solver": 'basic', 
             "reflection_distance": 'default',
             "frequency":    np.arange(50, 70 + 0.1, 1) *1e9, # [GHz]
             "field_width":  np.array([0, 100e-3]), 
             "field_height": np.array([-150e-3, 150e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015}, 
        "W7X QMR-E1":
            {"wavemode": 'O', "solver": 'basic', 
             "reflection_distance": 'default',
             "frequency":    np.arange(60, 90 + 0.1, 1) *1e9, # [GHz]
             "field_width":  np.array([0, 100e-3]), 
             "field_height": np.array([-150e-3, 150e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015},
        "test":
            {"wavemode": 'O', "solver": 'basic', 
             "reflection_distance": 'default',
             "frequency":    np.arange(25, 40 + 0.1, 1) *1e9, # [GHz]
             "field_width":  np.array([0, 100e-3]), 
             "field_height": np.array([-150e-3 - 16.24e-3, 150e-3 + 16.24e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015} }
    
    def __init__(self, library = "W7X QMR-V1", t = "default",
                 density_evolution = ["default"], ez_evolution = ["default"],
                 working_directory = ''):
        """
        

        Parameters
        ----------
        library : TYPE, optional
            DESCRIPTION. The default is "W7X QMR-V1".
        density_evolution : TYPE, optional
            DESCRIPTION. The default is ["default"].
        ez_evolution : TYPE, optional
            DESCRIPTION. The default is ["default"].
        working_directory : TYPE, optional
            DESCRIPTION. The default is ''.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.path = working_directory
        
        if   isinstance(library, dict):
            self.library = library
        elif isinstance(library, str):
            self.library = self.setup_library[library]
        # else: raise ValueError
        
        self.density_evolution = density_evolution
        self.ez_evolution = ez_evolution
        
        self.__create_config()
        self.__initialize()
        
        self.__positional_correction()
        
    def __positional_correction(self):
        lib = self.library
        Lx, Ly = lib["field_width"], lib["field_height"]
        x,y,z = lib["antenna_pos"]
        r, theta = np.sqrt(x**2 + y**2), np.arctan2(y, x)
        XYZ, RTZ = np.array([x,y,z]), np.array([r, theta, z])
        
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
                     "antenna_XYZ": XYZ, "antenna_RTZ": RTZ}
    
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
        
    def __initialize(self):
        lib = self.library
        Basic.__init__(self,
            x = np.linspace(0, lib["field_width"][1] - lib["field_width"][0],
                            self.density_evolution.shape[2]),
            y = np.linspace(0, lib["field_height"][1]-lib["field_height"][0],
                            self.density_evolution.shape[1]),
            density = self.density_evolution[0], 
            b_field = self.ez_evolution[0],    **self.config)
        
    def get_libraries(self): return self.setup_library
        
    
    def frequency_sweep(self, frequency_indices = "default", 
                        density_index = "default",
                        con_filename = False, path = "default"):
        if path == "default": path = self.path
        
        if frequency_indices == "default": 
            frequencies = self.library["frequency"]
        else: frequencies = self.library["frequency"][frequency_indices]
        
        if density_index != "default": 
            self.update_density(self.density_evolution[density_index], 
                                self.x, self.y, reflection_distance='default')
        
        if con_filename == True: 
            con_filename = "config_{dens:04d}_{freq}.json".format(
                dens=density_index, freq="{freq:03d}")
            
        func.get_frequency_sweep(self, frequencies,
                                 con_filename=con_filename, path=path)
            
        
    def density_sweep(self, frequency_index = "default", 
                      density_indices = "default",
                      con_filename = False, path = "default"):
        if path == "default": path = self.path
        
        if density_indices == "default": 
            density_indices = np.arange(self.density_evolution.shape[0])
        
        if frequency_index != "default":
            self.update_frequency(self.library["frequency"][frequency_index])
        
        if con_filename == True: 
            con_filename = "config_{dens}_{freq:03d}.json".format(
                dens="{dens:03d}", freq=frequency_index)
            
        func.get_density_sweep(self, self.density_evolution, 
                               self.x, self.y, density_indices,
                               con_filename=con_filename, path=path)
        
    
    def full_sweep(self, frequency_indices = "default", 
                   density_indices = "default",
                   con_filename = False, path = "default"):
        
        if path == "default": path = self.path
        if con_filename != False: 
            path = join(path, "config_files")
            try: mkdir(path)
            except: pass
        
        if frequency_indices == "default": 
            frequencies = self.library["frequency"]
        else: frequencies = self.library["frequency"][frequency_indices]
        
        if density_indices == "default": 
            density_indices = np.arange(self.density_evolution.shape[0])
        
        if con_filename == True: 
            con_filename = "config_{dens:04d}_{freq:03d}.json"
            
        func.get_full_sweep(self, self.density_evolution, self.x, self.y,
                            frequencies, density_indices,
                            con_filename=con_filename, path=path)
    
    
    # def plot_section():
    
    # def plot_wave( frame = 0 ):
        
            
            
    # _________________________________________________________________________
    # update the properties, referenced in a dictionary
    # def update_parameters(self, dictionary: dict):
    #     for key, value in dictionary.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         else: raise(ValueError('Referenced attribute does not exist'))
    










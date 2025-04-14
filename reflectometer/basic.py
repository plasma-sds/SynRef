# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:51:15 2025

@author: asztalos
"""

import os
import ctypes
import numpy
import scipy.constants as constant
from scipy.interpolate import RegularGridInterpolator as rgi

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
                 reflection_distance='default'):
        
        
        self.data = InputData()
        self.__set_frequency(frequency)
        self.__set_frequency_dependence()
        self.__set_density_field(x=x, y=y, density=density)
        self.__set_magnetic_field(x=x, y=y, b_field=b_field)
        self.__set_angle_antenna(angle=angle)
        self.__set_simulation_timesteps(reflection_distance=reflection_distance)
        self.__set_beam_waist(waist=beam_waist)
        self.__set_antenna_pos(antenna_pos=antenna_pos)
        
  
        
    def __set_frequency(self, frequency):
        self.data.f0 = frequency
        self.frequency = frequency
        
    def __set_dt(self):
        self.dt = 1 / self.frequency / 40
        
    def __set_wavelength(self):
        self.wavelength = constant.c / self.frequency
                
    def __set_dx(self):
        self.dx = self.wavelength / 20
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
        self.nx = (self.x[-1] - self.x[0]) // self.dx
        self.ny = (self.y[-1] - self.y[0]) // self.dx
        self.data.nx = self.nx
        self.data.ny = self.ny
            
    def __make_default_density(self):
        self.x = numpy.arrange(0,100,1) * 0.001# in m
        self.y = numpy.arrange(0,100,1) * 0.001# in m
        self.__set_spatail_resolutions()
        
        profile = numpy.zeros(self.nx)
        profile[50:] = numpy.linspace(0, 3e19, num=(self.nx-50))
        default_density = numpy.array([list(profile) for i in range(int(self.nx))])
        
        ne = (ctypes.POINTER(ctypes.c_double) * self.data.nx)()
        for i in range(self.data.nx):
            ne[i] = (ctypes.c_double * self.data.ny)()
            for j in range(self.data.ny):
                ne[i][j] = default_density[i,j]
        self.data.ne = ne
        
    def __fit_density_to_grid(self, x, y, density):
        self.x
        self.y
        self.__set_spatial_resolutions()
        
        density_interpolator = rgi((x, y), density, method="cubic", fill_value=None)
        x_grid = x[0] + numpy.arange(int(self.nx))*self.dx
        y_grid = y[0] + numpy.arange(int(self.ny))*self.dx

        ne = (ctypes.POINTER(ctypes.c_double) * self.data.nx)()
        for i in range(self.data.nx):
            ne[i] = (ctypes.c_double * self.data.ny)()
            for j in range(self.data.ny):
                ne[i][j] = density_interpolator((x_grid[i], y_grid[j]))
        self.data.ne = ne      
        
    def __set_magnetic_field(self, x, y, b_field):
        if isinstance(b_field, str):
            self.__make_default_bfield()
        elif isinstance(b_field, numpy.ndarray()):
             self.__fit_bfield_to_grid(x=x, y=y, b_field=b_field)
        else:
            raise(ValueError('Expected a numpy ndarray data type. Input datatype does not match'))
            
            
    def __make_default_bfield(self):
        b0 = (ctypes.POINTER(ctypes.c_double) * self.data.nx)()  # Create an array of pointers (for each row)
        for i in range(self.data.nx):
            b0[i] = (ctypes.c_double * self.data.ny)()  # Create the row with ny elements
            for j in range(self.data.ny):
                b0[i][j] = 2.5
        self.data.b0 = b0
        
    def __fit_bfield_to_grid(self, x, y, b_field):
        bfield_interpolator = rgi((x, y), b_field, method="cubic", fill_value=None)
        x_grid = x[0] + numpy.arange(int(self.nx))*self.dx
        y_grid = y[0] + numpy.arange(int(self.ny))*self.dx
        
        b0 = (ctypes.POINTER(ctypes.c_double) * self.data.nx)()  # Create an array of pointers (for each row)
        for i in range(self.data.nx):
            b0[i] = (ctypes.c_double * self.data.ny)()  # Create the row with ny elements
            for j in range(self.data.ny):
                b0[i][j] = bfield_interpolator((x_grid[i], y_grid[j]))
        self.data.b0 = b0
        
    def __set_angle_antenna(self, angle):
        self.angle = angle
        self.data.angle = angle
        
    def __set_simulation_timesteps(self, reflection_distance):
        if isinstance(reflection_distance, str):
            time = 2*numpy.cos(numpy.radians(self.angle))*self.x_range / constant.c
        else:
            time = 2*numpy.cos(numpy.radians(self.angle))*reflection_distance / constant.c
        
        self.__set_timesteps(simulation_time=time*1.05)
            
    def __set_timesteps(self, simulation_time):
        self.nt = simulation_time // self.dt
        self.data.nt = self.nt
        
    def __set_beam_waist(self, waist):
        if isinstance(waist, str):
            self.beam_waist_si = 0.03 # in cm
        else:
            self.beam_waist_si = waist
        self.data.waist = self.beam_waist_si // self.dx
    
    def __set_antenna_pos(self, antenna_pos):
        if isinstance(antenna_pos, str):
            self.antenna_pos = 0.05 #in cm
        else:
            self.antenna_pos = antenna_pos
        self.data.yante = int(self.ny - (antenna_pos - self.y[0]))
        
    def __set_solver(self, wavemode, solver):
        self.__set_solver_path(wavemode=wavemode, solver=solver)
        self.fw2d = ctypes.CDLL(self.fw2d_path)
        self.fw2d.maxwell_2d_omode.argtypes = [ctypes.POINTER(InputData)]
        self.fw2d.maxwell_2d_omode.restype = ctypes.c_int
        
    def __set_solver_path(self, wavemode, solver):
        if not isinstance(wavemode, str):
            raise TypeError('The expected type for the wavemode input is str.')
        if wavemode == "O":
            self.wavemode = wavemode
            mode_path='maxwell_2d_omode'
        else:
            raise ValueError('The requested wave type is not supported. The class is set up to support X or O mode waves.')
            
        if not isinstance(solver, str):
            raise TypeError('The expected type for the solver input is str.')
        if solver == 'basic':
            self.solver=solver
            solver_path = ''
        elif solver == 'ez_evo':
            pass
        elif solver == 'multi_ant':
            pass
        elif solver == 'multi_evo':
            pass
        else:
            raise ValueError('The requested solver type is not supported. Please consult documentation.')
   
        self.fw2d_path = os.path.join(os.path.dirname(__file__), '..', 
                                      'fw2d', mode_path+solver_path+'.dll')       
   
    
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
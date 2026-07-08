# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:51:15 2025

@author: asztalos

This module provides a Python interface for Basic FW2D (Full Wave 2D) simulation.
It handles plasma density and magnetic field data, and interfaces with C-based
electromagnetic wave propagation solvers.
"""

import os
import ctypes
import numpy


from .conversions import from_unit_to_centi
import scipy.constants as constant
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from hardware.utils.antenna.pyramidal_farfield_to_fw2d import pyramidal_farfield_to_fw2d

class InputData(ctypes.Structure):                                  #Input data structure for Basic FW2D
    """
    C structure for passing input data to the Basic FW2D simulation.
    
    This structure contains all the parameters needed for the electromagnetic
    wave propagation simulation including frequency, spatial resolution,
    plasma density, magnetic field, and antenna configuration.
    """
    _fields_ = [
        ("f0", ctypes.c_double),                                            # Inpute wave frequency [Hz]
        ("nt", ctypes.c_int),                                               # Number of tenporal iterations [-] 
        ("nx", ctypes.c_int),                                               # Number of points along x axis [-]
        ("ny", ctypes.c_int),                                               # Number of points along y axis [-] 
        ("dx", ctypes.c_double),                                            # Spatail resolution [-]
        ("ampl_inc", ctypes.c_double),                                      # Amplitude array of the incident wave [-]
        ("phase_inc", ctypes.c_double),                                     # Phase array of the incident wave [-]
        ("b0", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),            # Magnetic field
        ("ne", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),            # Plasma density field
        ("ez_final", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),      # Final version of the ez field
        ("ampl_ant", ctypes.POINTER(ctypes.c_double)),                      # E amplitude at the antenna
        ("fase_ant", ctypes.POINTER(ctypes.c_double)),                      # Phase at the antenna
    ]
    
""" Moved to Python environment:
("yante", ctypes.c_int),                                            # Position of the antenna
("waist", ctypes.c_int),                                            # Beam waist in mesh point numbers [-]
("angle", ctypes.c_double),                                         # Angle of propagation in [deg]"""
    
class Basic():
    """
    Main class for Basic FW2D electromagnetic wave propagation simulation.
    
    This class provides an interface to C-based electromagnetic wave propagation
    solvers for plasma physics applications. It handles plasma density and
    magnetic field data, antenna configuration, and simulation parameters.
    
    Attributes:
        data (InputData): C structure containing all simulation parameters
        frequency (float): Wave frequency in Hz
        wavelength (float): Wavelength in meters
        dx (float): Spatial resolution in meters
        dt (float): Temporal resolution in seconds
        nx (int): Number of grid points along x-axis
        ny (int): Number of grid points along y-axis
        nt (int): Number of temporal iterations
        angle (float): Propagation angle in degrees
        beam_waist_si (float): Beam waist in meters
        antenna_pos (float): Antenna position in meters
        reflection_distance (float): Distance for reflection calculation
        x (numpy.ndarray): X-axis coordinates
        y (numpy.ndarray): Y-axis coordinates
        fw2d: C library interface for the solver
    """
    
    def __init__(self, wavemode='O', solver='basic', frequency=3e10, 
                 density='default', b_field='default', x='default', y='default',
                 antenna_pos='default', beam_waist='default', angle=0, 
                 reflection_distance='default', wavesource='gaussian',
                 horn_a1=40, horn_b1=40, horn_rho1=50, horn_rho2=50, horn_x=-10, horn_y=30):
        """
        Initialize the Basic FW2D simulation.
        
        Args:
            wavemode (str): Wave mode ('O' for O-mode, 'X' for X-mode)
            solver (str): Solver type ('basic', 'ez_evo', 'multi_ant', 'multi_evo')
            frequency (float): Wave frequency in Hz
            density (str or numpy.ndarray): Plasma density field or 'default'
            b_field (str or numpy.ndarray): Magnetic field or 'default'
            x (str or numpy.ndarray): X-axis coordinates or 'default'
            y (str or numpy.ndarray): Y-axis coordinates or 'default'
            antenna_pos (str or float): Antenna position in meters or 'default'
            beam_waist (str or float): Beam waist in meters or 'default'
            angle (float): Propagation angle in degrees
            reflection_distance (str or float): Reflection distance in meters or 'default'
            wavesource (str or numpy.ndarray): Wave source field or 'default'
        """
        
        self.__set_solver_and_datastruct(wavemode=wavemode, solver=solver)
        self.__set_frequency(frequency)
        self.__set_frequency_dependence()
        self.__set_density_field(x=x, y=y, density=density)
        self.__set_magnetic_field(x=x, y=y, b_field=b_field)
        self.__set_angle_antenna(angle=angle)
        self.__set_simulation_timesteps(reflection_distance=reflection_distance)
        self.__set_beam_waist(beam_waist=beam_waist)
        self.__set_antenna_pos(antenna_pos=antenna_pos)
        self.__set_outputdata(solver=solver)
        self.__set_ampl_inc_phase_inc(wavesource=wavesource)

        self.horn_a1 = horn_a1
        self.horn_b1 = horn_b1
        self.horn_rho1 = horn_rho1
        self.horn_rho2 = horn_rho2
        self.horn_x = horn_x
        self.horn_y = horn_y

    def __set_frequency(self, frequency):
        """
        Set the wave frequency and update the data structure.
        
        Args:
            frequency (float): Wave frequency in Hz
        """
        self.data.f0 = frequency
        self.frequency = frequency
        
    def __set_dt(self):
        """
        Set the temporal resolution based on the wave frequency.
        
        The temporal resolution is set to 1/40th of the wave period
        to ensure sufficient temporal sampling.
        """
        self.dt = 1 / self.frequency / 40
        
    def __set_wavelength(self):
        """
        Calculate the wavelength based on the wave frequency.
        """
        self.wavelength = constant.c / self.frequency
                
    def __set_dx(self):
        """
        Set the spatial resolution based on the wavelength.
        
        The spatial resolution is set to 1/20th of the wavelength
        to ensure sufficient spatial sampling.
        """
        self.dx = self.wavelength / 20
        self.data.dx = self.dx 
        
    def __set_frequency_dependence(self):
        """
        Set all frequency-dependent parameters.
        
        This includes temporal resolution, wavelength, and spatial resolution.
        """
        self.__set_dt()
        self.__set_wavelength()
        self.__set_dx()
        
    def __set_density_field(self, x, y, density):
        """
        Set the plasma density field.
        
        Args:
            x (str or numpy.ndarray): X-axis coordinates
            y (str or numpy.ndarray): Y-axis coordinates
            density (str or numpy.ndarray): Plasma density field
        """
        if isinstance(density, str):
            self.__make_default_density()
        elif isinstance(density, numpy.ndarray):
            self.__fit_density_to_grid(x=x, y=y, density=density)
        else:
            raise(ValueError('Expected a numpy ndarray data type. Input datatype does not match'))            
        
    def __set_spatial_resolutions(self):
        """
        Calculate the number of grid points based on spatial extent and resolution.
        """
        self.nx = int(numpy.abs((self.x[-1] - self.x[0]) // self.dx))
        self.ny = int(numpy.abs((self.y[-1] - self.y[0]) // self.dx))
        self.data.nx = self.nx
        self.data.ny = self.ny
            
    def __make_default_density(self):
        """
        Create a default plasma density profile.
        
        Creates a linear density ramp from 0 to 3e19 m^-3 starting at x=50mm.
        The density is constant along the y-axis.
        """
        self.x = numpy.arange(0,100,1) * 0.001# in m
        self.y = numpy.arange(0,200,1) * 0.001# in m
        self.__set_spatial_resolutions()
        
        profile = numpy.zeros(self.nx)
        profile[50:] = numpy.linspace(0, 3e19, num=(self.nx-50))
        default_density = numpy.array([list(profile) for i in range(int(self.ny))])
        
        ne = (ctypes.POINTER(ctypes.c_double) * self.data.ny)()
        for j in range(self.data.ny):
            ne[j] = (ctypes.c_double * self.data.nx)()
            for i in range(self.data.nx):
                ne[j][i] = default_density[j,i]
        self.data.ne = ne
        
    def __fit_density_to_grid(self, x, y, density):
        """
        Interpolate density data to the simulation grid.
        
        Args:
            x (numpy.ndarray): X-axis coordinates of input density data in m
            y (numpy.ndarray): Y-axis coordinates of input density data in m
            density (numpy.ndarray): Plasma density data in m^-3 [i_y, i_x]
        """
        self.x = x
        self.y = y
        self.__set_spatial_resolutions()
        
        x_grid = x[0] + numpy.arange(int(self.nx))*self.dx
        y_grid = y[0] + numpy.arange(int(self.ny))*self.dx
        interp = RectBivariateSpline(y, x, density)
        interp_density = interp(y_grid, x_grid)

        ne = (ctypes.POINTER(ctypes.c_double) * self.data.ny)()
        for j in range(self.data.ny):
            ne[j] = (ctypes.c_double * self.data.nx)()
            for i in range(self.data.nx):
                ne[j][i] = interp_density[j,i]
        self.data.ne = ne      
        
    def __set_magnetic_field(self, x, y, b_field):
        """
        Set the magnetic field.
        
        Args:
            x (str or numpy.ndarray): X-axis coordinates in m
            y (str or numpy.ndarray): Y-axis coordinates in m
            b_field (str or numpy.ndarray): Magnetic field data in T
        """
        if isinstance(b_field, str):
            self.__make_default_bfield()
        elif isinstance(b_field, numpy.ndarray):
             self.__fit_bfield_to_grid(x=x, y=y, b_field=b_field)
        else:
            raise(ValueError('Expected a numpy ndarray data type. Input datatype does not match'))
            
            
    def __make_default_bfield(self):
        """
        Create a default uniform magnetic field of 2.5 Tesla.
        """
        b0 = (ctypes.POINTER(ctypes.c_double) * self.data.ny)()  # Create an array of pointers (for each row)
        for j in range(self.data.ny):
            b0[j] = (ctypes.c_double * self.data.nx)()  # Create the row with ny elements
            for i in range(self.data.nx):
                b0[j][i] = 2.5
        self.data.b0 = b0
        
    def __fit_bfield_to_grid(self, x, y, b_field):
        """
        Interpolate magnetic field data to the simulation grid.
        
        Args:
            x (numpy.ndarray): X-axis coordinates of input magnetic field data in m
            y (numpy.ndarray): Y-axis coordinates of input magnetic field data in m
            b_field (numpy.ndarray): Magnetic field data in T
        """
        x_grid = x[0] + numpy.arange(int(self.nx))*self.dx
        y_grid = y[0] + numpy.arange(int(self.ny))*self.dx
        interp = RectBivariateSpline(y, x, b_field)
        interp_bfield = interp(y_grid, x_grid)
        
        b0 = (ctypes.POINTER(ctypes.c_double) * self.data.ny)()  # Create an array of pointers (for each row)
        for j in range(self.data.ny):
            b0[j] = (ctypes.c_double * self.data.nx)()  # Create the row with ny elements
            for i in range(self.data.nx):
                b0[j][i] = interp_bfield[j,i]
        self.data.b0 = b0
        
    def __set_angle_antenna(self, angle):
        """
        Set the propagation angle.
        
        Args:
            angle (float): Propagation angle in degrees
        """
        self.angle = angle
        
        
    def __set_simulation_timesteps(self, reflection_distance):
        """
        Calculate the number of temporal iterations based on reflection distance.
        
        Args:
            reflection_distance (str or float): Reflection distance in meters
        """
        if isinstance(reflection_distance, str):
            self.reflection_distance = (self.x[-1]-self.x[0])
        else:
            self.reflection_distance = reflection_distance            
        time = 2*self.reflection_distance / numpy.cos(numpy.radians(self.angle)) / constant.c
        self.__set_timesteps(simulation_time=time*1.05)
            
    def __set_timesteps(self, simulation_time):
        """
        Set the number of temporal iterations.
        
        Args:
            simulation_time (float): Total simulation time in seconds
        """
        self.nt = int(simulation_time // self.dt)
        self.data.nt = self.nt
        
    def __set_beam_waist(self, beam_waist):
        """
        Set the beam waist.
        
        Args:
            beam_waist (str or float): Beam waist in meters or 'default'
        """
        if isinstance(beam_waist, str):
            self.beam_waist_si = 0.03 # in cm
        else:
            self.beam_waist_si = beam_waist
        waist = int(self.beam_waist_si // self.dx)
        return waist

    def __set_antenna_pos(self, antenna_pos):
        """
        Set the antenna position.
        
        Args:
            antenna_pos (str or float): Antenna position in meters or 'default'
        """
        if isinstance(antenna_pos, str):
            self.antenna_pos = 0.05 #in cm
        else:
            self.antenna_pos = antenna_pos
        yante = int(self.ny - (self.antenna_pos - self.y[0]) // self.dx)
        return yante
        
    def __set_solver_and_datastruct(self, wavemode, solver):
        """
        Set up the C solver library.
        
        Args:
            wavemode (str): Wave mode ('O' or 'X')
            solver (str): Solver type
        """
        self.__set_solver_path_and_datastruct(wavemode=wavemode, solver=solver)
        self.fw2d = ctypes.CDLL(self.fw2d_path)
        self.fw2d.maxwell_2d_omode.argtypes = [ctypes.POINTER(InputData)]
        self.fw2d.maxwell_2d_omode.restype = ctypes.c_int
        
    def __set_solver_path_and_datastruct(self, wavemode, solver):
        """
        Set the path to the C solver library.
        
        Args:
            wavemode (str): Wave mode ('O' or 'X')
            solver (str): Solver type
        """
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
            self.data = InputData()
            self.solver = solver
            solver_path = '_ezf'
        elif solver == 'ez_evo':
            self.data = InputData()
            self.solver = solver
            solver_path = '_ezf_time'
        elif solver == 'multi_ant':
            pass
        elif solver == 'multi_evo':
            pass
        else:
            raise ValueError('The requested solver type is not supported. Please consult documentation.')
   
        self.fw2d_path = os.path.join(os.path.dirname(__file__), '..', 
                                      'fw2d', mode_path+solver_path+'.dll')
        
    def __set_outputdata(self, solver):
        """
        Initialize output arrays for antenna amplitude and phase.
        
        Args:
            solver (str): Solver type
        """
        antenna_amplitude = (ctypes.c_double * self.data.nx)()  # 1D array for amplitudes
        antenna_phase = (ctypes.c_double * self.data.nx)()  # 1D array for phases
        for index in range(self.data.nx):
            antenna_amplitude[index] = 1.0
            antenna_phase[index] = 0.0
        self.data.ampl_ant = antenna_amplitude
        self.data.fase_ant = antenna_phase

    def __set_ampl_inc_phase_inc(self, wavesource):
        if wavesource == 'gaussian':
            self.__set_gaussian_wave()
        elif wavesource == 'pyramidal_horn':
            self.__set_pyramidal_horn_wave()
        elif isinstance(wavesource, str):
            raise ValueError(
                "Unknown wavesource. Choose 'gaussian' or 'pyramidal_horn'.")
        else:
            raise TypeError(
                f"Expected str, got {type(wavesource).__name__}.")
    
    def __set_gaussian_wave(self):
        """
        Python translation of the fw2d C aperture initialisation block.

        Computes a Gaussian amplitude profile with a linear phase ramp
        across the antenna plane (j = 0 .. ny).

        Used properties:
        ----------
        ny    : int    last grid index along y
        yante : int    grid index of the beam centre
        waist : float  Gaussian beam waist in grid units
        angle : float  beam steering angle from boresight [rad]
        dx    : float  grid spacing [m]
        f0    : float  frequency [Hz]
        C     : float  speed of light [m/s]  (default 3e8)

        Returns
        -------
        ampl_inc  : ndarray shape (ny+1,)   Gaussian amplitude  [0, 1]
        phase_inc : ndarray shape (ny+1,)   wrapped phase [rad] in (-pi, pi]
        """
        j_arr = numpy.arange(self.ny + 1, dtype=float)
        yante = self.__set_antenna_pos(self.antenna_pos)
        waist = self.__set_beam_waist(self.beam_waist_si)

        # --- Amplitude: Gaussian centred at yante, width = waist/cos(angle) ---
        aux       = numpy.cos(self.angle) * (j_arr - yante) / float(waist)
        ampl_inc  = numpy.exp(-(aux ** 2))

        # --- Phase: linear ramp, step = k*dx*sin(angle) per element ---
        dfase     = 2.0 * numpy.pi * self.frequency / C * self.dx * numpy.sin(self.angle)
        fase      = -j_arr * dfase                 # fase starts at 0 for j=0,
                                                # decrements by dfase each step

        # Wrap to (-pi, pi]  — equivalent to the C loop's wrapping
        phase_inc = (fase + numpy.pi) % (2.0 * numpy.pi) - numpy.pi

        self.data.ampl_inc = ampl_inc
        self.data.phase_inc = phase_inc

    def __set_pyramidal_horn_wave(self):
        """
        Set ampl_inc / phase_inc from the pyramidal horn far-field pattern.

        Horn geometry attributes must be set before calling this method:
            self.horn_a1    : H-plane aperture width [m]
            self.horn_b1    : E-plane aperture height [m]
            self.horn_rho1  : E-plane slant length [m]
            self.horn_rho2  : H-plane slant length [m]
            self.horn_x     : horn x-position in grid units (<=0)
            self.horn_y     : horn y-position in grid units
        """
        ampl_inc, phase_inc = pyramidal_farfield_to_fw2d(
            ny       = self.ny,
            dy       = self.dx,           # same spacing in both directions
            dx       = self.dx,
            x_horn   = self.horn_x,
            y_horn   = self.horn_y,
            angle    = numpy.radians(self.angle),
            a1       = self.horn_a1,
            b1       = self.horn_b1,
            rho1     = self.horn_rho1,
            rho2     = self.horn_rho2,
            freq     = self.frequency,
        )
        self.data.ampl_inc = ampl_inc
        self.data.phase_inc = phase_inc

    def update_frequency(self, frequency):
        """
        Update the wave frequency and recalculate dependent parameters.
        
        Args:
            frequency (float): New wave frequency in Hz
        """
        x_old, y_old, time_old = self.get_axis()
        magnetic = self.get_fields(kind='magnetic')
        density = self.get_fields(kind='density')
        
        self.__set_frequency(frequency=frequency)
        self.__set_frequency_dependence()
        self.__set_density_field(x=x_old, y=y_old, density=density)
        self.__set_magnetic_field(x=x_old, y=y_old, b_field=magnetic)
        self.__set_angle_antenna(angle=self.angle)
        self.__set_simulation_timesteps(reflection_distance=self.reflection_distance)
        self.__set_beam_waist(beam_waist=self.beam_waist_si)
        self.__set_antenna_pos(antenna_pos=self.antenna_pos)        
        
    def update_density(self, density, x, y, reflection_distance='default'):
        """
        Update the plasma density field and recalculate dependent parameters.
        
        Args:
            density (numpy.ndarray): New plasma density data
            x (numpy.ndarray): X-axis coordinates
            y (numpy.ndarray): Y-axis coordinates
            reflection_distance (str or float): Reflection distance or 'default'
        """
        x_old, y_old, time_old = self.get_axis()
        magnetic = self.get_fields(kind='magnetic')
        
        self.__set_density_field(x=x, y=y, density=density)
        self.__set_magnetic_field(x=x_old, y=y_old, b_field=magnetic)
        
        self.__set_angle_antenna(angle=self.angle)
        self.__set_simulation_timesteps(reflection_distance=reflection_distance)
        self.__set_beam_waist(waist=self.beam_waist_si)
        self.__set_antenna_pos(antenna_pos=self.antenna_pos)
            
    def update_angle(self, angle):
        """
        Update the propagation angle and recalculate dependent parameters.
        
        Args:
            angle (float): New propagation angle in degrees
        """
        self.__set_angle_antenna(angle=angle)
        self.__set_simulation_timesteps(reflection_distance=self.reflection_distance)
    
    def update_antenna(self, antenna):
        """
        Update the antenna position and recalculate dependent parameters.
        
        Args:
            antenna (float): New antenna position in meters
        """
        self.__set_antenna_pos(antenna_pos=antenna)
    
    def update_waist(self, waist):
        """
        Update the beam waist and recalculate dependent parameters.
        
        Args:
            waist (float): New beam waist in meters
        """
        self.__set_beam_waist(beam_waist=waist)
    
    def update_solver(self, wavemode, solver):
        """
        Update the solver configuration and recalculate dependent parameters.
        
        Args:
            wavemode (str): New wave mode
            solver (str): New solver type
        """
        self.__set_solver(wavemode=wavemode, solver=solver)
    
    def get_antenna_output(self):
        """
        Get the antenna output amplitude and phase arrays and return them as numpy values.
        
        Returns:
            tuple: (amplitude_values, phase_values) - C arrays containing antenna output
        """
        return self.data.ampl_ant, self.data.fase_ant
    
    def get_fields(self, kind='density'):
        """
        Get the input field data as a numpy array and return it.
        
        Args:
            kind (str): Field type ('density', 'magnetic' or 'electric')
            
        Returns:
            numpy.ndarray: 2D array containing the field data
            
        Raises:
            ValueError: If kind is not 'density' , 'magnetic' or 'electric'
        """
        field = numpy.zeros((self.data.ny, self.data.nx))
        for y_index in range(self.data.ny):
            for x_index in range(self.data.nx):
                if kind == 'density':
                    field[y_index, x_index] = self.data.ne[y_index][x_index]
                elif kind == 'magnetic':
                    field[y_index, x_index] = self.data.b0[y_index][x_index]
                elif kind == 'electric':
                    field[y_index, x_index] = self.data.ez_final[y_index][x_index]
                else:
                    raise ValueError('The requested output type is not supported. Supported types are: <density> or <magnetic>')
        return field
    
    def get_axis(self):
        """
        Get the coordinate axes for the simulation.
        
        Returns:
            tuple: (x_coordinates, y_coordinates, time_coordinates)
                  - x: numpy array of x-coordinates in meters
                  - y: numpy array of y-coordinates in meters  
                  - time: numpy array of time points in seconds
        """
        x = self.x[0] + numpy.arange(self.data.nx)*self.dx
        y = self.y[0] + numpy.arange(self.data.ny)*self.dx
        time = numpy.arange(self.data.nt)*self.dt
        return x, y, time
    
    def plot_results(self, title=''):
        """
        Plot the plasma density field, the magnetic field and the electic field.
        
        Args:
            title (str): Optional title for the plot
        """
        x, y, time = self.get_axis()
        density = self.get_fields()
        ez = self.get_fields(kind='electric')
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        try:
            img_dens = ax[0].contourf(from_unit_to_centi(x), 
                                      from_unit_to_centi(y), 
                                      density, levels=200, cmap='plasma')
        except TypeError:
            img_dens = ax[0].contourf(from_unit_to_centi(x), 
                                      from_unit_to_centi(y),
                                      density.T, levels=200, cmap='plasma')
        ax[0].set_title("Density field"+title, fontsize=14, fontweight = 'bold')
        ax[0].tick_params(axis='both', labelsize= 12)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xlabel('X axis [cm]', fontsize=14, fontweight = 'bold')
        ax[0].set_ylabel('Y axis [cm]', fontsize=14, fontweight = 'bold')
        
        col = fig.colorbar(img_dens, ax=ax[0])
        col.ax.tick_params(labelsize= 12, which='both')
        col.ax.set_ylabel('Density [m-3]',fontsize=12, fontweight = 'bold')
        
        try:
            img_ez = ax[1].contourf(from_unit_to_centi(x), 
                                    from_unit_to_centi(y),
                                    ez, levels=200, cmap="RdBu_r",
                                    vmin=-numpy.max(ez), vmax=numpy.max(ez))
        except TypeError:
            img_ez = ax[1].contourf(from_unit_to_centi(x), 
                                    from_unit_to_centi(y),
                                    ez.T, levels=200, cmap="RdBu_r",
                                    vmin=-numpy.max(ez), vmax=numpy.max(ez))
        ax[1].set_title("Electric field"+title, fontsize=14, fontweight = 'bold')
        ax[1].tick_params(axis='both', labelsize= 12)
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_xlabel('X axis [cm]', fontsize=14, fontweight = 'bold')
        
        col = fig.colorbar(img_ez, ax=ax[1])
        col.ax.tick_params(labelsize= 12, which='both')
        col.ax.set_ylabel('Electric Field [V/m]',fontsize=12, fontweight = 'bold')
        
        
        plt.tight_layout()
        plt.show()

    def print_aperture(ampl_inc, phase_inc, ny, yante, max_rows=10):
        """
        Print a summary of the aperture arrays so you can cross-check with the C output.

        Prints:
        - key scalar values (dfase, first/last phase step)
        - a table of j, ampl, phase for:
            first max_rows//2 elements
            the region around yante (beam centre)
            last  max_rows//2 elements
        """
        n = len(ampl_inc)
        half = max_rows // 2

        print("=" * 58)
        print(f"  ny={ny}   yante={yante}   total points={n}")
        print(f"  ampl range : [{ampl_inc.min():.6f}, {ampl_inc.max():.6f}]")
        print(f"  phase range: [{numpy.degrees(phase_inc.min()):.2f}°,"
            f" {numpy.degrees(phase_inc.max()):.2f}°]")
        print(f"  ampl  at yante : {ampl_inc[yante]:.6f}  (should be 1.0)")
        print(f"  phase at j=0   : {numpy.degrees(phase_inc[0]):.4f}°  (should be 0.0)")
        print("-" * 58)
        print(f"  {'j':>5}  {'ampl':>10}  {'phase [deg]':>12}  {'phase [rad]':>12}")
        print("-" * 58)

        # indices to print: start, around yante, end
        indices = sorted(set(
            list(range(0, min(half, n))) +
            list(range(max(0, yante - half//2), min(n, yante + half//2 + 1))) +
            list(range(max(0, n - half), n))
        ))

        prev = -1
        for j in indices:
            if prev >= 0 and j > prev + 1:
                print(f"  {'...':>5}")
            print(f"  {j:>5}  {ampl_inc[j]:>10.6f}  "
                f"{numpy.degrees(phase_inc[j]):>12.4f}  {phase_inc[j]:>12.6f}")
            prev = j

        print("=" * 58)


# --- Example run ---
freq  = 10.0e9
lam   = 3e8 / freq
ny    = 200
yante = 100
waist = 30
dx    = 0.5 * lam
angle = numpy.radians(20.0)

ampl, phase = Basic.__set_gaussian_aperture(ny, yante, waist, angle, dx, freq)
print_aperture(ampl, phase, ny, yante, max_rows=12)

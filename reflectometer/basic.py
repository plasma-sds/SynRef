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
import datetime
import platform


from reflectometer.conversions import from_unit_to_centi
import scipy.constants as constant
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from hardware.utils.antenna.pyramidal_farfield_to_fw2d import pyramidal_farfield_to_fw2d
from hardware.components.antenna import Antenna, HORN_PRESETS
from reflectometer.conversions import antenna_pos_to_unit, _build_extended_incident_arrays

NXPML = 8
TFSF = NXPML + 10  # 18

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
        ("ampl_inc", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),                                      # Amplitude array of the incident wave [-]
        ("phase_inc", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),                                     # Phase array of the incident wave [-]
        ("b0", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),            # Magnetic field
        ("ne", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),            # Plasma density field
        ("ez_final", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
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
                reflection_distance='default', wavesource='default',
                horn='default'):
        """
        Initialize the Basic FW2D simulation.

        Args:
            wavemode (str): Wave propagation mode. 'O' for O-mode (ordinary),
                'X' for X-mode (extraordinary). Default 'O'.
            solver (str): Solver variant to load. One of 'basic', 'ez_evo',
                'multi_ant', 'multi_evo', or 'antenna'. Determines which
                compiled .so backend and struct layout are used. Default 'basic'.
            frequency (float): Probing wave frequency in Hz. Default 3e10 (30 GHz).
            density (str or numpy.ndarray): Plasma density field ne(x, y) in m^-3.
                Pass 'default' to use a built-in preset profile, or provide a
                2D numpy array matching the (ny, nx) grid shape.
            b_field (str or numpy.ndarray): Background magnetic field in Tesla.
                Pass 'default' for a built-in preset, or a 2D numpy array
                matching the (ny, nx) grid shape.
            x (str or numpy.ndarray): X-axis grid coordinates in meters.
                Pass 'default' to auto-generate from nx and dx, or provide
                a 1D numpy array explicitly.
            y (str or numpy.ndarray): Y-axis grid coordinates in meters.
                Pass 'default' to auto-generate from ny and dx, or provide
                a 1D numpy array explicitly.
            antenna_pos (str or float): Antenna position along the Y-axis,
                in meters. Pass 'default' to use a built-in default position,
                or a float to set it explicitly.
            beam_waist (str or float): Beam waist (spot size) in meters,
                used to shape the Gaussian source amplitude taper.
                Pass 'default' for a built-in value, or a float to override.
            angle (float): Antenna/beam boresight elevation angle, in degrees,
                measured from the horizontal (X) axis. Default 0 (broadside).
            reflection_distance (str or float): Expected distance in meters
                to the reflecting plasma layer, used to size the simulation
                time window. Pass 'default' to auto-estimate it, or a float
                to set it explicitly.
            wavesource (str): Source field model used to compute ampl_inc and
                phase_inc. One of 'default' (analytic Gaussian beam with
                linear phase ramp) or 'pyramidal_horn' (far-field pattern of
                a pyramidal horn antenna, via Fresnel-integral computation).
                Default 'default'.
            horn (str): Horn parameters called from a dictionary. 'Default'
                calls a generic pyramidal antenna, possible antennas: 'W7X' etc.
        """
        
        self._bufs = {}
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
        self.__set_ezfinal_output()

        
        self.antenna = Antenna.from_horn_preset(horn)
        self.horn_x  = HORN_PRESETS[horn]['x']
        self.E1  = HORN_PRESETS[horn]['E1']

        self.__set_ampl_inc_phase_inc(wavesource=wavesource)
    
        self._all_buffers = [
        self.data.ne,
        self.data.b0,
        self.data.ampl_inc,
        self.data.phase_inc,
        #self.data.ez_final,
        self.data.ampl_ant,
        self.data.fase_ant,
        ]
        print("Buffer ids:", [id(b) for b in self._all_buffers])

        print("Buffer addresses:")
        for k, v in self._bufs.items():
            try:
                addr = ctypes.addressof(v[0])
            except TypeError:
                addr = ctypes.addressof(v)
            print(f"  {k:12s}: id={id(v)}  ctypes_addr={addr}")

        print("Python struct size:", ctypes.sizeof(InputData)) 

        ny_phys = self.data.ny          # physical grid points (e.g. 132)
        ny_ext  = ny_phys - 1 + 2*TFSF  # extended grid in C

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
        self.x = numpy.arange(0,200,1) * 0.001# in m
        self.y = numpy.arange(0,400,1) * 0.001# in m
        self.__set_spatial_resolutions()
        
        profile = numpy.zeros(self.nx)
        profile[50:] = numpy.linspace(0, 3e19, num=(self.nx-50))
        default_density = numpy.array([list(profile) for i in range(int(self.ny))])
        
        ne = (ctypes.POINTER(ctypes.c_double) * self.data.ny)()
        for j in range(self.data.ny):
            ne[j] = (ctypes.c_double * self.data.nx)()
            for i in range(self.data.nx):
                ne[j][i] = default_density[j,i]
        self._bufs['ne'] = ne
        self.data.ne = self._bufs['ne']
        
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
        self._bufs['ne'] = ne
        self.data.ne = self._bufs['ne']  
        
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
        self._bufs['b0'] = b0
        self.data.b0 = self._bufs['b0']
        
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
        self._bufs['b0'] = b0
        self.data.b0 = self._bufs['b0']
        
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
            self.antenna_pos = 0.005 #in meters
        else:
            self.antenna_pos = antenna_pos
        yante = antenna_pos_to_unit(self.antenna_pos, self.y[0], self.ny, self.dx)
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
        elif solver == 'antenna':
            self.data = InputData()
            self.solver = solver
            solver_path = '_antenna'
        elif solver == 'multi_ant':
            pass
        elif solver == 'multi_evo':
            pass
        else:
            raise ValueError('The requested solver type is not supported. Please consult documentation.')


        system = platform.system()

        if system == "Windows":
            ext_name = ".dll"
        elif system == "Linux":
            ext_name = ".so"
        else:
            raise OSError(f"Unsupported OS: {system}")

        self.fw2d_path = os.path.join(os.path.dirname(__file__), '..', 
                                      'fw2d', mode_path+solver_path+ext_name)
        
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

        self._bufs['ampl_ant']  = antenna_amplitude
        self._bufs['fase_ant']  = antenna_phase
        self.data.ampl_ant  = self._bufs['ampl_ant']
        self.data.fase_ant  = self._bufs['fase_ant']

    def __set_ampl_inc_phase_inc(self, wavesource):
        if wavesource == 'default':
            self.__set_gaussian_wave()
        elif wavesource == 'pyramidal_horn':
            self.__set_pyramidal_horn_wave()
        elif isinstance(wavesource, str):
            raise ValueError(
                "Unknown wavesource. Choose 'default' or 'pyramidal_horn'.")
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
        yante : int    grid index of thebeam center
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
        j_phys = numpy.arange(self.data.ny + 1, dtype=float)
        yante  = self.__set_antenna_pos(self.antenna_pos)
        waist  = self.__set_beam_waist(self.beam_waist_si)

        # Amplitude on PHYSICAL grid (0 .. ny_phys), as before
        aux      = numpy.cos(numpy.deg2rad(self.angle)) * (j_phys - yante) / float(waist)
        ampl_phys  = numpy.exp(-(aux ** 2))

        # Phase on PHYSICAL grid (0 .. ny_phys)
        dfase      = 2.0 * numpy.pi * self.frequency / constant.c * self.dx * numpy.sin(numpy.deg2rad(self.angle))
        fase       = -j_phys * dfase
        phase_phys = (fase + numpy.pi) % (2.0 * numpy.pi) - numpy.pi

        _, _ = _build_extended_incident_arrays(self=self, ampl_phys=ampl_phys, phase_phys=phase_phys, TFSF=TFSF)


    def __set_pyramidal_horn_wave(self):
        """
        Set ampl_inc / phase_inc from the pyramidal horn far-field pattern.

        Horn geometry attributes must be set before calling this method:
            self.horn_a1    : H-plane aperture width [m]
            self.horn_b1    : E-plane aperture height [m]
            self.horn_rho1  : E-plane slant length [m]
            self.horn_rho2  : H-plane slant length [m]
            self.horn_x     : horn x-position in [m] (<=0)
            self.horn_y     : horn y-position in [m]
        """
        yante = self.__set_antenna_pos(self.antenna_pos)

        ampl_phys, phase_phys = pyramidal_farfield_to_fw2d(
            y        = self.y,
            ny       = self.ny,
            dy       = self.dx,           # same spacing in both directions
            dx       = self.dx,
            x_horn   = self.horn_x,
            antenna_pos   = self.antenna_pos,
            yante    = yante,
            angle    = numpy.deg2rad(self.angle),
            a1       = self.antenna.a1,
            b1       = self.antenna.b1,
            rho1     = self.antenna.rho1,
            rho2     = self.antenna.rho2,
            freq     = self.frequency,
            E1       = self.E1,
        )
        
        _, _ = _build_extended_incident_arrays(self=self, ampl_phys=ampl_phys, phase_phys=phase_phys, TFSF=TFSF)

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
    
    def __set_ezfinal_output(self):
        """
        Initializes the ez_final memory allocation
        
        Args:
            None
        """
        ez_final = (ctypes.POINTER(ctypes.c_double) * self.data.ny)()  # Create an array of pointers (for each row)
        # Store references to prevent garbage collection
        self._ez_final_arrays = []
        for j in range(self.data.ny):
            row_data = (ctypes.c_double * self.data.nx)()  # Create the row with nx elements
            for i in range(self.data.nx):
                row_data[i] = 0.0  # Initialize to zero
            ez_final[j] = row_data  # Assign the row to the pointer array
            self._ez_final_arrays.append(row_data)  # Store reference to prevent GC
        self.data.ez_final = ez_final

  
    def run(self):
        """
        Execute the Maxwell 2D simulation by calling the C function.
        
        This method runs the FDTD simulation and populates the ez_final field
        with the computed electric field values.
        
        Returns:
            int: Return code from the C function (0 on success)
        """
        return self.fw2d.maxwell_2d_omode(ctypes.byref(self.data))
    
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

        # --- Overlay antenna boresight line on both panels ---
        ante = True
        if ante:
            yante = self.__set_antenna_pos(self.antenna_pos)

            # Horn's true physical position (horn_x is negative, i.e. behind the domain)
            x0_cm = from_unit_to_centi(self.horn_x)
            y0_cm = from_unit_to_centi(self.antenna_pos)

            # Extend the boresight ray from the horn position across the domain's X-range
            x_line = numpy.array([self.horn_x, x[-1]])
            # angle is boresight elevation; y = y0 + (x - x0) * tan(angle), now anchored at horn_x
            y_line = self.antenna_pos + (x_line - self.horn_x) * numpy.tan(numpy.deg2rad(self.angle))

            x_line_cm = from_unit_to_centi(x_line)
            y_line_cm = from_unit_to_centi(y_line)

            for a in ax:
                a.plot(x_line_cm, y_line_cm, color='lime', linewidth=2,
                    linestyle='--', label='Antenna boresight')
                a.plot(x0_cm, y0_cm, marker='o', color='lime', markersize=8,
                    markeredgecolor='black')
                a.legend(loc='upper right', fontsize=10)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.tight_layout()
        plt.savefig(f"plots/antenna_wave_{timestamp}.png")

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:55:33 2025

@author: akosk
"""
import numpy as np
import sys
from pathlib import Path
from os.path import dirname
from os.path import join
from os import mkdir
from pywt import cwt as pywt_cwt
from pywt import central_frequency as pywt_fc
from pywt import scale2frequency as pywt_s2f
sys.path.append(dirname(__file__))

from reflectometer.basic import Basic
import reflectometer.functions as func
import file_management as fm


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
             "field_height": np.array([-150e-3 + 16.24e-3, 150e-3 + 16.24e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015},
        "test":
            {"wavemode": 'O', "solver": 'basic', 
             "reflection_distance": 'default',
             "frequency":    np.arange(25, 40 + 0.1, 1) *1e9, # [GHz]
             "field_width":  np.array([0, 100e-3]), 
             "field_height": np.array([-150e-3 , 150e-3]),
             "antenna_pos":  np.array([2.0441, 6.2912, -0.146]), 
             "LOS":          np.array([18, -3.7, 0]),
             "beam_waist": 0.015} }
    
    def __init__(self, library = "W7X QMR-V1", time = "default",
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
        self.time = time
        
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
        y_abs = np.linspace(z + Ly[0] , z + Ly[1],  len(self.y) )
        
        X_abs = np.linspace(r - Lx[0] , r - Lx[1],      self.nx )
        Y_abs = np.linspace(z + Ly[0] , z + Ly[1],      self.ny )
        
        x = np.linspace(0, Lx[1] - Lx[0], len(self.x) )
        y = np.linspace(0, Ly[1] - Ly[0], len(self.y) )
        
        X = np.linspace(0, Lx[1] - Lx[0],    self.nx  )
        Y = np.linspace(0, Ly[1] - Ly[0],    self.ny  )
        
        self.setup_map = {"x": x, "y": y, "x_abs": x_abs, "y_abs": y_abs,
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
    
    
    def single_run(self, frequency_index = "default", 
                   density_index = "default",
                   con_filename = False, path = "default"):
        if con_filename != False: 
            path = join(path, "config_files")
            try: mkdir(path)
            except: pass
        
        if path == "default": path = self.path
        
        # default settings
        if frequency_index == "default": frequency_index = 0
        if density_index == "default": density_index = 0
            
        # initialize
        self.update_density(self.density_evolution[density_index],
                            reflection_distance='default')
        self.update_frequency(self.library["frequency"][frequency_index])
        
        # filename of configuration file
        if con_filename == True: 
            con_filename = "config_{dens:04d}_{freq:03d}.json"
            
        # run sweeps
        self.amplitudes, self.phases = func.get_full_sweep(
            self, self.density_evolution, self.library["frequency"],
            [density_index], [frequency_index],
            con_filename=con_filename, path=path)
        
    
    def frequency_sweep(self, frequency_indices = "default", 
                        density_index = "default",
                        con_filename = False, path = "default"):
        if path == "default": path = self.path
        if con_filename != False: 
            path = join(path, "config_files")
            try: mkdir(path)
            except: pass
        
        # default settings
        if frequency_indices == "default":
            frequency_indices = np.arange(self.library["frequency"].shape[0])
        if density_index == "default": density_index = 0
        
        # initialize
        self.update_density(self.density_evolution[density_index],
                            reflection_distance='default')
        self.update_frequency(self.library["frequency"][frequency_indices[0]])
        
        # filename of configuration file
        if con_filename == True: 
            con_filename = "config_{dens:04d}_{freq:03d}.json"
        
        # run sweeps
        self.amplitudes, self.phases = func.get_full_sweep(
            self, self.density_evolution, self.library["frequency"],
            [density_index], frequency_indices,
            con_filename=con_filename, path=path)
            
        
    def density_sweep(self, frequency_index = "default", 
                      density_indices = "default",
                      con_filename = False, path = "default"):
        if path == "default": path = self.path
        if con_filename != False: 
            path = join(path, "config_files")
            try: mkdir(path)
            except: pass
        
        # default settings
        if frequency_index == "default": frequency_index = 0
        if density_indices == "default": 
            density_indices = np.arange(self.density_evolution.shape[0])
        
        # initialize
        self.update_density(self.density_evolution[density_indices[0]],
                            reflection_distance='default')
        self.update_frequency(self.library["frequency"][frequency_index])
        
        # filename of configuration file
        if con_filename == True: 
            con_filename = "config_{dens:04d}_{freq:03d}.json"
        
        # run sweeps
        self.amplitudes, self.phases = func.get_full_sweep(
            self, self.density_evolution, self.library["frequency"],
            density_indices, [frequency_index],
            con_filename=con_filename, path=path)
        
    
    def full_sweep(self, frequency_indices = "default", 
                   density_indices = "default",
                   con_filename = False, path = "default"):
        
        if path == "default": path = self.path
        if con_filename != False: 
            path = join(path, "config_files")
            try: mkdir(path)
            except: pass
        
        # default settings
        if frequency_indices == "default": 
            frequency_indices = np.arange(len(self.library["frequency"]))
        if density_indices == "default": 
            density_indices = np.arange(self.density_evolution.shape[0])
        
        # initialize
        self.update_density(self.density_evolution[density_indices[0]])
        self.update_frequency(self.library["frequency"][frequency_indices[0]])
        
        # filename of configuration file
        if con_filename == True: 
            con_filename = "config_{dens:04d}_{freq:03d}.json"
        
        
        # run sweeps
        self.amplitudes, self.phases = func.get_full_sweep(
            self, self.density_evolution, self.library["frequency"],
            density_indices, frequency_indices,
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
    



class Doppler_signal():
    
    
    def __init__(self, working_directory):
        self.path = Path(working_directory)
        
    
    def __create_complex_signal(self):
        self.complex_signal = (self.data["amplitude_field"] 
                               * np.exp(1j * self.data["phase_field"]))
        self.complex_sig_rel = (self.complex_signal
                                - np.average(self.complex_signal, axis = 0))
        
        
    def read(self, signal_filename = "signal_field.h5"):
        
        self.signal_filename = signal_filename
        data = fm.read_ref_signal(self.path, signal_filename=signal_filename)
        self.data = data
        self.__create_complex_signal()
        
        
    def save(self, signal_filename = "signal_field.h5", 
             config_filename = "config_{dens:04d}_{freq:03d}.json",
             config_path = "default"):
        
        self.signal_filename = signal_filename
        if (config_path == "default"): config_path = self.path
        
        fm.save_ref_signal(config_path, filename=config_filename,
                           signal_path=self.path, 
                           signal_filename=signal_filename,
                           time = "default")
        
        
    def analyze(self, scale = "default", frequency_indices = "all",
                wavelet = 'cmor1.5-1.0'):
        
        t = self.data["time"]
        f = self.data["frequencies"] # antenna frequencies
        fc = pywt_fc(wavelet) # wavelet reference frequency != antenna freq
        dt = t[1]-t[0]
        
        if frequency_indices == "all":
            frequency_indices = np.arange(len(f))
        
        if scale == "default": 
            scale = np.arange(2, len(t)/3)
            
        freq = pywt_s2f(wavelet, scale) / dt
        period = 1 / freq
        
        cwt = np.zeros((len(frequency_indices), len(scale), len(t)), 
                       dtype = np.complex128)
        
        for i, f_ind in enumerate(frequency_indices):
            cwt[i, :, :], _ = pywt_cwt(self.complex_sig_rel[:, f_ind],
                            scale, wavelet, sampling_period = dt)
            
        cwt_data = {"cwt_matrix": cwt,   # c onvolution matrix of the wavelets
                    "scale": scale,      #      relative scale of the wavelets
                    "freq": freq,        #           frequency of the wavelets
                    "period": period,    #         time period of the wavelets
                    "time": t,           # time
                    "fc": fc,            # reference frequency of the wavelets
                    "wavelet": wavelet,  #                type of the wavelets
                    
                    "frequency_indices": frequency_indices,
                    "frequencies": f[frequency_indices]
                    # antenna frequencies
                    }
        
        self.cwt_data = cwt_data
        
        self.__find_event()
        
        
    def __find_event(self):
        from scipy.optimize import curve_fit
        
        n_f = len(self.cwt_data["frequency_indices"])
        keywords = ["A", "x0", "y0", "sigma_x", "sigma_y", 
                    "theta", "offset", "asym_x", "asym_y"]
        p0   = np.zeros((n_f, len(keywords)))
        popt = np.zeros((n_f, len(keywords)))
        pcov = np.zeros((n_f, len(keywords), len(keywords)))
        
        t, T = self.cwt_data["time"], self.cwt_data["period"]
        gauss_field = np.zeros((n_f, len(T), len(t)))
        FWHM_inds = np.zeros((n_f, 2), dtype= np.int16)
        
        for i in range(n_f):
            Z = np.abs(self.cwt_data["cwt_matrix"][i, :, :])
            coords = np.meshgrid(t, T)
            max_T, max_t = np.unravel_index(Z.argmax(), Z.shape)
            # print(Z.shape, X.shape, Y.shape, t.shape, T.shape)
            
            # Initial guess parameters
            guess_A = Z.max() - Z.min()
            guess_x0 = t[max_t]
            guess_y0 = T[max_T]
            guess_sigma_x = guess_y0 / 6 # 6 sigma can cover the 99.7%
            guess_sigma_y = 0.2 *(T[-1] - T[0]) / 6 
            guess_theta = 0
            guess_offset = Z.min()
            guess_asym_x = guess_asym_y = 1.0
            
            p0 = [guess_A, guess_x0, guess_y0, 
                  guess_sigma_x, guess_sigma_y, guess_theta,
                  guess_offset, guess_asym_x, guess_asym_y]
            
            # Fit
            popt[i, :], pcov[i, :] = curve_fit(self.__gaussian_2d, coords,
                                               Z.ravel(), p0=p0)
            
            gauss = self.__gaussian_2d(
                coords,  **dict(zip(keywords, popt[i, :]))
                ).reshape((len(T), len(t)))
            
            half_max = popt[i, 0] / 2 # half of the amplitude
            
            FWHM_inds[i, 0] = np.where(np.max(gauss, axis=0) > half_max)[0][ 0]
            FWHM_inds[i, 1] = np.where(np.max(gauss, axis=0) > half_max)[0][-1]
            
            gauss_field[i, :, :] = gauss
        
        self.cwt_data.update({"gauss_p0": p0, "gauss_popt": popt,
                              "gauss_pcov": pcov, "gauss_keywords": keywords,
                              "gauss_fit": gauss_field})
        self.event = {"A": popt[:, 0], "x0": popt[:, 1], "y0": popt[:, 2],
                      "FWHM_inds": FWHM_inds, 
                      "FWHM_lims": self.cwt_data["time"][FWHM_inds] }
        
    
    def __gaussian_2d(self, coords, A=1, x0=0, y0=0, sigma_x=1, sigma_y=1,
                      theta=0, offset=0, asym_x=1.0, asym_y=1.0):
        
        # Convert scalars to arrays for unified handling
        X = np.asarray(coords[0], dtype=float)
        Y = np.asarray(coords[1], dtype=float)

        xr = np.cos(theta) * (X-x0) + np.sin(theta) * (Y-y0)
        yr = -np.sin(theta) * (X-x0) + np.cos(theta) * (Y-y0)

        # Apply asymmetry (different sigma on + and - sides)
        sigma_x_eff = np.where(xr >= 0, sigma_x * asym_x, sigma_x / asym_x)
        sigma_y_eff = np.where(yr >= 0, sigma_y * asym_y, sigma_y / asym_y)

        # Gaussian formula
        gauss = A * np.exp(-0.5 * (
            (xr / sigma_x_eff)**2 + (yr / sigma_y_eff)**2) )

        return (offset + gauss).ravel()
            
        
        
        
            



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
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann  # or another window
from scipy.ndimage import maximum_filter, label
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
        
        Antenna_complex_V = (self.Antenna_data["amplitude_field"] 
                               * np.exp(1j * self.Antenna_data["phase_field"]))
        self.Antenna_complex_V = (
            Antenna_complex_V - np.average(Antenna_complex_V, axis = 0))
        self.Antenna_time = self.Antenna_data["time"]
        self.Antenna_freq = self.Antenna_data["frequencies"]
        
    def read(self, signal_filename = "signal_field.h5"):
        
        self.signal_filename = signal_filename
        self.Antenna_data = fm.read_ref_signal(
            self.path, signal_filename = signal_filename)
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
        
        
    def analyze_cwt(self, scale = "default", frequency_indices = "all",
                    wavelet = 'cmor1.5-1.0'):
        
        t = self.Antenna_time
        f = self.Antenna_freq
        fc = pywt_fc(wavelet) # wavelet reference frequency != antenna freq
        dt = t[1]-t[0]
        
        if frequency_indices == "all":
            frequency_indices = np.arange(len(f))
        
        if scale == "default": 
            scale = np.arange(8, len(t)*2)
        
        nf = len(scale)
        freq = fc/scale/dt
        # print(freq)
        freq_full = np.concatenate([-freq, freq[::-1]])
        
        cwt = np.zeros((len(frequency_indices), nf*2, len(t)), 
                       dtype = np.complex128)
        
        for i, f_ind in enumerate(frequency_indices):
            V = self.Antenna_complex_V[:, f_ind]
            cwt_pos, _ = pywt_cwt(V, scale, wavelet, 
                                         sampling_period = dt)
            cwt_neg, _ = pywt_cwt(np.conj(V), scale, wavelet, 
                                         sampling_period = dt)
            cwt_full = np.vstack([cwt_neg, cwt_pos[::-1]])
            cwt[i, :, :] = cwt_full
            
        cwt_data = {
            "matrix": cwt,   # c onvolution matrix of the wavelets
            "freq": freq_full,        #           frequency of the wavelets
            "time": t,           # time
            
            "frequency_indices": frequency_indices,
            "frequencies": f[frequency_indices] # antenna frequencies
                    }
        
        self.cwt_data = cwt_data
        
        self.cwt_peaks = self.find_and_refine_peaks(t, freq_full, np.abs(cwt))
        
        # gauss_fit = self.__find_event(cwt_data)
        # self.cwt_gauss = gauss_fit
        
        
    def analyze_stft(self, frequency_indices = "all", res_fft = "default",
                     window = "default", overlap_percentage = "default"):
        
        t = self.Antenna_time
        f = self.Antenna_freq
        dt = t[1]-t[0]
        
        if frequency_indices == "all": frequency_indices = np.arange(len(f))
        if window == "default": window = int(len(t)/5)
        if overlap_percentage == "default": overlap_percentage = 0.5
        if res_fft == "default": res_fft = len(t)*4
        
        SFT = ShortTimeFFT(hann(window), hop = int(overlap_percentage*window),
                           fs = 1/dt, fft_mode = 'centered', mfft = res_fft,
                           scale_to='magnitude')
        freq = SFT.f
        times = SFT.t(len(t))
        
        stft = np.zeros((len(frequency_indices), len(freq), len(times)), 
                        dtype = np.complex128)
        
        for i, f_ind in enumerate(frequency_indices):
            V = self.Antenna_complex_V[:, f_ind]
            stft[i, :, :] = SFT.stft(V)
            
        stft_data = {
            "matrix": stft,      #      Convolution matrix of the STFT
            "freq": freq,        #                     frequency range
            "time": times,       # time
            
            "frequency_indices": frequency_indices,
            "frequencies": f[frequency_indices] # antenna frequencies
                    }
        
        self.stft_data = stft_data
        
        self.stft_peaks = self.find_and_refine_peaks(times, freq, np.abs(stft))
        
        # gauss_fit = self.__find_event(stft_data)
        # self.stft_gauss = gauss_fit
    

        

    def find_and_refine_peaks(self, x, y, Z, 
                              halfsize="default", threshold="default"):
        """
        Find and refine 2D peaks using centroid interpolation.
    
        Parameters
        ----------
        x, y : 1D arrays
            Grid coordinates (lengths nx, ny).
        Z : 2D array (shape (ny, nx))
            Field values defined on the grid.
        halfsize : int, optional
            Half-size of the local window around each detected peak (default=2).
        threshold : float, optional
            Minimum value of Z to consider as a valid peak.
    
        Returns
        -------
        peaks : list of dict
            Each element: {'x': float, 'y': float, 'z': float}
            containing subgrid coordinates and interpolated peak value.
        """
        nf, ny, nx = Z.shape
        
        peak_list = []
        for i in range(nf):
            z = Z[i,:,:]
            if halfsize=="default": halfsize = 2
            if threshold=="default": 
                trshld = (np.max(z) - np.min(z))*0.3 + np.min(z)
            else: trshld = (np.max(z) - np.min(z))*threshold + np.min(z)
            
            window_size_y = window_size_x = int((nx*ny)**0.5 / 12)
            
            # --- Step 1. Find local maxima (integer grid peaks)
            neighborhood = np.ones((window_size_y, window_size_x))
            local_max = (z == maximum_filter(z, footprint=neighborhood))
            if threshold is not None:
                local_max &= (z >= trshld)
        
            labeled, num = label(local_max)
        
            peaks = []
            for i in range(1, num + 1):
                coords = np.argwhere(labeled == i)
                if coords.size == 0:
                    continue
                # pick pixel with max value in this region
                r, c = coords[np.argmax(z[coords[:, 0], coords[:, 1]])]
        
                # --- Step 2. Extract patch around the peak
                r0, r1 = max(0, r - halfsize), min(ny, r + halfsize + 1)
                c0, c1 = max(0, c - halfsize), min(nx, c + halfsize + 1)
                patch = z[r0:r1, c0:c1].astype(float)
        
                # Coordinates in physical units
                yy, xx = np.meshgrid(y[r0:r1], x[c0:c1], indexing='ij')
        
                # --- Step 3. Centroid refinement
                patch -= np.min(patch)  # background subtraction
                patch[patch < 0] = 0
                total = patch.sum()
                if total == 0:
                    continue
                y_c = (yy * patch).sum() / total
                x_c = (xx * patch).sum() / total
                # --- Step 4. Interpolate z (bilinear)
                # Find surrounding grid indices
                ix = np.searchsorted(x, x_c) - 1
                iy = np.searchsorted(y, y_c) - 1
                ix = np.clip(ix, 0, nx-2)
                iy = np.clip(iy, 0, ny-2)
                
                # Bilinear interpolation
                x1, x2 = x[ix], x[ix+1]
                y1, y2 = y[iy], y[iy+1]
                Q11, Q12 = z[iy, ix], z[iy+1, ix]
                Q21, Q22 = z[iy, ix+1], z[iy+1, ix+1]
    
                z_c = (
                    Q11 * (x2 - x_c) * (y2 - y_c) +
                    Q21 * (x_c - x1) * (y2 - y_c) +
                    Q12 * (x2 - x_c) * (y_c - y1) +
                    Q22 * (x_c - x1) * (y_c - y1)
                    ) / ((x2 - x1) * (y2 - y1))
        
                peaks.append([x_c, y_c, float(z[r, c])])
            peaks = np.asarray(peaks)
            peaks = peaks[peaks[:, 2].argsort(), :][::-1, :]
            peak_list.append(peaks)
        return peak_list

        
    # OBSOLETE GAUSSIAN FIT - unreliable even with good initial parameters
    def __find_event(self, data):
        from scipy.optimize import curve_fit
        
        n_f = len(data["frequency_indices"])
        keywords = ["A", "x0", "y0", "sigma_x", "sigma_y", 
                    "theta", "offset", "asym_x", "asym_y"]
        p0   = np.zeros((n_f, len(keywords)))
        popt = np.zeros((n_f, len(keywords)))
        pcov = np.zeros((n_f, len(keywords), len(keywords)))
        
        t, f = data["time"], data["freq"]
        # gauss_field = np.zeros((n_f, len(T), len(t)))
        FWHM_inds = np.zeros((n_f, 2), dtype= np.int16)
        
        for i in range(n_f):
            Z = np.abs(data["matrix"][i, :, :])
            coords = np.meshgrid(t, f)
            max_f, max_t = np.unravel_index(Z.argmax(), Z.shape)
            # print(Z.shape, X.shape, Y.shape, t.shape, T.shape)
            
            # Initial guess parameters
            guess_A = Z.max() - Z.min()
            guess_x0 = t[max_t]
            guess_y0 = f[max_f]
            guess_sigma_x = guess_y0 / 6 # 6 sigma can cover the 99.7%
            guess_sigma_y = 0.1 *(f[-1] - f[0]) / 6
            guess_theta = 0
            guess_offset = Z.min()
            guess_asym_x = guess_asym_y = 1.0
            
            p0[i, :] = [guess_A, guess_x0, guess_y0, 
                        guess_sigma_x, guess_sigma_y, guess_theta,
                        guess_offset, guess_asym_x, guess_asym_y]
            
            # Fit
            try:
                popt[i, :], pcov[i, :] = curve_fit(
                    self.__gaussian_2d, coords, Z.ravel(), p0=p0[i, :])
            except:
                print("no peaks found")
                continue
            
            gauss = self.__gaussian_2d(
                coords,  **dict(zip(keywords, popt[i, :]))
                ).reshape((len(f), len(t)))
            
            half_max = popt[i, 0] / 2 # half of the amplitude
            
            try:
                FWHM_inds[i, 0] = np.where(np.max(gauss, axis=0) > half_max)[0][ 0]
                FWHM_inds[i, 1] = np.where(np.max(gauss, axis=0) > half_max)[0][-1]
            except:
                continue
            # gauss_field[i, :, :] = gauss
        
        
        gauss_fit = {"keywords": keywords, 
                     "freqs": data["frequencies"][data["frequency_indices"]],
                     "p0": p0, 
                     "popt": popt,
                     "FWHM_inds": FWHM_inds}
                            # "gauss_pcov": pcov,
                            # "gauss_fit": gauss_field,
                          
        return gauss_fit
        
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
    
    def write_cwt(self, filename_cwt = "event_data_CWT.json"):
        
        fa      = (self.cwt_data["frequencies"] * 1.0).tolist()
        fcwt    = (self.cwt_data["freq"] * 1.0).tolist()
        tcwt    = (self.cwt_data["time"] * 1.0).tolist()
        eA_cwt  = [self.cwt_peaks[f][0,2] for f in range(len(fa))]
        ef_cwt  = [self.cwt_peaks[f][0,1] for f in range(len(fa))]
        et_cwt  = [self.cwt_peaks[f][0,0] for f in range(len(fa))]
        
        dictionary = {"event_amp": eA_cwt, "event_freq": ef_cwt,
                      "event_time": et_cwt, "time": tcwt,
                      "A_frequencies": fa, "T_frequencies": fcwt}
    
        return fm.export_dict(dictionary, filename_cwt, path = self.path)
        
    def write_stft(self, filename_stft = "event_data_STFT.json"):
        
        fa       = (self.stft_data["frequencies"] * 1.0).tolist()
        fstft    = (self.stft_data["freq"] * 1.0).tolist()
        tstft    = (self.stft_data["time"] * 1.0).tolist()
        eA_stft  = [self.stft_peaks[f][0,2] for f in range(len(fa))]
        ef_stft  = [self.stft_peaks[f][0,1] for f in range(len(fa))]
        et_stft  = [self.stft_peaks[f][0,0] for f in range(len(fa))]
        
        dictionary = {"event_amp": eA_stft, "event_freq": ef_stft,
                      "event_time": et_stft, "time": tstft,
                      "A_frequencies": fa, "T_frequencies": fstft}
    
        return fm.export_dict(dictionary, filename_stft, path = self.path)
        
        
            
        
        
        
            



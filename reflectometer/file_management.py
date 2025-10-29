# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:26:16 2025

@author: akosk
"""

from json import dump, load
from os import mkdir, listdir
from re import match
from shutil import copy, copy2, move
from os.path import join, exists, isfile
from datetime import datetime
from pathlib import Path
from h5py import File

import numpy as np

# _____________________________________________________________________________
# save the input and output parameters of basic reflectometer
# class into a dictionary
def create_config(ref):
    """ 
    Extracts key input and output parameters from a reflectometer class
    and organizes them into a dictionary.
    Two subdictionary for input (this can be directly used when the
    reflectometer class is created) and output.

    This function is useful for saving or logging simulation configuration
    and results in a structured format, that is directly compatible with
    json extraction.

    Parameters
    ----------
    ref : class
        basic reflectometer class

    Returns
    -------
    dict
        Contains two subdictionary.
        - The input parameters (`ref_input`) include: wave mode, solver type,
        operating frequency, antenna position, beam waist (in SI units),
        angle of incidence, and reflection distance.
        - The output parameters (`ref_output`) include: time and space
        discretization steps (`dt`, `dx`), wavelength, 
        grid dimensions (`nx`, `ny`, `nt`), and the antenna's
        amplitude and phase values.
    """
    
    amp_array, phase_array = ref.get_antenna_output()
    return {'ref_input':
            {'wavemode': ref.wavemode, 'solver': ref.solver,
             'frequency': ref.frequency, 'antenna_pos': ref.antenna_pos,
             'beam_waist': ref.beam_waist_si, 'angle': ref.angle, 
             'reflection_distance': ref.reflection_distance},
            'ref_output':
            {'dt': ref.dt, 'dx': ref.dx, 'wavelength': ref.wavelength,
             'nx': ref.nx, 'ny': ref.ny, 'nt': ref.nt,
             'amplitude': amp_array[0], 'phase': phase_array[0]}}

# save a dictionary into a .json file
def export_dict(dictionary, filename, path = ''):
    """
    It saves a Python dictionary to a `.json` file.
    The function combines the path and filename, writes the dictionary
    to the file in JSON format with an indentation of 4 spaces for
    readability, and returns the full path to the saved file.
    
    
    Parameters
    ----------
    dictionary : dictionary
        The dictionary to be saved. The configuration of the reflectometer
        class, created by "create_config(reflectometer)".
    filename : string
        The name of the output JSON file.
    path : string, optional
        The directory path where the file should be saved.
        Defaults to the current directory (check it with: os.getcwd() ).

    Returns
    -------
    filename : string
        Full filepath of the created config file.
    """
    
    filename = join(path, filename)
    with open(filename, "w") as f:
        dump(dictionary, f, indent=4)
    return filename

# open the dictionary of a .json file
def import_dict(filename, path = ''):
    """
    Open and load the contents of a JSON file into a Python dictionary. 
    The configuration file of the reflectometer class, created by
    "export_dict(reflectometer)".

    Parameters
    ----------
    filename : string
        Name of the JSON file to import.
    path : string, optional
        Directory path to the JSON file. 
        The default is '' (current directory - check it with: os.getcwd()).

    Returns
    -------
    dict
        Dictionary loaded from the JSON file, that contains the configuration
        parameters of a reflectometer class
    """
    
    filename = join(path, filename)
    with open(filename, "r") as f:
        return load(f)

# create a directory for the relevant results (optional: copy files into)
def create_directory(simulation_name, path = '', file = "default", date = False):
    """
    Create a directory for storing simulation results, 
    optionally copying specified files (for example density arrays) into it.

    Parameters
    ----------
    simulation_name : string
        Name of the simulation, used to create a unique directory name.
    path : string, optional
        Base directory where the new directory will be created.
        The default is '' (current directory - check it with: os.getcwd()).
    file : string or list, optional
        File or list of files to copy into the created directory.
        If set to "default", no files are copied.

    Returns
    -------
    string
        Full path to the created directory.
    """
    
    if (date): datetime.now().strftime("%Y%m%d_") + simulation_name
    
    path = join(path, simulation_name)
    try: mkdir(path)
    except: pass
    if isinstance(file, list): # string of filenames should be stored in a list
        for f in file: 
            try: copy(f, path)
            except: print("Incorrectly referenced file:\n", f)
    elif file != "default": copy(file, path)
    return path


def copy_config_files(source_folder, destination_folder, pattern):
    """
    Copy all files matching pattern 'config_####_008.json' from source folder to destination folder.
    
    Args:
        source_folder (str): Path to the source folder
        destination_folder (str): Path to the destination folder
    """
    # Create destination folder if it doesn't exist
    if not exists(destination_folder):
        mkdir(destination_folder)
        print(f"Created destination folder: {destination_folder}")
    
    copied_count = 0
    
    for filename in listdir(source_folder):
        file_path = join(source_folder, filename)
        
        if isfile(file_path) and match(pattern, filename):
            destination_path = join(destination_folder, filename)
            copy2(file_path, destination_path)  # copy2 preserves metadata
            copied_count += 1
            print(f"Copied: {filename}")
    
    print(f"Operation completed. Copied {copied_count} files.")

def move_files(source_folder, destination_folder, pattern):
    """
   Moving files
    """
    # Create destination folder if it doesn't exist
    if not exists(destination_folder):
        mkdir(destination_folder)
    
    moved_count = 0
    
    for filename in listdir(source_folder):
        file_path = join(source_folder, filename)
        
        if isfile(file_path) and match(pattern, filename):
            move(file_path, join(destination_folder, filename))
            moved_count += 1
            print(f"Moved: {filename}")
    
    print(f"Operation completed. Moved {moved_count} files.")


def save_ref_signal(folder, filename = "config_{dens:04d}_{freq:03d}.json",
                    signal_path = "default", signal_filename = "default",
                    time = "default"):
    """
    Extracting the signal field of the reflectometer from the raw config files.
    
    Parameters
    ----------
    filepath : str or pathlib.Path
        Directory containing the configuration JSON files.
    filename : str, optional
        Filename template used to locate configuration files. It must contain
        placeholders for density (`dens`) and frequency (`freq`) indices.
        The default is "config_{dens:04d}_{freq:03d}.json".
    signal_path : str or pathlib.Path, optional
        Output directory for the generated HDF5 signal field file.
        The default is "default", which uses the same directory as `path`.
    signal_filename : str, optional
        Name of the output HDF5 file containing the processed signal data.
        The default is "default", which saves to "signal_field.h5".
    time : array-like or str, optional
        Time values corresponding to the density indices. If "default",
        it is generated as `dens_inds * 1e-6`.
    
    Returns
    -------
    None.
        The function writes the processed amplitude and phase data, along with
        frequency and time information, into an HDF5 file.
    
    """
    path = join(folder, "config_files")
    
    # Collect all matching filenames, find the dens and freq indices
    filenames = sorted([p.name for p in Path(path).glob("config_????_???.json")])
    freq_start, freq_end = int(filenames[0][ 7:11]), int(filenames[-1][ 7:11])+1
    dens_start, dens_end = int(filenames[0][12:15]), int(filenames[-1][12:15])+1
    
    # Read all config data into the configs dictionary
    dens_inds = np.arange(freq_start, freq_end)
    freq_inds = np.arange(dens_start, dens_end)
    n_dens, n_freq = len(dens_inds), len(freq_inds)
    configs = {}
    for dens_ind in dens_inds:
        for freq_ind in freq_inds:
            actual = filename.format(dens=dens_ind, freq=freq_ind)
            configs.update({actual: import_dict(actual, path=path)})
    
    # Convert: frequency index -> frequency values, frames -> time
    freqs = np.linspace(configs[filenames[0]]["ref_input"]["frequency"]/1e9, 
                configs[filenames[-1]]["ref_input"]["frequency"]/1e9, n_freq,
                dtype = np.int16)
    if (time == "default"): time = dens_inds * 1e-6
    
    # Extract the amplitude and phase information into the freq-dens field
    amplitude_array = np.zeros((n_dens, n_freq))
    phase_array = np.zeros((n_dens, n_freq))
    for i, dens_ind in enumerate(dens_inds):
        for j, freq_ind in enumerate(freq_inds):
            actual = configs[filename.format(dens=dens_ind, freq=freq_ind)]
            amplitude_array[i, j] = actual["ref_output"]["amplitude"]
            phase_array[i, j] = actual["ref_output"]["phase"]
    
    # Export the data into a h5 file:
    if (signal_filename == "default"): signal_filename = "signal_field.h5"
    if (signal_path == "default"): signal_path = folder
    signal_file = join(signal_path, signal_filename)
        
    f = File(signal_file, "w")
    f.create_dataset("frames", data = dens_inds, compression="gzip")
    f.create_dataset("time", data = time, compression="gzip")
    f.create_dataset("frequency_indices", data = freq_inds, compression="gzip")
    f.create_dataset("frequencies", data = freqs, compression="gzip")
    f.create_dataset("amplitude_field", data = amplitude_array, compression="gzip")
    f.create_dataset("phase_field", data = phase_array, compression="gzip")



def read_ref_signal(path, signal_filename = "signal_field.h5"):
    """
    Import the created signal field file from HDF5 format.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing the HDF5 signal file.
    signal_filename : str, optional
        Name of the HDF5 file to be read.
        The default is "signal_field.h5".

    Returns
    -------
    dict
        Dictionary where keys correspond to dataset names
        (e.g., "frames", "time", "frequency_indices", "frequencies",
        "amplitude_field", "phase_field") and values are the
        corresponding NumPy arrays.
    """
    file = join(path, signal_filename)
    with File(file, "r") as f:
        return {key: f[key][()] for key in f.keys()}
    

def save_events(working_directory, folder, filename = "default",
                res_file = "default"):
    
    # Export the data into a h5 file:
    if (filename == "default"): filename = "event_data.json"
    if (res_file == "default"): 
        res_path = create_directory("_analysis_", working_directory)
        res_file = join(res_path, "event_table.h5")
    


    keys = ["event_amp", "event_freq", "event_time", 
            "guess_amp", "guess_freq", "guess_time",
            "event_start", "event_end"]
    values = [np.zeros((2,3,3,9)) for _ in range(len(keys))]
    res = dict(zip(keys, values))


    for i in range(2):
        for j in range(3):
            for k in range(3):
                path = Path(working_directory, 
                            folder.format(v=i+1, A=j+1, s=k+1), filename)
                data = import_dict(path)
                
                for key in keys:
                    res[key][i,j,k,:] = data[key]
                
                
    from h5py import File
    f = File(res_file, "w")
    f.create_dataset("frequencies", data = np.array(data["frequencies"]),
                     compression="gzip")

    for key in keys:
        f.create_dataset(key, data = res[key], compression="gzip")
    f.close()
    
    
    
    
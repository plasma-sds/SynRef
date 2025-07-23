# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:26:16 2025

@author: akosk
"""

from json import dump, load
from os import mkdir
from shutil import copy
from os.path import join
from datetime import datetime

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
def create_directory(simulation_name, path = '', file = "default"):
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
    
    path = join(path, datetime.now().strftime("%Y%m%d_") + simulation_name)
    try: mkdir(path)
    except: pass
    if isinstance(file, list): # string of filenames should be stored in a list
        for f in file: 
            try: copy(f, path)
            except: print("Incorrectly referenced file:\n", f)
    elif file != "default": copy(file, path)
    return path
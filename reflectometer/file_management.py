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
# save the input and output parameters of basic class into a dictionary
def create_config(ref):
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
    filename = join(path, filename)
    with open(filename, "w") as f:
        dump(dictionary, f, indent=4)
    return filename

# open the dictionary of a .json file
def import_dict(filename, path = ''):
    filename = join(path, filename)
    with open(filename, "r") as f:
        return load(f)

# create a directory for the relevant results (optional: copy files into)
def create_directory(simulation_name, path = '', file = "default"):
    path = join(path, datetime.now().strftime("%Y%m%d_") + simulation_name)
    try: mkdir(path)
    except: pass
    if isinstance(file, list): # string of filenames should be stored in a list
        for f in file: 
            try: copy(f, path)
            except: print("Incorrectly referenced file:\n", f)
    elif file != "default": copy(file, path)
    return path
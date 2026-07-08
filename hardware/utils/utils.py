"""Utility functions for noise temperature and gain conversions."""
import numpy as np
import h5py
from dataclasses import dataclass

T0 = 290.0  # K
kB = 1.380649e-23  # Boltzmann constant J/K
c = 299792458.0  # Speed of light m/s

def wave_number(wavelength_m: float) -> float:
    """Calculate wave number k = 2 * pi / wavelength."""
    return 2 * np.pi / wavelength_m

def wavelength(frequency_hz: float) -> float:
    """Calculate wavelength from frequency."""
    return c / frequency_hz

def frequency(wavelength_m: float) -> float:
    """Calculate frequency from wavelength."""
    return c / wavelength_m

def db_to_lin(x_db: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (x_db / 10)

def lin_to_db(x: float) -> float:
    """Convert linear to dB scale."""
    return 10 * np.log10(x)

def W_to_dBm(x: float) -> float:
    """Convert power in W to dBm."""
    return 10 * np.log10(x * 1000)

def dBm_to_W(x: float) -> float:
    """Convert power in dBm to W."""
    return 10 ** (x / 10) / 1000

def db_m_to_nepers_m(x_db_m: float) -> float:
    """Convert attenuation in dB/m to Np/m."""
    return x_db_m / (20 * np.log10(np.e))

def nf_db_to_te(nf_db: float, t0: float = T0) -> float:
    """Convert noise figure (dB) to equivalent noise temperature (K)."""
    F = db_to_lin(nf_db)
    return (F - 1.0) * t0

def thermal_noise_rms_voltage(noise_temp_k: float, bandwidth_hz: float, resistance_ohm: float) -> float:
    """Thermal noise voltage from temperature, bandwidth and resistance.""" 
    return np.sqrt(4 * kB * noise_temp_k * bandwidth_hz * resistance_ohm)

def thermal_noise_rms_power(noise_temp_k: float, bandwidth_hz: float) -> float:
    """Thermal noise power from temperature and bandwidth.""" 
    return kB * noise_temp_k * bandwidth_hz

def aperture_discretication(a: float, b: float, num_points_a: int, num_points_b: int) -> (np.ndarray, np.ndarray):
    """Discretize rectangular aperture into grid of points."""
    Nx, Ny = 100, 80
    x = np.linspace(-a/2, a/2, Nx)
    y = np.linspace(-b/2, b/2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y

def field_to_open_circuit_voltage(E_field_vm: np.ndarray, effective_aperture_m2: float, resistance_ohm: float) -> np.ndarray:
    """
    Convert 2D wavefront E-field (V/m) to open-circuit voltage (V). Assumes conjugate matching for maximum power transfer.
    
    Args:
        E_field_vm: Incident electric field (V/m), scalar or array
        effective_aperture_m2: Effective aperture area (m^2)
        resistance_ohm: Resistance (Ohm), assuming conjugate matching for maximum power transfer
        
    Returns:
        Open-circuit voltage V_oc = E * sqrt(A_e * R_A / (60 * pi))
    """
    return E_field_vm * np.sqrt(4 * effective_aperture_m2 * resistance_ohm / (2 *120 * np.pi))

def complex_awgn(n_samples: int, noise_power_w: float, rng=None):
    """Generate complex additive white Gaussian noise."""
    rng = np.random.default_rng(rng)
    sigma = np.sqrt(noise_power_w / 2.0)  
    return sigma * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))

def keV_to_K(x: float) -> float:
    """Convert energy in keV to temperature in K."""
    return x * 11604.5250061657

def walk_h5(group, indent=0):
    for name, item in group.items():
        prefix = " " * indent
        if isinstance(item, h5py.Group):
            print(f"{prefix}Group: {group.name}/{name}")
            walk_h5(item, indent + 2)
        else:
            print(f"{prefix}Dataset: {group.name}/{name}, shape={item.shape}, dtype={item.dtype}")


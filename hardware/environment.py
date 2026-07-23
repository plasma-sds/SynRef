"""Environment model as noisy source. Workaround until proper modelling is implemented."""
from dataclasses import dataclass, field
from hardware.utils.utils import keV_to_K, thermal_noise_rms_voltage, complex_awgn, wavelength, wave_number
import numpy as np

@dataclass
class Environment:
    """Environment modeled as noisy source.
    inputs: 
    - physical temperature in keV,
    - polarization loss factor,
    - resistance in ohm,
    - source position in lab frame (x, y, z) in meters
    output: noise voltage, noise power, E-field distribution across aperture"""
    name: str
    plasma_physical_temp_keV: float = 4  # Physical temperature of environment
    PLF: float = 0.5 # Polarization loss factor (0 to 1)
    resistance_ohm: float = 2.0
    source_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) 
        
    @property
    def noise_temp_k(self):
        return keV_to_K(self.plasma_physical_temp_keV) 

    def generate_noise(self, n_samples: int, receiver_RBW_hz: float, rng=None) -> np.ndarray:
        """Generate antenna thermal noise contribution."""
        noise_voltage = thermal_noise_rms_voltage(self.noise_temp_k, receiver_RBW_hz, resistance_ohm=self.resistance_ohm)
        return complex_awgn(n_samples, noise_voltage, rng) * self.PLF
    

        

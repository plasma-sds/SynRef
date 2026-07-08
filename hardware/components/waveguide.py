"""Passive waveguide/cable model."""
from dataclasses import dataclass
from utils import db_to_lin, db_m_to_nepers_m, thermal_noise_rms_power, thermal_noise_rms_voltage, complex_awgn, T0
import numpy as np

@dataclass
class Waveguide:
    """Passive lossy element with temperature-dependent noise."""
    name: str
    physical_temp_k: float = T0
    attenuation_db_m: float = 0.5  # Attenuation in dB/m
    length_m: float = 1.0  # Length of waveguide in meters 
    


    @property
    def attenuation_Np_m(self):
        """Attenuation in Np/m."""
        return db_m_to_nepers_m(self.attenuation_db_m)
    
    @property
    def efficiency(self):
        """Waveguide efficiency (0 to 1) from loss."""
        return np.e ** (-self.attenuation_Np_m * self.length_m)
    
    @property
    def gain_lin(self):
        """Linear gain (loss) of waveguide."""
        return self.efficiency
    
    @property
    def loss_factor(self):
        """Loss factor L = 1 / efficiency (attenuation due to losses)."""
        return 1.0 / self.efficiency
    
    @property
    def input_noise_temp_k(self):
        """Equivalent input noise temperature due to physical temperature and losses"""
        return self.physical_temp_k * (self.loss_factor - 1.0)
        
    @property
    def output_noise_temp_k(self):
        """Equivalent output noise temperature"""
        return self.physical_temp_k * (1.0 - self.efficiency)
    
    def generate_noise(self, n_samples: int, receiver_RBW_hz: float, rng=None) -> np.ndarray:
        """Generate antenna thermal noise contribution."""
        noise_voltage = thermal_noise_rms_voltage(self.output_noise_temp_k, receiver_RBW_hz, resistance_ohm=50.0)
        return complex_awgn(n_samples, noise_voltage, rng)
    
    def apply_gain(self, signal: np.ndarray) -> np.ndarray:
        """Apply waveguide loss to signal."""
        return np.sqrt(self.gain_lin) * signal  
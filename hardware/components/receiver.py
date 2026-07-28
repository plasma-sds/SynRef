"""Low Noise Amplifier model."""
from dataclasses import dataclass
from hardware.utils.utils import db_to_lin, nf_db_to_te, thermal_noise_rms_power, thermal_noise_rms_voltage, complex_awgn, T0
import numpy as np

@dataclass
class Receiver:
    """Active amplifier with noise figure."""
    name: str
    gain_db: float
    noise_figure_db: float
    RBW_hz: float = 1e6
    sampling_rate_hz: float = 12e6 
    rfl_freq_hz: float = 30e9
    nquist_margin: float = 2


    @property
    def gain_lin(self):
        return db_to_lin(self.gain_db)
    
    @property
    def input_noise_temp_k(self):
        """Equivalent input noise temperature."""
        return nf_db_to_te(self.noise_figure_db, T0)

    @property
    def output_noise_temp_k(self):
        """Equivalent output noise temperature."""
        return self.input_noise_temp_k * self.gain_lin
    
    @property
    def duration_s(self):
        """Duration to achieve desired frequency resolution, assuming Hanning window."""
        return 1.44 / self.RBW_hz
    
    @property
    def n_samples(self):
        """Number of samples for measurement."""
        return int(self.sampling_rate_hz * self.duration_s)
    
    def generate_noise(self, n_samples: int, receiver_RBW_hz: float, rng=None) -> np.ndarray:
        """Generate antenna thermal noise contribution."""
        noise_voltage = thermal_noise_rms_voltage(self.output_noise_temp_k, self.RBW_hz, resistance_ohm=50.0)
        return complex_awgn(n_samples, noise_voltage, rng)
    
    def apply_gain(self, signal: np.ndarray) -> np.ndarray:
        """Apply LNA gain to signal."""
        return np.sqrt(self.gain_lin) * signal
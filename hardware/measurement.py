"""End-to-end reflectometry measurement with per-component noise."""
import numpy as np
from typing import List, Dict
from signal_own import ideal_reflectometry_trace, sinusoidal_signal
from scipy import signal as sgn
from hardware.environment import Environment
from hardware.components import Antenna, Waveguide, Receiver
from hardware.utils.utils import thermal_noise_rms_power
from hardware.utils.antenna.antenna_direction_angles import antenna_look_angles
from hardware.utils.antenna.pyramidal_farfield_to_fw2d import pyramidal_farfield_to_fw2d

class ReflectometrySystem:
    def __init__(self, blocks: Dict, rng=None):
        self.blocks = blocks
        self.rng = np.random.default_rng(rng)
    
    def measure(self, n_samples: int, sampling_rate_hz: int, rfl_freq_hz: float = 20e9, environment: Environment = None, duration_s: float = 1.0, E_field_vm: float = 1.0, A: float = 1.0) -> np.ndarray:
        
        point_source = False
        if point_source:
            theta, phi = antenna_look_angles(
                ant_pos=self.blocks["antenna"].ant_pos,
                boresight=self.blocks["antenna"].boresight,
                e_plane_up=self.blocks["antenna"].e_plane_up,
                src_pos=environment.source_pos
            )
            print(f"Antenna look angles: theta={np.degrees(theta):.2f} deg, phi={np.degrees(phi):.2f} deg")
        
        """Generate complete noisy measurement."""
        artificial_signal = False
        if artificial_signal:
            # 1. Generate ideal signal
            #ideal_signal = ideal_reflectometry_trace(n_samples, center_freq_hz, self.bandwidth_hz)
            #ideal_signal = sgn.chirp(t=np.linspace(0, duration_s, n_samples), f0=wave_freq_hz, f1=wave_freq_hz*2, t1=duration_s/2, method='linear', complex=True)
            ideal_signal = sinusoidal_signal(fs=sampling_rate_hz, f=rfl_freq_hz, A=A, phi=0, duration=duration_s)  
            #ideal_signal *=  self.blocks["antenna"].wavefront_to_voltage(E_field_vm, wave_freq_hz)
            
            # 2. Generate noise
            signal = ideal_signal
            total_noise = environment.generate_noise(n_samples, receiver_RBW_hz=self.blocks["receiver"].RBW_hz, rng=self.rng)

        amp, phase = pyramidal_farfield_to_fw2d(
            ny=100, 
            dy=0.5 * (3e8 / rfl_freq_hz), 
            dx=0.5 * (3e8 / rfl_freq_hz),
            x_horn=self.blocks["antenna"].ant_pos[0],
            y_horn=self.blocks["antenna"].ant_pos[1],
            angle=np.radians(0.0),
            a1=self.blocks["antenna"].a1,
            b1=self.blocks["antenna"].b1,
            rho1=self.blocks["antenna"].rho1,
            rho2=self.blocks["antenna"].rho2,
            freq=self.blocks["receiver"].rfl_freq_hz
            E1=self.blocks["antenna"].E1
        )
              
        # 3. Cascade each block with individual noise contribution
        input_gain_product = 1.0
        for name, block in self.blocks.items():
            # Add block-specific noise (Friis referred to output)
            if isinstance(block, Antenna):
                # For antenna, we need to consider the signal transformation from E-field to voltage
                signal = block.apply_gain(signal = signal, rfl_freq_hz = self.blocks["receiver"].rfl_freq_hz, theta=theta, phi=phi)  # Apply antenna gain to signal
                total_noise = block.apply_gain(total_noise, rfl_freq_hz=self.blocks["receiver"].rfl_freq_hz, theta=theta, phi=phi)  # Scale existing noise by antenna gain
                block_noise = block.generate_noise(n_samples, self.blocks["receiver"].RBW_hz, self.rng)
                total_noise += block_noise
            else:
                total_noise = block.apply_gain(total_noise)  # Scale existing noise by block gain
                block_noise = block.generate_noise(n_samples, self.blocks["receiver"].RBW_hz, self.rng)
                total_noise += block_noise
                # Apply block gain to signal
                signal = block.apply_gain(signal)
            
        
        return signal + total_noise, ideal_signal, signal, total_noise